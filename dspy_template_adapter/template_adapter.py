"""
TemplateAdapter — A DSPy Adapter that enables exact-fidelity prompt templates.

Instead of generating prompts from signature metadata (like ChatAdapter), this adapter
renders user-provided message templates with variable interpolation and template functions,
while preserving DSPy's full pipeline: preprocess (tool calling, native reasoning),
custom type splitting, output parsing, tracing, evaluation, and optimization.

Designed for upstream contribution to ``dspy.adapters.template_adapter``.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import warnings
from typing import TYPE_CHECKING, Any, Callable

import json_repair
import pydantic
import regex

from dspy.adapters.base import Adapter
from dspy.adapters.types.base_type import split_message_content_for_custom_types
from dspy.adapters.types.history import History
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    serialize_for_json,
)
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

if TYPE_CHECKING:
    from dspy.adapters.types.base_type import Type
    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe dict for partial str.format_map — unknown keys pass through unchanged
# ---------------------------------------------------------------------------

_ESCAPED_BRACE_OPEN = "\x00LBRACE\x00"
_ESCAPED_BRACE_CLOSE = "\x00RBRACE\x00"


class _SafeDict(dict):
    """Dict subclass that returns ``{key}`` for missing keys."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


# ---------------------------------------------------------------------------
# Template function regex — matches {func_name()} and {func_name(kw=val, …)}
# ---------------------------------------------------------------------------

_TEMPLATE_FUNC_RE = re.compile(r"\{(\w+)\(([^)]*)\)\}")


# ---------------------------------------------------------------------------
# TemplateAdapter
# ---------------------------------------------------------------------------


class TemplateAdapter(Adapter):
    """A DSPy Adapter that uses user-provided message templates.

    The messages **are** the prompt — no hidden rewriting.  The signature defines the I/O
    contract (field names, types) and is used only for:

    * ``{field_name}`` interpolation (input field values)
    * ``{instruction}`` — resolves to ``signature.instructions`` (optimisable by MIPRO/COPRO)
    * Template functions: ``{inputs()}``, ``{outputs()}``, ``{demos()}``
    * Message directives: ``{"role": "demos"}``, ``{"role": "history"}``
    * Output parsing (``json``, ``full_text``, ``chat``, or a custom callable)

    Parameters
    ----------
    messages : list[dict]
        The message template list.  Each element is either a regular message
        (``{"role": "system"|"user"|"assistant", "content": "…"}``) or a
        **message directive** (``{"role": "demos", …}`` or ``{"role": "history"}``).
    parse_mode : str | Callable
        How to parse the LM completion into output fields:
        * ``"json"`` — extract a JSON object and map keys to output fields (default).
        * ``"full_text"`` — map the entire completion to the single output field.
        * ``"xml"`` — extract ``<field_name>value</field_name>`` tags.
        * ``"chat"`` — delegate to ``ChatAdapter.parse()``.
        * A callable ``(signature, completion) -> dict[str, Any]``.
    callbacks : list[BaseCallback] | None
        Passed to the base ``Adapter``.
    use_native_function_calling : bool
        Passed to the base ``Adapter``.
    native_response_types : list[type[Type]] | None
        Passed to the base ``Adapter``.
    """

    def __init__(
        self,
        messages: list[dict[str, Any]],
        parse_mode: str | Callable = "json",
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type] | None = None,
    ):
        # DSPy changed Adapter.__init__ across versions. Build kwargs dynamically so
        # this adapter works with both old and new signatures.
        adapter_init_params = inspect.signature(Adapter.__init__).parameters
        adapter_kwargs: dict[str, Any] = {}
        if "callbacks" in adapter_init_params:
            adapter_kwargs["callbacks"] = callbacks
        if "use_native_function_calling" in adapter_init_params:
            adapter_kwargs["use_native_function_calling"] = use_native_function_calling
        if "native_response_types" in adapter_init_params:
            adapter_kwargs["native_response_types"] = native_response_types

        super().__init__(**adapter_kwargs)

        if not messages:
            raise ValueError("TemplateAdapter requires at least one message template.")
        self.message_templates = messages
        self.parse_mode = parse_mode
        self._custom_helpers: dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def register_helper(self, name: str, fn: Callable) -> None:
        """Register a custom template function callable as ``{name()}`` in templates."""
        self._custom_helpers[name] = fn

    def preview(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the rendered messages **without** calling the LM.

        Useful for debugging and fidelity verification.
        """
        return self.format(signature, demos or [], inputs or {})

    def _get_history_field_name(self, signature: type[Signature]) -> str | None:
        """Find the history field name with a local fallback.

        We prefer the base Adapter implementation when available for compatibility
        with DSPy internals, but keep a local fallback to avoid breakage if DSPy
        refactors the base method name/signature.
        """
        base_method = getattr(super(), "_get_history_field_name", None)
        if callable(base_method):
            try:
                return base_method(signature)
            except Exception:
                pass

        for name, field in signature.input_fields.items():
            if field.annotation == History:
                return name
        return None

    # ------------------------------------------------------------------
    # format()  — the core override
    # ------------------------------------------------------------------

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        **kwargs,
    ) -> list[dict[str, Any]]:
        inputs = dict(inputs)  # shallow copy — we may mutate (pop history)

        # --- Extract History if present ---
        history_messages: list[dict[str, Any]] = []
        history_obj: History | None = None
        history_field_name = self._get_history_field_name(signature)
        if history_field_name and history_field_name in inputs:
            raw_history = inputs.pop(history_field_name)
            if isinstance(raw_history, History):
                history_obj = raw_history
                history_messages = self._expand_history(signature, raw_history)
            elif isinstance(raw_history, list):
                history_obj = History(messages=raw_history)
                history_messages = raw_history

        # --- Build the rendering context ---
        ctx = self._build_context(signature, demos, inputs, history_obj)

        # --- Walk message templates, expanding directives ---
        rendered: list[dict[str, Any]] = []
        demos_injected = False
        history_consumed = False

        for tmpl in self.message_templates:
            role = tmpl.get("role", "")

            # ── Directive: demos ──
            if role == "demos":
                rendered.extend(self._expand_demos_directive(tmpl, signature, demos, ctx))
                demos_injected = True
                continue

            # ── Directive: history ──
            if role == "history":
                rendered.extend(history_messages)
                continue

            # ── Regular message ──
            content = tmpl.get("content", "")
            rendered_content, used_demos, used_history = self._render(
                content, ctx, signature, demos, history_obj
            )
            if used_demos:
                demos_injected = True
            if used_history:
                history_consumed = True
            rendered.append({"role": role, "content": rendered_content})

        # --- Auto-inject demos if not consumed by template ---
        if demos and not demos_injected:
            demo_msgs = self._format_demos_as_messages(signature, demos)
            # Insert before the last user message
            last_user_idx = None
            for i in range(len(rendered) - 1, -1, -1):
                if rendered[i].get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx is not None:
                rendered = rendered[:last_user_idx] + demo_msgs + rendered[last_user_idx:]
            else:
                rendered = demo_msgs + rendered

        # --- Inject conversation history if not consumed by template ---
        if history_messages and not history_consumed and not any(
            t.get("role") == "history" for t in self.message_templates
        ):
            last_user_idx = None
            for i in range(len(rendered) - 1, -1, -1):
                if rendered[i].get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx is not None:
                rendered = rendered[:last_user_idx] + history_messages + rendered[last_user_idx:]

        # --- Split custom type markers (Image, Audio, File, etc.) ---
        rendered = split_message_content_for_custom_types(rendered)

        return rendered

    # ------------------------------------------------------------------
    # parse()
    # ------------------------------------------------------------------

    def parse(self, signature: type[Signature], completion: str, **kwargs) -> dict[str, Any]:
        if callable(self.parse_mode) and not isinstance(self.parse_mode, str):
            return self.parse_mode(signature, completion)

        if self.parse_mode == "full_text":
            return self._parse_full_text(signature, completion)
        if self.parse_mode == "chat":
            return self._parse_chat(signature, completion)
        if self.parse_mode == "xml":
            return self._parse_xml(signature, self._strip_markdown_code_fences(completion))
        # Default: json
        return self._parse_json(signature, self._strip_markdown_code_fences(completion))

    # ------------------------------------------------------------------
    # format_finetune_data()
    # ------------------------------------------------------------------

    def format_finetune_data(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        **kwargs,
    ) -> dict[str, list[Any]]:
        messages = self.format(signature=signature, demos=demos, inputs=inputs)
        # Build assistant message matching parse_mode
        if self.parse_mode == "full_text" and len(signature.output_fields) == 1:
            key = next(iter(signature.output_fields))
            assistant_content = str(outputs.get(key, ""))
        elif self.parse_mode == "xml":
            parts = []
            for k, v in outputs.items():
                parts.append(f"<{k}>{v}</{k}>")
            assistant_content = "\n".join(parts)
        elif self.parse_mode == "json":
            assistant_content = json.dumps(
                {k: serialize_for_json(v) for k, v in outputs.items()}, indent=2
            )
        else:
            assistant_content = json.dumps(
                {k: serialize_for_json(v) for k, v in outputs.items()}, indent=2
            )
        messages.append({"role": "assistant", "content": assistant_content})
        return {"messages": messages}

    # ------------------------------------------------------------------
    # Required abstract method stubs (not used — we override format() directly)
    # ------------------------------------------------------------------

    def format_field_description(self, signature: type[Signature]) -> str:
        return ""

    def format_field_structure(self, signature: type[Signature]) -> str:
        return ""

    def format_task_description(self, signature: type[Signature]) -> str:
        return ""

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        # Used by base class format_demos() — render inputs as simple key: value
        parts = [prefix] if prefix else []
        for name in signature.input_fields:
            if name in inputs:
                parts.append(f"{name}: {format_field_value(signature.input_fields[name], inputs[name])}")
        if suffix:
            parts.append(suffix)
        return "\n".join(parts).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        # Used by base class format_demos() — render outputs as JSON
        d = {}
        for name in signature.output_fields:
            d[name] = outputs.get(name, missing_field_message)
        return json.dumps(serialize_for_json(d), indent=2)

    # ==================================================================
    # Private implementation
    # ==================================================================

    # --- Context building ---

    def _build_context(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        history: History | None = None,
    ) -> dict[str, Any]:
        """Build the variable context for template rendering."""
        ctx: dict[str, Any] = {}
        # Input field values — formatted through DSPy's serialization
        for name, field_info in signature.input_fields.items():
            if name in inputs:
                ctx[name] = format_field_value(field_info, inputs[name])
            else:
                ctx[name] = ""
        # The optimisable instruction slot
        ctx["instruction"] = signature.instructions or ""
        # Optional history object for custom helpers / {history()} template function
        ctx["history"] = history if history is not None else ""
        return ctx

    # --- Template rendering ---

    def _render(
        self,
        template: str,
        ctx: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        history: History | None = None,
    ) -> tuple[str, bool, bool]:
        """Two-pass rendering: template functions, then variable interpolation.

        Returns:
            rendered_text, used_demos, used_history
        """
        # Escape literal double braces {{ / }} so str.format_map doesn't consume them
        text = template.replace("{{", _ESCAPED_BRACE_OPEN).replace("}}", _ESCAPED_BRACE_CLOSE)

        # Pass 1: template functions  {func_name(...)}
        text, used_demos, used_history = self._eval_template_functions(
            text, ctx, signature, demos, history
        )

        # Pass 2: simple variable interpolation  {field_name}
        text = text.format_map(_SafeDict(ctx))

        # Restore escaped braces
        text = text.replace(_ESCAPED_BRACE_OPEN, "{").replace(_ESCAPED_BRACE_CLOSE, "}")
        return text, used_demos, used_history

    def _eval_template_functions(
        self,
        text: str,
        ctx: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        history: History | None = None,
    ) -> tuple[str, bool, bool]:
        used_demos = False
        used_history = False

        def _replace(match: re.Match) -> str:
            nonlocal used_demos, used_history

            func_name = match.group(1)
            raw_args = match.group(2).strip()
            kwargs = _parse_func_kwargs(raw_args) if raw_args else {}

            result: str | None = None
            if func_name == "inputs":
                result = self._render_inputs(ctx, signature, **kwargs)
            elif func_name == "outputs":
                result = self._render_outputs(signature, **kwargs)
            elif func_name == "demos":
                used_demos = True
                result = self._render_demos_inline(demos, signature, **kwargs)
            elif func_name == "history":
                used_history = True
                result = self._render_history_inline(history, signature, **kwargs)
            elif func_name in self._custom_helpers:
                result = str(self._custom_helpers[func_name](ctx=ctx, signature=signature, demos=demos, **kwargs))

            if result is not None:
                # Escape braces in the output so str.format_map in pass 2 doesn't choke
                return result.replace("{", _ESCAPED_BRACE_OPEN).replace("}", _ESCAPED_BRACE_CLOSE)

            # Unknown function — leave untouched
            return match.group(0)

        new_text = _TEMPLATE_FUNC_RE.sub(_replace, text)
        return new_text, used_demos, used_history

    # --- Template functions ---

    @staticmethod
    def _render_inputs(ctx: dict[str, Any], signature: type[Signature], style: str = "default", **kwargs) -> str:
        parts = []
        for name, field_info in signature.input_fields.items():
            # History is rendered via {history(...)} or {"role": "history"}, not {inputs()}
            if field_info.annotation == History:
                continue
            val = ctx.get(name, "")
            if style == "xml":
                parts.append(f"<{name}>{val}</{name}>")
            elif style == "yaml":
                parts.append(f"{name}: {val}")
            elif style == "json":
                parts.append(f'"{name}": {json.dumps(val, ensure_ascii=False)}')
            else:
                parts.append(f"{name}: {val}")
        if style == "json":
            return "{\n  " + ",\n  ".join(parts) + "\n}"
        return "\n".join(parts)

    @staticmethod
    def _render_outputs(signature: type[Signature], style: str = "default", **kwargs) -> str:
        if style == "schema":
            schema: dict[str, Any] = {}
            for name, field in signature.output_fields.items():
                try:
                    schema[name] = pydantic.TypeAdapter(field.annotation).json_schema()
                except Exception:
                    schema[name] = {"type": get_annotation_name(field.annotation)}
            return json.dumps(schema, indent=2, ensure_ascii=False)

        if style == "xml":
            wrap = kwargs.get("wrap", "")
            indent_str = "  " if wrap else ""
            parts = []
            for name, field in signature.output_fields.items():
                desc = (getattr(field, "json_schema_extra", None) or {}).get("desc", name)
                parts.append(f"{indent_str}<{name}>{desc}</{name}>")
            body = "\n".join(parts)
            if wrap:
                return f"<{wrap}>\n{body}\n</{wrap}>"
            return body

        # Default: numbered list
        return get_field_description_string(signature.output_fields)

    @staticmethod
    def _render_demos_inline(
        demos: list[dict[str, Any]],
        signature: type[Signature],
        style: str = "default",
        **kwargs,
    ) -> str:
        if not demos:
            return ""
        rendered = []
        all_fields = list(signature.input_fields) + list(signature.output_fields)
        for i, demo in enumerate(demos):
            if style == "xml":
                lines = []
                for k in all_fields:
                    if k in demo:
                        lines.append(f"<{k}>{demo[k]}</{k}>")
                rendered.append("\n".join(lines))
            elif style == "json":
                d = {k: demo[k] for k in all_fields if k in demo}
                rendered.append(json.dumps(serialize_for_json(d), indent=2, ensure_ascii=False))
            elif style == "yaml":
                lines = []
                for k in all_fields:
                    if k in demo:
                        lines.append(f"{k}: {demo[k]}")
                rendered.append("\n".join(lines))
            else:
                lines = [f"Example {i + 1}:"]
                for k in all_fields:
                    if k in demo:
                        lines.append(f"  {k}: {demo[k]}")
                rendered.append("\n".join(lines))
        return "\n\n".join(rendered)

    @staticmethod
    def _render_history_inline(
        history: History | None,
        signature: type[Signature],
        style: str = "default",
        **kwargs,
    ) -> str:
        """Render dspy.History inline inside a message via {history(...)}."""
        if history is None or not history.messages:
            return ""

        messages = history.messages

        if style == "json":
            return json.dumps(serialize_for_json(messages), indent=2, ensure_ascii=False)

        if style == "yaml":
            blocks = []
            for i, msg in enumerate(messages, 1):
                lines = [f"- turn: {i}"]
                for k, v in msg.items():
                    lines.append(f"  {k}: {v}")
                blocks.append("\n".join(lines))
            return "\n".join(blocks)

        if style == "xml":
            blocks = []
            for msg in messages:
                fields = []
                for k, v in msg.items():
                    fields.append(f"  <{k}>{v}</{k}>")
                blocks.append("<turn>\n" + "\n".join(fields) + "\n</turn>")
            return "\n".join(blocks)

        # default style
        blocks = []
        for i, msg in enumerate(messages, 1):
            lines = [f"Turn {i}:"]
            for k, v in msg.items():
                lines.append(f"  {k}: {v}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # --- Directive expansion ---

    def _expand_demos_directive(
        self,
        directive: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        ctx: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not demos:
            return []

        # Custom demo template: {"role": "demos", "user": "…", "assistant": "…"}
        if "user" in directive and "assistant" in directive:
            messages = []
            all_fields = list(signature.input_fields) + list(signature.output_fields)
            for demo in demos:
                demo_ctx = {k: str(demo.get(k, "")) for k in all_fields}
                demo_ctx["outputs_json"] = json.dumps(
                    serialize_for_json({k: demo.get(k, "") for k in signature.output_fields}),
                    ensure_ascii=False,
                )
                messages.append({
                    "role": "user",
                    "content": directive["user"].format_map(_SafeDict(demo_ctx)),
                })
                messages.append({
                    "role": "assistant",
                    "content": directive["assistant"].format_map(_SafeDict(demo_ctx)),
                })
            return messages

        # Default expansion: use base adapter's format_demos (input→user, output→assistant)
        return self._format_demos_as_messages(signature, demos)

    def _format_demos_as_messages(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format demos as user/assistant message pairs.

        The assistant message format matches ``parse_mode``:
        * ``"xml"`` → ``<field>value</field>`` tags
        * ``"full_text"`` → raw value (single output field only)
        * Otherwise → JSON object
        """
        messages = []
        for demo in demos:
            # User message: input fields
            user_parts = []
            for name in signature.input_fields:
                if name in demo:
                    user_parts.append(
                        f"{name}: {format_field_value(signature.input_fields[name], demo[name])}"
                    )
            if not user_parts:
                continue
            messages.append({"role": "user", "content": "\n".join(user_parts)})

            # Assistant message: output fields — format matches parse_mode
            out = {}
            for name in signature.output_fields:
                if name in demo:
                    out[name] = demo[name]

            # Only append assistant message if we actually have output data
            if out:
                if self.parse_mode == "xml":
                    parts = [f"<{k}>{v}</{k}>" for k, v in out.items()]
                    messages.append({"role": "assistant", "content": "\n".join(parts)})
                elif self.parse_mode == "full_text" and len(out) == 1:
                    messages.append({
                        "role": "assistant",
                        "content": str(next(iter(out.values()))),
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(serialize_for_json(out), indent=2, ensure_ascii=False),
                    })
        return messages

    # --- History expansion ---

    def _expand_history(
        self,
        signature: type[Signature],
        history: History,
    ) -> list[dict[str, Any]]:
        """Expand dspy.History into user/assistant message pairs.

        Assistant turn formatting follows parse_mode for consistency with demos:
        - xml      -> XML tags
        - full_text (single output field) -> raw text
        - otherwise -> JSON object
        """
        messages = []
        for msg in history.messages:
            # User message: input fields from the history entry
            user_parts = []
            for name, field_info in signature.input_fields.items():
                # Never render nested history objects in history expansion
                if field_info.annotation == History:
                    continue
                if name in msg:
                    user_parts.append(f"{name}: {msg[name]}")
            if user_parts:
                messages.append({"role": "user", "content": "\n".join(user_parts)})

            # Assistant message: output fields from the history entry
            out_parts = {}
            for name in signature.output_fields:
                if name in msg:
                    out_parts[name] = msg[name]
            if out_parts:
                if self.parse_mode == "xml":
                    parts = [f"<{k}>{v}</{k}>" for k, v in out_parts.items()]
                    messages.append({"role": "assistant", "content": "\n".join(parts)})
                elif self.parse_mode == "full_text" and len(out_parts) == 1:
                    messages.append({
                        "role": "assistant",
                        "content": str(next(iter(out_parts.values()))),
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(serialize_for_json(out_parts), indent=2, ensure_ascii=False),
                    })
        return messages

    # --- Parsers ---

    @staticmethod
    def _strip_markdown_code_fences(text: str) -> str:
        """Strip common markdown code fences around model output.

        Handles patterns like:
            ```json\n{...}\n```
            ```xml\n<answer>...</answer>\n```
            ```\n...\n```
        """
        stripped = text.strip()
        if not stripped.startswith("```"):
            return text

        # Remove first fence line
        lines = stripped.splitlines()
        if not lines:
            return text
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove trailing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _parse_json(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Extract a JSON object from the completion and map to output fields."""
        fields = json_repair.loads(completion)

        if not isinstance(fields, dict):
            # Try to find a JSON object with recursive regex
            match = regex.search(r"\{(?:[^{}]|(?R))*\}", completion, regex.DOTALL)
            if match:
                fields = json_repair.loads(match.group(0))

        if not isinstance(fields, dict):
            raise AdapterParseError(
                adapter_name="TemplateAdapter",
                signature=signature,
                lm_response=completion,
                message="No JSON object found in LM response.",
            )

        # Keep only known output fields and cast to expected types
        parsed = {}
        for name, field_info in signature.output_fields.items():
            if name in fields:
                try:
                    parsed[name] = parse_value(fields[name], field_info.annotation)
                except Exception as e:
                    raise AdapterParseError(
                        adapter_name="TemplateAdapter",
                        signature=signature,
                        lm_response=completion,
                        message=f"Failed to parse field '{name}': {e}",
                    )

        if parsed.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="TemplateAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=parsed,
            )
        return parsed

    @staticmethod
    def _parse_full_text(signature: type[Signature], completion: str) -> dict[str, Any]:
        """Map the entire completion to the single output field."""
        output_keys = list(signature.output_fields.keys())
        if len(output_keys) != 1:
            raise AdapterParseError(
                adapter_name="TemplateAdapter",
                signature=signature,
                lm_response=completion,
                message=(
                    f"parse_mode='full_text' requires exactly 1 output field, "
                    f"got {len(output_keys)}: {output_keys}"
                ),
            )
        key = output_keys[0]
        annotation = signature.output_fields[key].annotation
        return {key: parse_value(completion.strip(), annotation)}

    @staticmethod
    def _parse_xml(signature: type[Signature], completion: str) -> dict[str, Any]:
        """Extract output fields from XML tags like ``<field_name>value</field_name>``.

        Handles:
        * Tags anywhere in the completion (surrounding text is ignored).
        * Whitespace inside tags is stripped.
        * Missing required fields raise ``AdapterParseError``.
        """
        parsed: dict[str, Any] = {}
        for name, field_info in signature.output_fields.items():
            # Match <name>...</name> (DOTALL so value can span multiple lines)
            pattern = re.compile(rf"<{re.escape(name)}>(.*?)</{re.escape(name)}>", re.DOTALL)
            match = pattern.search(completion)
            if match:
                raw_value = match.group(1).strip()
                try:
                    parsed[name] = parse_value(raw_value, field_info.annotation)
                except Exception as e:
                    raise AdapterParseError(
                        adapter_name="TemplateAdapter",
                        signature=signature,
                        lm_response=completion,
                        message=f"Failed to parse XML field '{name}': {e}",
                    )

        if parsed.keys() != signature.output_fields.keys():
            missing = set(signature.output_fields.keys()) - set(parsed.keys())
            raise AdapterParseError(
                adapter_name="TemplateAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=parsed,
                message=f"Missing XML tags for output fields: {missing}",
            )
        return parsed

    @staticmethod
    def _parse_chat(signature: type[Signature], completion: str) -> dict[str, Any]:
        """Delegate to ChatAdapter's parsing (``[[ ## field ## ]]`` markers)."""
        from dspy.adapters.chat_adapter import ChatAdapter

        return ChatAdapter().parse(signature, completion)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_func_kwargs(raw: str) -> dict[str, str]:
    """Parse ``key='value', key2='value2'`` considering quoted strings."""
    kwargs: dict[str, str] = {}
    if not raw:
        return kwargs
    
    # Regex to match key='value' or key="value" pairs, handling escaped quotes
    # Matches: word characters as key, then =, then single or double quoted string
    pattern = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")")
    
    for match in pattern.finditer(raw):
        key = match.group(1)
        # value is group 2 (single quote) or group 3 (double quote)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        kwargs[key] = value
        
    return kwargs
