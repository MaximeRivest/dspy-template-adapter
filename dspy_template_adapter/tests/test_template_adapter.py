"""Tests for TemplateAdapter — no LM calls, only format() and parse() logic."""

from __future__ import annotations

import base64
import json
import pytest
import dspy

from dspy_template_adapter import TemplateAdapter

# Tiny valid PNG files (1x1 px) to avoid requiring Pillow in test env.
_RED_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
_BLUE_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sig(instructions: str = "", **fields):
    """Shorthand to create a dspy.Signature dynamically."""
    from dspy.signatures.signature import make_signature

    return make_signature(fields, instructions)


def _in(desc: str = "") -> tuple:
    return (str, dspy.InputField(desc=desc))


def _out(desc: str = "") -> tuple:
    return (str, dspy.OutputField(desc=desc))


# ---------------------------------------------------------------------------
# Signature fixtures
# ---------------------------------------------------------------------------


class Summarize(dspy.Signature):
    """You are a helpful assistant."""

    text: str = dspy.InputField()
    summary: str = dspy.OutputField()


class Triage(dspy.Signature):
    """Analyze a support ticket."""

    ticket_text: str = dspy.InputField()
    user_status: str = dspy.InputField()
    category: str = dspy.OutputField()
    priority: str = dspy.OutputField()


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction:
    def test_empty_messages_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            TemplateAdapter(messages=[])

    def test_basic_construction(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "Hello {text}"}],
            parse_mode="full_text",
        )
        assert len(adapter.message_templates) == 1
        assert adapter.parse_mode == "full_text"


# ===========================================================================
# format() — basic template rendering
# ===========================================================================


class TestFormatBasic:
    def test_simple_interpolation(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Summarize: {text}"},
            ],
        )
        msgs = adapter.format(Summarize, demos=[], inputs={"text": "hello world"})
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are concise."}
        assert msgs[1] == {"role": "user", "content": "Summarize: hello world"}

    def test_instruction_slot(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Task: {instruction}"},
                {"role": "user", "content": "{text}"},
            ],
        )
        msgs = adapter.format(Summarize, demos=[], inputs={"text": "hi"})
        assert "You are a helpful assistant." in msgs[0]["content"]

    def test_literal_braces_preserved(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "user", "content": 'Output JSON: {{"key": "value"}} for {text}'},
            ],
        )
        msgs = adapter.format(Summarize, demos=[], inputs={"text": "test"})
        assert '{"key": "value"}' in msgs[0]["content"]
        assert "test" in msgs[0]["content"]

    def test_unknown_variables_pass_through(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "{text} {unknown_var}"}],
        )
        msgs = adapter.format(Summarize, demos=[], inputs={"text": "hello"})
        assert msgs[0]["content"] == "hello {unknown_var}"

    def test_multi_input_fields(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "user", "content": "Status: {user_status}\nTicket: {ticket_text}"},
            ],
        )
        msgs = adapter.format(
            Triage,
            demos=[],
            inputs={"ticket_text": "My bill is wrong", "user_status": "VIP"},
        )
        assert "VIP" in msgs[0]["content"]
        assert "My bill is wrong" in msgs[0]["content"]


# ===========================================================================
# format() — template functions
# ===========================================================================


class TestTemplateFunctions:
    def test_inputs_default(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "{inputs()}"}],
        )
        msgs = adapter.format(Summarize, demos=[], inputs={"text": "hello"})
        assert "text: hello" in msgs[0]["content"]

    def test_inputs_yaml(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "{inputs(style='yaml')}"}],
        )
        msgs = adapter.format(Triage, demos=[], inputs={"ticket_text": "help", "user_status": "VIP"})
        content = msgs[0]["content"]
        assert "ticket_text: help" in content
        assert "user_status: VIP" in content

    def test_inputs_json(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "{inputs(style='json')}"}],
        )
        msgs = adapter.format(Summarize, demos=[], inputs={"text": "hi"})
        content = msgs[0]["content"]
        assert '"text"' in content

    def test_outputs_default(self):
        adapter = TemplateAdapter(
            messages=[{"role": "system", "content": "Produce:\n{outputs()}"}],
        )
        msgs = adapter.format(Triage, demos=[], inputs={"ticket_text": "", "user_status": ""})
        content = msgs[0]["content"]
        assert "category" in content
        assert "priority" in content

    def test_outputs_schema(self):
        adapter = TemplateAdapter(
            messages=[{"role": "system", "content": "{outputs(style='schema')}"}],
        )
        msgs = adapter.format(Summarize, demos=[], inputs={"text": ""})
        content = msgs[0]["content"]
        parsed = json.loads(content)
        assert "summary" in parsed

    def test_demos_inline_json(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Examples:\n{demos(style='json')}"},
                {"role": "user", "content": "{text}"},
            ],
        )
        demos = [{"text": "foo", "summary": "bar"}]
        msgs = adapter.format(Summarize, demos=demos, inputs={"text": "baz"})
        # demos should be inline in system message, NOT auto-injected as separate messages
        assert len(msgs) == 2
        assert "foo" in msgs[0]["content"]
        assert "bar" in msgs[0]["content"]

    def test_custom_helper(self):
        def my_helper(ctx, signature, demos, **kwargs):
            return f"CUSTOM:{kwargs.get('arg', 'none')}"

        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "{my_helper(arg='hello')}"}],
        )
        adapter.register_helper("my_helper", my_helper)
        msgs = adapter.format(Summarize, demos=[], inputs={"text": ""})
        assert msgs[0]["content"] == "CUSTOM:hello"


# ===========================================================================
# format() — demo directives and auto-injection
# ===========================================================================


class TestDemos:
    def test_role_demos_directive_default(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "demos"},
                {"role": "user", "content": "{text}"},
            ],
        )
        demos = [{"text": "example input", "summary": "example output"}]
        msgs = adapter.format(Summarize, demos=demos, inputs={"text": "real input"})
        # system + user demo + assistant demo + final user
        assert len(msgs) == 4
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["content"] == "real input"

    def test_role_demos_custom_template(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Be helpful."},
                {"role": "demos", "user": "Input: {text}", "assistant": "{outputs_json}"},
                {"role": "user", "content": "{text}"},
            ],
        )
        demos = [{"text": "in1", "summary": "out1"}]
        msgs = adapter.format(Summarize, demos=demos, inputs={"text": "real"})
        assert msgs[1]["content"] == "Input: in1"
        parsed_assistant = json.loads(msgs[2]["content"])
        assert parsed_assistant["summary"] == "out1"

    def test_auto_inject_demos_when_no_directive(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "{text}"},
            ],
        )
        demos = [{"text": "demo_in", "summary": "demo_out"}]
        msgs = adapter.format(Summarize, demos=demos, inputs={"text": "real"})
        # system + demo_user + demo_assistant + final_user
        assert len(msgs) == 4
        assert msgs[-1]["content"] == "real"

    def test_no_demos_no_extra_messages(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "System."},
                {"role": "user", "content": "{text}"},
            ],
        )
        msgs = adapter.format(Summarize, demos=[], inputs={"text": "hi"})
        assert len(msgs) == 2


# ===========================================================================
# format() — history
# ===========================================================================


class TestHistory:
    def test_history_directive(self):
        class ChatSig(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "You are a chatbot."},
                {"role": "history"},
                {"role": "user", "content": "{question}"},
            ],
        )
        history = dspy.History(messages=[
            {"question": "What is 1+1?", "answer": "2"},
        ])
        msgs = adapter.format(
            ChatSig,
            demos=[],
            inputs={"question": "What is 2+2?", "history": history},
        )
        # system + history_user + history_assistant + final_user
        assert len(msgs) == 4
        assert "1+1" in msgs[1]["content"]
        assert "2" in msgs[2]["content"]
        assert "2+2" in msgs[3]["content"]

    def test_history_function_inline_prevents_auto_injection(self):
        class ChatSig(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "History:\n{history(style='yaml')}"},
                {"role": "user", "content": "{question}"},
            ],
        )
        history = dspy.History(messages=[
            {"question": "What is 1+1?", "answer": "2"},
        ])
        msgs = adapter.format(
            ChatSig,
            demos=[],
            inputs={"question": "What is 2+2?", "history": history},
        )

        # history() consumed history inline, so there should be only 2 messages
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "What is 1+1?" in msgs[0]["content"]
        assert "2" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert "What is 2+2?" in msgs[1]["content"]

    def test_history_directive_respects_full_text_mode(self):
        class ChatSig(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "You are a chatbot."},
                {"role": "history"},
                {"role": "user", "content": "{question}"},
            ],
            parse_mode="full_text",
        )
        history = dspy.History(messages=[
            {"question": "What is 1+1?", "answer": "2"},
        ])
        msgs = adapter.format(
            ChatSig,
            demos=[],
            inputs={"question": "What is 2+2?", "history": history},
        )

        # assistant history message should be raw text in full_text mode
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "2"

    def test_history_directive_respects_xml_mode(self):
        class ChatSig(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "You are a chatbot."},
                {"role": "history"},
                {"role": "user", "content": "{question}"},
            ],
            parse_mode="xml",
        )
        history = dspy.History(messages=[
            {"question": "What is 1+1?", "answer": "2"},
        ])
        msgs = adapter.format(
            ChatSig,
            demos=[],
            inputs={"question": "What is 2+2?", "history": history},
        )

        # assistant history message should be xml in xml mode
        assert msgs[2]["role"] == "assistant"
        assert "<answer>2</answer>" in msgs[2]["content"]

    def test_inputs_helper_excludes_history_field(self):
        class ChatSig(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = TemplateAdapter(
            messages=[
                {"role": "user", "content": "{inputs(style='yaml')}"},
            ],
        )

        history = dspy.History(messages=[{"question": "Q1", "answer": "A1"}])
        msgs = adapter.format(
            ChatSig,
            demos=[],
            inputs={"question": "Q2", "history": history},
        )

        content = msgs[0]["content"]
        assert "question:" in content
        assert "history:" not in content


# ===========================================================================
# parse() — JSON mode
# ===========================================================================


class TestParseJson:
    def test_basic_json(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="json")
        result = adapter.parse(Summarize, '{"summary": "hello world"}')
        assert result == {"summary": "hello world"}

    def test_json_with_surrounding_text(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="json")
        result = adapter.parse(Summarize, 'Here is my answer: {"summary": "test"} done.')
        assert result == {"summary": "test"}

    def test_json_inside_markdown_fence(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="json")
        completion = "```json\n{\"summary\": \"hello\"}\n```"
        result = adapter.parse(Summarize, completion)
        assert result == {"summary": "hello"}

    def test_json_multi_field(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="json")
        completion = '{"category": "Billing", "priority": "HIGH"}'
        result = adapter.parse(Triage, completion)
        assert result["category"] == "Billing"
        assert result["priority"] == "HIGH"

    def test_json_missing_field_raises(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="json")
        with pytest.raises(Exception):
            adapter.parse(Triage, '{"category": "Billing"}')

    def test_no_json_raises(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="json")
        with pytest.raises(Exception):
            adapter.parse(Summarize, "no json here at all")


# ===========================================================================
# parse() — full_text mode
# ===========================================================================


class TestParseFullText:
    def test_single_output(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="full_text")
        result = adapter.parse(Summarize, "This is the summary.")
        assert result == {"summary": "This is the summary."}

    def test_multi_output_raises(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="full_text")
        with pytest.raises(Exception, match="exactly 1 output field"):
            adapter.parse(Triage, "some text")


# ===========================================================================
# parse() — XML mode
# ===========================================================================


class TestParseXml:
    def test_basic_xml(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="xml")
        result = adapter.parse(Summarize, "<summary>hello world</summary>")
        assert result == {"summary": "hello world"}

    def test_xml_with_surrounding_text(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="xml")
        result = adapter.parse(Summarize, "Sure, here you go:\n<summary>the answer</summary>\nDone!")
        assert result == {"summary": "the answer"}

    def test_xml_inside_markdown_fence(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="xml")
        completion = "```xml\n<summary>wrapped</summary>\n```"
        result = adapter.parse(Summarize, completion)
        assert result == {"summary": "wrapped"}

    def test_xml_multi_field(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="xml")
        completion = "<category>Billing</category>\n<priority>HIGH</priority>"
        result = adapter.parse(Triage, completion)
        assert result["category"] == "Billing"
        assert result["priority"] == "HIGH"

    def test_xml_multiline_value(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="xml")
        completion = "<summary>\nLine one.\nLine two.\n</summary>"
        result = adapter.parse(Summarize, completion)
        assert result["summary"] == "Line one.\nLine two."

    def test_xml_missing_field_raises(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="xml")
        with pytest.raises(Exception):
            adapter.parse(Triage, "<category>Billing</category>")

    def test_xml_no_tags_raises(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="xml")
        with pytest.raises(Exception):
            adapter.parse(Summarize, "no xml here at all")


# ===========================================================================
# parse() — chat mode
# ===========================================================================


class TestParseChat:
    def test_chat_mode_delegates(self):
        adapter = TemplateAdapter(messages=[{"role": "user", "content": ""}], parse_mode="chat")
        completion = "[[ ## summary ## ]]\nHello world\n\n[[ ## completed ## ]]"
        result = adapter.parse(Summarize, completion)
        assert result == {"summary": "Hello world"}


# ===========================================================================
# parse() — callable mode
# ===========================================================================


class TestParseCallable:
    def test_custom_parser(self):
        def my_parser(sig, text):
            return {"summary": text.upper()}

        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": ""}],
            parse_mode=my_parser,
        )
        result = adapter.parse(Summarize, "hello")
        assert result == {"summary": "HELLO"}


# ===========================================================================
# format_finetune_data()
# ===========================================================================


class TestFinetune:
    def test_finetune_json(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Summarize."},
                {"role": "user", "content": "{text}"},
            ],
            parse_mode="json",
        )
        data = adapter.format_finetune_data(
            Summarize,
            demos=[],
            inputs={"text": "hello"},
            outputs={"summary": "world"},
        )
        assert "messages" in data
        msgs = data["messages"]
        assert msgs[-1]["role"] == "assistant"
        parsed = json.loads(msgs[-1]["content"])
        assert parsed["summary"] == "world"

    def test_finetune_full_text(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "{text}"}],
            parse_mode="full_text",
        )
        data = adapter.format_finetune_data(
            Summarize,
            demos=[],
            inputs={"text": "hello"},
            outputs={"summary": "world"},
        )
        msgs = data["messages"]
        assert msgs[-1]["content"] == "world"

    def test_finetune_xml(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "{text}"}],
            parse_mode="xml",
        )
        data = adapter.format_finetune_data(
            Summarize,
            demos=[],
            inputs={"text": "hello"},
            outputs={"summary": "world"},
        )
        msgs = data["messages"]
        assert msgs[-1]["content"] == "<summary>world</summary>"


# ===========================================================================
# preview()
# ===========================================================================


class TestPreview:
    def test_preview_returns_messages(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "Hello {text}"}],
        )
        msgs = adapter.preview(Summarize, inputs={"text": "world"})
        assert msgs[0]["content"] == "Hello world"

    def test_preview_no_inputs(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "Static prompt"}],
        )
        msgs = adapter.preview(Summarize)
        assert msgs[0]["content"] == "Static prompt"


# ===========================================================================
# format() — XML-style template functions
# ===========================================================================


class TranslateToFrench(dspy.Signature):
    """You are an English-to-French translator."""

    user_english_text: str = dspy.InputField(desc="The English text to translate")
    user_french_target: str = dspy.InputField(desc="The target French variant")
    is_translation_task: str = dspy.OutputField(desc="yes or no")
    corrected_english: str = dspy.OutputField(desc="the English input with fixes")
    detected_tone: str = dspy.OutputField(desc="the tone of the English text")
    french_variant: str = dspy.OutputField(desc="the French variant used")
    translation: str = dspy.OutputField(desc="the final French translation")


class TestXmlStyleTemplateFunctions:
    def test_inputs_xml(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "{inputs(style='xml')}"}],
        )
        msgs = adapter.format(
            TranslateToFrench,
            demos=[],
            inputs={"user_english_text": "Hello", "user_french_target": "Québécois"},
        )
        content = msgs[0]["content"]
        assert "<user_english_text>Hello</user_english_text>" in content
        assert "<user_french_target>Québécois</user_french_target>" in content

    def test_inputs_xml_multi_field(self):
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "{inputs(style='xml')}"}],
        )
        msgs = adapter.format(
            Triage,
            demos=[],
            inputs={"ticket_text": "My bill is wrong", "user_status": "VIP"},
        )
        content = msgs[0]["content"]
        assert "<ticket_text>My bill is wrong</ticket_text>" in content
        assert "<user_status>VIP</user_status>" in content

    def test_outputs_xml_no_wrap(self):
        adapter = TemplateAdapter(
            messages=[{"role": "system", "content": "Format:\n{outputs(style='xml')}"}],
        )
        msgs = adapter.format(Triage, demos=[], inputs={"ticket_text": "", "user_status": ""})
        content = msgs[0]["content"]
        assert "<category>" in content
        assert "</category>" in content
        assert "<priority>" in content
        assert "</priority>" in content
        # No wrapper tag
        assert "<response>" not in content

    def test_outputs_xml_with_wrap(self):
        adapter = TemplateAdapter(
            messages=[{"role": "system", "content": "{outputs(style='xml', wrap='response')}"}],
        )
        msgs = adapter.format(Triage, demos=[], inputs={"ticket_text": "", "user_status": ""})
        content = msgs[0]["content"]
        assert content.startswith("<response>")
        assert content.endswith("</response>")
        assert "  <category>" in content
        assert "  <priority>" in content

    def test_outputs_xml_uses_field_descriptions(self):
        adapter = TemplateAdapter(
            messages=[{"role": "system", "content": "{outputs(style='xml')}"}],
        )
        msgs = adapter.format(
            TranslateToFrench,
            demos=[],
            inputs={"user_english_text": "", "user_french_target": ""},
        )
        content = msgs[0]["content"]
        # Descriptions from dspy.OutputField(desc=...) should appear as placeholder values
        assert "<is_translation_task>yes or no</is_translation_task>" in content
        assert "<translation>the final French translation</translation>" in content

    def test_outputs_xml_wrapped_uses_field_descriptions(self):
        adapter = TemplateAdapter(
            messages=[{"role": "system", "content": "{outputs(style='xml', wrap='response')}"}],
        )
        msgs = adapter.format(
            TranslateToFrench,
            demos=[],
            inputs={"user_english_text": "", "user_french_target": ""},
        )
        content = msgs[0]["content"]
        assert "<response>" in content
        assert "</response>" in content
        assert "  <is_translation_task>yes or no</is_translation_task>" in content
        assert "  <corrected_english>the English input with fixes</corrected_english>" in content

    def test_outputs_xml_fallback_to_field_name_when_no_desc(self):
        """When a field has no desc, the field name itself is used as placeholder."""

        class NakedSig(dspy.Signature):
            """Test."""
            text: str = dspy.InputField()
            result: str = dspy.OutputField()

        adapter = TemplateAdapter(
            messages=[{"role": "system", "content": "{outputs(style='xml')}"}],
        )
        msgs = adapter.format(NakedSig, demos=[], inputs={"text": ""})
        content = msgs[0]["content"]
        # Field name as fallback placeholder
        assert "<result>" in content
        assert "</result>" in content

    def test_demos_inline_xml(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Examples:\n{demos(style='xml')}"},
                {"role": "user", "content": "{text}"},
            ],
        )
        demos = [{"text": "foo", "summary": "bar"}]
        msgs = adapter.format(Summarize, demos=demos, inputs={"text": "baz"})
        # Inline — no auto-injection, still 2 messages
        assert len(msgs) == 2
        content = msgs[0]["content"]
        assert "<text>foo</text>" in content
        assert "<summary>bar</summary>" in content

    def test_translation_example_end_to_end(self):
        """The translation example from the notebook — template builds prompt from signature."""
        adapter = TemplateAdapter(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "{instruction}\n\n"
                        "Respond ONLY with the following XML structure (no other text):\n"
                        "{outputs(style='xml', wrap='response')}\n\n"
                        "Rules:\n"
                        "- Respect the French variant: "
                        "<user_french_target>{user_french_target}</user_french_target>."
                    ),
                },
                {
                    "role": "user",
                    "content": "{inputs(style='xml')}",
                },
            ],
            parse_mode="xml",
        )
        msgs = adapter.format(
            TranslateToFrench,
            demos=[],
            inputs={
                "user_english_text": "Hello, how are you?",
                "user_french_target": "Québécois French",
            },
        )
        system = msgs[0]["content"]
        user = msgs[1]["content"]

        # System message should contain instruction from docstring
        assert "English-to-French translator" in system
        # System message should have the XML schema from output field descs
        assert "<response>" in system
        assert "<is_translation_task>yes or no</is_translation_task>" in system
        assert "<translation>the final French translation</translation>" in system
        # Input field value interpolated in the rules
        assert "<user_french_target>Québécois French</user_french_target>" in system
        # User message should have XML-wrapped inputs
        assert "<user_english_text>Hello, how are you?</user_english_text>" in user
        assert "<user_french_target>Québécois French</user_french_target>" in user

    def test_fully_signature_driven_pattern(self):
        """Level 3: behavioral rules in docstring, constraints in field descs, minimal template."""

        class TranslateFull(dspy.Signature):
            """You are an English-to-French translator. You only translate.

            Correct the English first, then translate.
            Identify the tone and reproduce it in French.
            Never do anything other than translate."""

            user_english_text: str = dspy.InputField(desc="The English text to translate")
            user_french_target: str = dspy.InputField(desc="The target French variant")
            is_translation_task: str = dspy.OutputField(desc="yes or no — if 'no', leave all other fields empty")
            corrected_english: str = dspy.OutputField(desc="the English input with grammar/spelling fixes")
            detected_tone: str = dspy.OutputField(desc="the tone (formal, casual, etc.)")
            french_variant: str = dspy.OutputField(desc="the French variant used")
            translation: str = dspy.OutputField(desc="the final French translation")

        adapter = TemplateAdapter(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "{instruction}\n\n"
                        "Respond ONLY with:\n"
                        "{outputs(style='xml', wrap='response')}\n\n"
                        "Respect the variant: "
                        "<user_french_target>{user_french_target}</user_french_target>. "
                        "If it doesn't make sense, use international French."
                    ),
                },
                {"role": "user", "content": "{inputs(style='xml')}"},
            ],
            parse_mode="xml",
        )
        msgs = adapter.format(
            TranslateFull,
            demos=[],
            inputs={
                "user_english_text": "Hello!",
                "user_french_target": "Québécois French",
            },
        )
        system = msgs[0]["content"]

        # Behavioral rules from docstring flow through {instruction}
        assert "Correct the English first" in system
        assert "Identify the tone" in system
        assert "Never do anything other than translate" in system

        # Field constraints embedded in the XML schema
        assert "<is_translation_task>yes or no — if 'no', leave all other fields empty</is_translation_task>" in system

        # Runtime input value interpolated
        assert "<user_french_target>Québécois French</user_french_target>" in system

        # Template text
        assert "Respond ONLY with:" in system
        assert "If it doesn't make sense, use international French." in system

        # User message has XML-wrapped inputs
        user = msgs[1]["content"]
        assert "<user_english_text>Hello!</user_english_text>" in user


# ===========================================================================
# format() — parse-mode-aware demo messages
# ===========================================================================


class TestDemoFormatMatchesParseMode:
    def test_auto_inject_demos_xml_mode(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Translate."},
                {"role": "user", "content": "{text}"},
            ],
            parse_mode="xml",
        )
        demos = [{"text": "Hello", "summary": "Bonjour"}]
        msgs = adapter.format(Summarize, demos=demos, inputs={"text": "Goodbye"})
        # system + demo_user + demo_assistant + final_user
        assert len(msgs) == 4
        assistant = msgs[2]["content"]
        assert "<summary>Bonjour</summary>" in assistant

    def test_auto_inject_demos_json_mode(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Classify."},
                {"role": "user", "content": "{text}"},
            ],
            parse_mode="json",
        )
        demos = [{"text": "Hello", "summary": "greeting"}]
        msgs = adapter.format(Summarize, demos=demos, inputs={"text": "Goodbye"})
        assistant = msgs[2]["content"]
        parsed = json.loads(assistant)
        assert parsed["summary"] == "greeting"

    def test_auto_inject_demos_full_text_mode(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Summarize."},
                {"role": "user", "content": "{text}"},
            ],
            parse_mode="full_text",
        )
        demos = [{"text": "Hello world", "summary": "A greeting"}]
        msgs = adapter.format(Summarize, demos=demos, inputs={"text": "real"})
        assistant = msgs[2]["content"]
        assert assistant == "A greeting"

    def test_demos_directive_xml_mode(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Translate."},
                {"role": "demos"},
                {"role": "user", "content": "{text}"},
            ],
            parse_mode="xml",
        )
        demos = [{"text": "Hello", "summary": "Bonjour"}]
        msgs = adapter.format(Summarize, demos=demos, inputs={"text": "real"})
        assistant = msgs[2]["content"]
        assert "<summary>Bonjour</summary>" in assistant

    def test_xml_demo_multi_output_fields(self):
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Triage."},
                {"role": "user", "content": "{inputs()}"},
            ],
            parse_mode="xml",
        )
        demos = [{"ticket_text": "Bill wrong", "user_status": "VIP", "category": "Billing", "priority": "HIGH"}]
        msgs = adapter.format(
            Triage,
            demos=demos,
            inputs={"ticket_text": "Real ticket", "user_status": "Normal"},
        )
        assistant = msgs[2]["content"]
        assert "<category>Billing</category>" in assistant
        assert "<priority>HIGH</priority>" in assistant


# ===========================================================================
# Image support
# ===========================================================================


class TestImages:
    """Verify that dspy.Image inputs are correctly split into multimodal
    content blocks by the adapter's format() pipeline."""

    @pytest.fixture(autouse=True)
    def _create_test_image(self, tmp_path):
        """Create a tiny PNG for each test (no Pillow dependency)."""
        self.img_path = str(tmp_path / "red.png")
        with open(self.img_path, "wb") as f:
            f.write(base64.b64decode(_RED_PNG_B64))

    def _image_sig(self):
        from dspy.adapters.types import Image

        class Describe(dspy.Signature):
            """Describe the image."""

            image: Image = dspy.InputField()
            description: str = dspy.OutputField()

        return Describe

    def _multi_image_sig(self):
        from dspy.adapters.types import Image

        class Compare(dspy.Signature):
            """Compare two images."""

            image_a: Image = dspy.InputField()
            image_b: Image = dspy.InputField()
            comparison: str = dspy.OutputField()

        return Compare

    def test_single_image_format(self):
        from dspy.adapters.types import Image

        Describe = self._image_sig()
        adapter = TemplateAdapter(
            messages=[
                {"role": "system", "content": "Describe the image."},
                {"role": "user", "content": "What is this? {image}"},
            ],
            parse_mode="full_text",
        )
        img = Image(self.img_path)
        msgs = adapter.format(Describe, demos=[], inputs={"image": img})

        assert len(msgs) == 2
        # System message stays a plain string
        assert isinstance(msgs[0]["content"], str)
        # User message is split into content blocks
        assert isinstance(msgs[1]["content"], list)
        types = [block["type"] for block in msgs[1]["content"]]
        assert "text" in types
        assert "image_url" in types

    def test_image_url_contains_base64(self):
        from dspy.adapters.types import Image

        Describe = self._image_sig()
        adapter = TemplateAdapter(
            messages=[{"role": "user", "content": "{image}"}],
            parse_mode="full_text",
        )
        img = Image(self.img_path)
        msgs = adapter.format(Describe, demos=[], inputs={"image": img})
        img_blocks = [b for b in msgs[0]["content"] if b.get("type") == "image_url"]
        assert len(img_blocks) == 1
        assert img_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_multi_image_format(self):
        from dspy.adapters.types import Image

        Compare = self._multi_image_sig()
        # Create a second image
        import os

        blue_path = os.path.join(os.path.dirname(self.img_path), "blue.png")
        with open(blue_path, "wb") as f:
            f.write(base64.b64decode(_BLUE_PNG_B64))

        adapter = TemplateAdapter(
            messages=[
                {"role": "user", "content": "A: {image_a}\nB: {image_b}\nCompare."},
            ],
            parse_mode="full_text",
        )
        msgs = adapter.format(
            Compare,
            demos=[],
            inputs={
                "image_a": Image(self.img_path),
                "image_b": Image(blue_path),
            },
        )

        assert len(msgs) == 1
        blocks = msgs[0]["content"]
        assert isinstance(blocks, list)
        img_blocks = [b for b in blocks if b.get("type") == "image_url"]
        text_blocks = [b for b in blocks if b.get("type") == "text"]
        assert len(img_blocks) == 2
        assert len(text_blocks) >= 1  # at least text around/between images

    def test_image_with_text_field(self):
        """Image + regular text field in the same user message."""
        from dspy.adapters.types import Image

        class Analyze(dspy.Signature):
            """Analyze an image."""

            image: Image = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = TemplateAdapter(
            messages=[
                {"role": "user", "content": "{question}\n{image}"},
            ],
            parse_mode="full_text",
        )
        img = Image(self.img_path)
        msgs = adapter.format(
            Analyze,
            demos=[],
            inputs={"image": img, "question": "What color?"},
        )

        blocks = msgs[0]["content"]
        assert isinstance(blocks, list)
        # Should have text block with the question, then image block
        text_blocks = [b for b in blocks if b.get("type") == "text"]
        img_blocks = [b for b in blocks if b.get("type") == "image_url"]
        assert any("What color?" in b["text"] for b in text_blocks)
        assert len(img_blocks) == 1
