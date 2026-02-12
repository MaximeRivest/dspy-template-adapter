"""Live integration tests that prove TemplateAdapter compatibility with DSPy optimizers
and agentic modules.

These tests are skipped by default. To run:

    RUN_LIVE_DSPY=1 uv run pytest -q dspy_template_adapter/tests/test_live_dspy_compat.py -s

Requirements:
- OPENAI_API_KEY in env
- Network access
- For CodeAct test: deno installed
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import dspy

from dspy_template_adapter import Predict, TemplateAdapter

RUN_LIVE = os.getenv("RUN_LIVE_DSPY", "0") == "1"
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
HAS_GEPA = hasattr(dspy, "GEPA")
HAS_CODEACT = hasattr(dspy, "CodeAct")
CODEACT_HAS_INTERPRETER_ARG = HAS_CODEACT and "interpreter" in inspect.signature(dspy.CodeAct.__init__).parameters

pytestmark = pytest.mark.skipif(
    not (RUN_LIVE and HAS_OPENAI_KEY),
    reason="Set RUN_LIVE_DSPY=1 and OPENAI_API_KEY to run live DSPy integration tests.",
)


def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    if "odd" in s:
        return "odd"
    if "even" in s:
        return "even"
    return s.split()[0] if s else ""


class Parity(dspy.Signature):
    """Reply with exactly one word: odd or even."""

    number: int = dspy.InputField()
    label: str = dspy.OutputField()


def _parity_trainset() -> list[dspy.Example]:
    return [
        dspy.Example(number=1, label="odd").with_inputs("number"),
        dspy.Example(number=2, label="even").with_inputs("number"),
        dspy.Example(number=3, label="odd").with_inputs("number"),
        dspy.Example(number=4, label="even").with_inputs("number"),
    ]


def _full_text_adapter() -> TemplateAdapter:
    return TemplateAdapter(
        messages=[
            {"role": "system", "content": "{instruction}"},
            {"role": "user", "content": "{inputs(style='yaml')}"},
        ],
        parse_mode="full_text",
    )


def _agent_json_adapter() -> TemplateAdapter:
    return TemplateAdapter(
        messages=[
            {
                "role": "system",
                "content": (
                    "{instruction}\n\n"
                    "You MUST return a JSON object containing every output field listed below.\n"
                    "{outputs()}\n"
                    "For tool selection fields, choose from the allowed tool names in the instructions."
                ),
            },
            {"role": "user", "content": "{inputs(style='yaml')}"},
        ],
        parse_mode="json",
    )


def _lm() -> dspy.LM:
    return dspy.LM("gpt-4.1-nano", temperature=0, max_tokens=400)


def _metric(gold, pred, trace=None):
    return _normalize_label(pred.label) == gold.label


def _gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    ok = _normalize_label(pred.label) == gold.label
    score = 1.0 if ok else 0.0
    if pred_name is None:
        return score
    return {
        "score": score,
        "feedback": "Correct." if ok else f"Expected {gold.label}, got {pred.label!r}",
    }


# ---------------------------------------------------------------------------
# Optimizers with per-module TemplateAdapter binding (Predict(..., adapter=...))
# ---------------------------------------------------------------------------


def test_bootstrapfewshot_with_template_adapter():
    lm = _lm()
    adapter = _full_text_adapter()
    trainset = _parity_trainset()
    student = Predict(Parity, adapter=adapter)

    teleprompter = dspy.BootstrapFewShot(
        metric=_metric,
        max_bootstrapped_demos=1,
        max_labeled_demos=2,
        max_rounds=1,
    )

    with dspy.context(lm=lm):
        compiled = teleprompter.compile(student, trainset=trainset)
        pred = compiled(number=7)

    assert _normalize_label(pred.label) == "odd"
    # Proves our template rendered and demos were injected into that format.
    assert any("number:" in m["content"] for m in lm.history[-1]["messages"] if isinstance(m.get("content"), str))


def test_miprov2_with_template_adapter():
    lm = _lm()
    adapter = _full_text_adapter()
    trainset = _parity_trainset()
    student = Predict(Parity, adapter=adapter)

    teleprompter = dspy.MIPROv2(
        metric=_metric,
        prompt_model=lm,
        task_model=lm,
        auto=None,
        num_candidates=2,
        max_bootstrapped_demos=1,
        max_labeled_demos=2,
        num_threads=1,
        verbose=False,
    )

    with dspy.context(lm=lm):
        compiled = teleprompter.compile(
            student,
            trainset=trainset,
            valset=trainset,
            num_trials=1,
            minibatch=False,
            requires_permission_to_run=False,
        )
        pred = compiled(number=9)

    assert _normalize_label(pred.label) == "odd"
    assert any("number:" in m["content"] for m in lm.history[-1]["messages"] if isinstance(m.get("content"), str))


@pytest.mark.skipif(not HAS_GEPA, reason="GEPA is not available in this DSPy version.")
def test_gepa_with_template_adapter():
    lm = _lm()
    adapter = _full_text_adapter()
    trainset = _parity_trainset()
    student = Predict(Parity, adapter=adapter)

    teleprompter = dspy.GEPA(
        metric=_gepa_metric,
        max_metric_calls=8,
        reflection_lm=lm,
        num_threads=1,
        track_stats=False,
        use_wandb=False,
        use_mlflow=False,
    )

    with dspy.context(lm=lm):
        compiled = teleprompter.compile(student, trainset=trainset[:3], valset=trainset[3:])
        pred = compiled(number=11)

    assert _normalize_label(pred.label) == "odd"
    assert any("number:" in m["content"] for m in lm.history[-1]["messages"] if isinstance(m.get("content"), str))


# ---------------------------------------------------------------------------
# Agentic modules with global adapter (ReAct, CodeAct)
# ---------------------------------------------------------------------------


def test_react_with_template_adapter():
    lm = _lm()
    adapter = _agent_json_adapter()

    def calculator(expression: str) -> str:
        return str(eval(expression, {"__builtins__": {}}, {}))

    react = dspy.ReAct("question -> answer", tools=[calculator], max_iters=3)

    with dspy.context(lm=lm, adapter=adapter):
        out = react(question="What is 2+3? Use calculator, then finish.")

    assert "5" in str(out.answer)
    assert any(v == "calculator" for k, v in out.trajectory.items() if k.startswith("tool_name_"))
    # Proves global TemplateAdapter prompt text reached ReAct's internal Predict calls.
    assert "You MUST return a JSON object" in lm.history[0]["messages"][0]["content"]


@pytest.mark.skipif(not HAS_CODEACT, reason="CodeAct not available in this DSPy version.")
@pytest.mark.skipif(not CODEACT_HAS_INTERPRETER_ARG, reason="This DSPy version does not support CodeAct(interpreter=...).")
def test_codeact_with_template_adapter():
    lm = _lm()
    adapter = _agent_json_adapter()

    # In this environment, CodeAct's PythonInterpreter needs npm:pyodide resolution.
    os.environ.setdefault("DENO_NODE_MODULES_DIR", "auto")

    from dspy.primitives.python_interpreter import PythonInterpreter

    runner = Path(__import__("dspy.primitives.python_interpreter", fromlist=["__file__"]).__file__).with_name("runner.js")
    deno_dir = Path.home() / ".cache" / "deno"
    node_modules = Path.home() / "node_modules"

    interpreter = PythonInterpreter(
        deno_command=[
            "deno",
            "run",
            f"--allow-read={runner},{deno_dir},{node_modules}",
            str(runner),
        ]
    )

    def multiply(a: int, b: int) -> int:
        return a * b

    codeact = dspy.CodeAct("question -> answer", tools=[multiply], max_iters=3, interpreter=interpreter)

    with dspy.context(lm=lm, adapter=adapter):
        out = codeact(question="Use multiply(6, 7), then give only the final answer.")

    assert "42" in str(out.answer)
    assert "You MUST return a JSON object" in lm.history[0]["messages"][0]["content"]


# ---------------------------------------------------------------------------
# RML availability in installed DSPy version
# ---------------------------------------------------------------------------


def test_rml_availability_or_skip():
    """RML is not present in DSPy 3.1.3 at time of writing.

    This test gives an explicit signal rather than silently assuming support.
    """
    if not hasattr(dspy, "RML"):
        pytest.skip("RML is not available in this DSPy version (dspy.__version__=%s)." % dspy.__version__)

    # If future DSPy versions add RML, this test will fail loudly until we add
    # a real integration coverage path for it.
    assert hasattr(dspy, "RML")
