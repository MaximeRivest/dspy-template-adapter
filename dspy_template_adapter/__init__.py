"""
dspy-template-adapter — A DSPy Adapter for exact-fidelity prompt templates.

Designed for contribution to dspy.adapters.template_adapter.

Usage:
    from dspy_template_adapter import TemplateAdapter

    adapter = TemplateAdapter(
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Summarize: {text}"},
        ],
        parse_mode="full_text",
    )

    class Summarize(dspy.Signature):
        text: str = dspy.InputField()
        summary: str = dspy.OutputField()

    dspy.configure(lm=dspy.LM("gpt-4.1-nano"), adapter=adapter)
    result = dspy.Predict(Summarize)(text="some text")
"""

from dspy_template_adapter.template_adapter import TemplateAdapter
from dspy_template_adapter.predict import Predict

__all__ = ["TemplateAdapter", "Predict"]
