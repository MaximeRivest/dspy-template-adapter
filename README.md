# dspy-template-adapter

A DSPy Adapter that gives you **exact control over the messages sent to the LM** — no hidden prompt rewriting. Define your prompt as a list of messages (just like the OpenAI API), and the adapter handles variable interpolation, few-shot demos, conversation history, output parsing, and integration with DSPy's full optimization pipeline.

```bash
pip install dspy-template-adapter
```

## Why?

DSPy's built-in adapters (`ChatAdapter`, `JSONAdapter`) rewrite your prompt into DSPy's own scaffolding format (`[[ ## field ## ]]` markers, field descriptions, etc.). This is great for optimization, but it means you can't reproduce the exact API calls from a vanilla OpenAI/Anthropic prompt.

`TemplateAdapter` solves this: **your messages are the prompt**. The signature defines the I/O contract, the adapter renders your templates, and DSPy handles the rest (caching, tracing, retries, evaluation, optimization).

## Quickstart

```python
import dspy
from dspy_template_adapter import TemplateAdapter, Predict

# 1. Define a signature (the I/O contract)
class Summarize(dspy.Signature):
    """Summarize input text concisely."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

# 2. Define the prompt template
adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "You are a concise assistant. {instruction}"},
        {"role": "user", "content": "Summarize:\n\n{text}"},
    ],
    parse_mode="full_text",
)

# 3. Bind adapter to predictor
summarizer = Predict(Summarize, adapter=adapter)

# 4. Configure LM and call
dspy.configure(lm=dspy.LM("gpt-4.1-nano"))
out = summarizer(text="DSPy is a framework for programming language models.")
print(out.summary)
```

What happens under the hood:

1. `{instruction}` is replaced with the signature docstring (`"Summarize input text concisely."`)
2. `{text}` is replaced with the input value
3. The two rendered messages are sent to the LM — nothing else added
4. The full response is mapped to `summary` (because `parse_mode="full_text"`)

## Core Concepts

The adapter has two jobs:

- **`format()`** — Render your message templates + input values into the final `messages` list sent to the LM
- **`parse()`** — Extract output fields from the LM's raw completion string

### Template Syntax

Inside message `content` strings, you can use:

| Syntax | What it does |
|--------|-------------|
| `{field_name}` | Replaced with the value of that input field |
| `{instruction}` | Replaced with the signature's docstring |
| `{inputs()}` | Renders all input fields. Supports `style='yaml'`, `style='json'`, or default |
| `{outputs()}` | Renders output field descriptions. Supports `style='schema'` for JSON schema |
| `{demos()}` | Renders few-shot demos inline. Supports `style='json'`, `style='yaml'`, or default |
| `{my_helper()}` | Calls a custom function registered with `register_helper()` |
| `{{` / `}}` | Literal braces (escaped) |

Two special **directive roles** expand into multiple messages at render time:

| Directive | What it does |
|-----------|-------------|
| `{"role": "demos"}` | Expands into user/assistant message pairs for each demo |
| `{"role": "history"}` | Expands a `dspy.History` object into prior conversation turns |

### Parse Modes

| Value | When to use | Requirement |
|-------|-------------|-------------|
| `"full_text"` | Entire LM response is your output | Exactly **1** output field |
| `"json"` (default) | LM returns JSON with keys matching output fields | Any number of output fields |
| `"xml"` | LM returns `<field_name>value</field_name>` tags | Tags for every output field |
| `"chat"` | Delegates to DSPy's `ChatAdapter` parser | DSPy's `[[ ## field ## ]]` format |
| A callable | Custom extraction logic | `(signature, completion) -> dict` |

## Structured JSON Output

When your signature has multiple output fields:

```python
class Triage(dspy.Signature):
    ticket: str = dspy.InputField()
    category: str = dspy.OutputField()
    priority: str = dspy.OutputField()

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "Return JSON only with keys: category, priority."},
        {"role": "user", "content": "{inputs(style='yaml')}"},
    ],
    parse_mode="json",
)

triage = Predict(Triage, adapter=adapter)
out = triage(ticket="Production checkout failing for VIP users")
print(out.category, out.priority)
```

The JSON parser uses `json_repair` for robustness — it handles malformed JSON and can extract JSON objects embedded in surrounding text.

## XML Output

Some models (especially Claude) perform well with XML-tagged output:

```python
class Review(dspy.Signature):
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField()
    reasoning: str = dspy.OutputField()

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": (
            "Analyze the sentiment. Respond with XML tags:\n"
            "<reasoning>your reasoning</reasoning>\n"
            "<sentiment>positive, negative, or neutral</sentiment>"
        )},
        {"role": "user", "content": "{text}"},
    ],
    parse_mode="xml",
)

reviewer = Predict(Review, adapter=adapter)
out = reviewer(text="This product exceeded all my expectations!")
print(out.sentiment, out.reasoning)
```

XML parsing handles tags anywhere in the response, multiline values, and raises clear errors for missing fields. XML avoids the common problem of models escaping quotes inside JSON string values.

## Template Functions

### `{inputs()}` — render all input field values

```python
# Default (same as yaml): "field: value" lines
{"role": "user", "content": "{inputs()}"}

# JSON object
{"role": "user", "content": "{inputs(style='json')}"}
```

### `{outputs()}` — render output field descriptions

```python
# Numbered list of field names/types
{"role": "system", "content": "Produce:\n{outputs()}"}

# JSON schema
{"role": "system", "content": "Match this schema:\n{outputs(style='schema')}"}
```

### `{demos()}` — render few-shot examples inline

```python
# Demos as JSON objects inside a message
{"role": "system", "content": "Examples:\n{demos(style='json')}"}
```

Styles: `'json'`, `'yaml'`, or default (numbered text blocks).

## Few-Shot Demos: Three Strategies

### A) Inline — demos as text inside a message

```python
adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "Examples:\n{demos(style='json')}"},
        {"role": "user", "content": "{inputs(style='yaml')}"},
    ],
    parse_mode="json",
)
```

Result: 2 messages. Demos are text inside the system message.

### B) Directive — demos as separate conversation turns

```python
adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "Classify tickets."},
        {"role": "demos"},
        {"role": "user", "content": "{inputs(style='yaml')}"},
    ],
    parse_mode="json",
)
```

Each demo becomes a user + assistant message pair. You can customize the format:

```python
{"role": "demos", "user": "Ticket: {ticket}", "assistant": "{outputs_json}"}
```

### C) Auto-injection

If your template doesn't mention demos at all, they're automatically injected as user/assistant pairs before the final user message when an optimizer adds them. This ensures compatibility with DSPy's optimization pipeline (BootstrapFewShot, etc.) without template changes.

## The `{instruction}` Slot

```python
class Summarize(dspy.Signature):
    """Summarize input text concisely."""  # <-- this is {instruction}
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "You are helpful. {instruction}"},
        {"role": "user", "content": "{text}"},
    ],
    parse_mode="full_text",
)
```

DSPy optimizers like **MIPRO** and **COPRO** rewrite the signature's instruction string. Including `{instruction}` in your template lets them optimize your prompt. Omitting it makes your prompt fully static.

## Conversation History

For multi-turn chatbots, use `dspy.History` and the `{"role": "history"}` directive:

```python
class ChatSig(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "You are a helpful chatbot."},
        {"role": "history"},
        {"role": "user", "content": "{question}"},
    ],
    parse_mode="full_text",
)

chat = Predict(ChatSig, adapter=adapter)
history = dspy.History(messages=[
    {"question": "What is 1+1?", "answer": "2"},
])
resp = chat(question="What is 2+2?", history=history)
```

The directive expands each history entry into user/assistant message pairs. If omitted, history is auto-injected before the last user message.

## Image Support

The adapter works seamlessly with `dspy.Image` inputs. Images are automatically converted to multimodal content blocks (the `image_url` format the OpenAI API expects).

### Single Image

```python
from dspy.adapters.types import Image

class Describe(dspy.Signature):
    """Describe the image."""
    image: Image = dspy.InputField()
    description: str = dspy.OutputField()

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "You describe images in one sentence."},
        {"role": "user", "content": "What is in this image? {image}"},
    ],
    parse_mode="full_text",
)

dspy.configure(lm=dspy.LM("gpt-4.1-nano"))
describer = Predict(Describe, adapter=adapter)
img = Image.from_file("photo.png")
out = describer(image=img)
print(out.description)
```

The `{image}` placeholder is first rendered to DSPy's internal image marker, then `split_message_content_for_custom_types()` splits the user message into proper content blocks:

```python
# What the LM actually receives:
[
    {"role": "system", "content": "You describe images in one sentence."},
    {"role": "user", "content": [
        {"type": "text", "text": "What is in this image? "},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}
]
```

### Multiple Images

```python
from dspy.adapters.types import Image

class Compare(dspy.Signature):
    """Compare two images."""
    image_a: Image = dspy.InputField()
    image_b: Image = dspy.InputField()
    comparison: str = dspy.OutputField()

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "Compare images in one sentence."},
        {"role": "user", "content": "Image A: {image_a}\nImage B: {image_b}\nCompare them."},
    ],
    parse_mode="full_text",
)

comparer = Predict(Compare, adapter=adapter)
out = comparer(
    image_a=Image.from_file("red.png"),
    image_b=Image.from_file("blue.png"),
)
print(out.comparison)
```

Each `{image_*}` placeholder becomes its own `image_url` block, with text blocks for the surrounding content.

### Images with Other Parse Modes

Images work with any parse mode — `full_text`, `json`, `xml`, or custom:

```python
from dspy.adapters.types import Image

class Analyze(dspy.Signature):
    """Analyze an image."""
    image: Image = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    confidence: str = dspy.OutputField()

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "Answer the question about the image. Return JSON with keys: answer, confidence."},
        {"role": "user", "content": "{question}\n{image}"},
    ],
    parse_mode="json",
)

dspy.configure(lm=dspy.LM("gpt-4.1-nano"))
analyzer = Predict(Analyze, adapter=adapter)
out = analyzer(image=Image.from_file("photo.png"), question="What color is this?")
print(out.answer, out.confidence)
```

## Custom Template Helpers

Register custom template functions for complex rendering logic:

```python
def format_as_xml(ctx, signature, demos, **kwargs):
    tag = kwargs.get("tag", "input")
    parts = []
    for name in signature.input_fields:
        val = ctx.get(name, "")
        parts.append(f"<{tag}_{name}>{val}</{tag}_{name}>")
    return "\n".join(parts)

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "Process XML input."},
        {"role": "user", "content": "{format_as_xml(tag='field')}"},
    ],
    parse_mode="full_text",
)
adapter.register_helper("format_as_xml", format_as_xml)
```

Helper signature: `(ctx: dict, signature, demos: list, **kwargs) -> str`

## Per-Module Adapter Isolation

The `Predict` wrapper lets each predictor use a different adapter:

```python
formal = Predict(Summarize, adapter=TemplateAdapter(
    messages=[{"role": "system", "content": "Formal tone."}, {"role": "user", "content": "{text}"}],
    parse_mode="full_text",
))
casual = Predict(Summarize, adapter=TemplateAdapter(
    messages=[{"role": "system", "content": "Casual tone."}, {"role": "user", "content": "{text}"}],
    parse_mode="full_text",
))

# Each call uses its own system prompt
out_formal = formal(text="Quantum computing is...")
out_casual = casual(text="Quantum computing is...")
```

## Custom Parse Functions

Pass any callable as `parse_mode`:

```python
import re

def extract_rating(signature, completion):
    match = re.search(r"(\d+)/10", completion)
    return {"rating": match.group(1) if match else "0"}

adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "Rate quality 1-10. Format: N/10."},
        {"role": "user", "content": "{text}"},
    ],
    parse_mode=extract_rating,
)
```

## Debugging

### Preview without calling the LM

```python
msgs = adapter.preview(Summarize, inputs={"text": "hello"})
for msg in msgs:
    print(f"[{msg['role']}] {msg['content']}")
```

### Inspect what was actually sent

```python
last = dspy.settings.lm.history[-1]
for msg in last["messages"]:
    print(f"[{msg['role']}] {msg['content']}")
```

### Checklist

1. **Import error?** Ensure `dspy-template-adapter` is installed (`pip install dspy-template-adapter`)
2. **Template wrong?** Use `adapter.preview(...)` — no LM call needed
3. **Parse error on `full_text`?** Signature must have exactly 1 output field
4. **Parse error on `json`?** LM response must contain all output field keys
5. **Parse error on `xml`?** LM response must contain `<field>...</field>` for every output field
6. **Demos missing?** `{demos()}` inline and `{"role": "demos"}` directive are mutually exclusive with auto-injection — pick one
7. **Literal braces eaten?** Use `{{` and `}}`

## Finetuning Data Export

```python
ft = adapter.format_finetune_data(
    Summarize,
    demos=[],
    inputs={"text": "DSPy is a framework."},
    outputs={"summary": "A framework for LMs."},
)
# ft["messages"] is an OpenAI-compatible message list with assistant response appended
```

## API Reference

```
TemplateAdapter(
    messages: list[dict],         # Message templates (required)
    parse_mode: str | callable,   # "json" (default), "full_text", "xml", "chat", or callable
)

Methods:
    .format(signature, demos, inputs) -> list[dict]
        Render messages from templates.

    .parse(signature, completion) -> dict
        Extract output fields from LM response.

    .preview(signature, demos=None, inputs=None) -> list[dict]
        Render without calling LM. For debugging.

    .register_helper(name, fn)
        Register custom {name()} template function.
        fn: (ctx, signature, demos, **kwargs) -> str

    .format_finetune_data(signature, demos, inputs, outputs) -> dict
        Generate OpenAI-compatible finetuning entry.

Predict(signature, adapter=adapter)
    dspy.Predict subclass with per-module adapter binding.
```

## License

MIT
