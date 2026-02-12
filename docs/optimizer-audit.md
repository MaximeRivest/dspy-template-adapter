# Optimizer Audit (Reproducible, JSON-Backed)

This project includes a live audit script that proves `TemplateAdapter` works with DSPy optimizers and that prompt/messages actually change in compiled programs.

## What it validates

- `BootstrapFewShot` works with `Predict(..., adapter=TemplateAdapter(...))`
- `MIPROv2` works with the adapter and can change instructions
- `GEPA` works with the adapter and can evolve instructions (when available)
- LM request messages are captured before/after compilation and written to disk

## Script

- `scripts/live_optimizer_audit.py`

It runs a harder 4-class task with an intentionally flawed starting instruction, then compiles with each optimizer and saves:

- baseline vs compiled accuracy
- instruction text before/after
- demo count before/after
- probe call messages (`lm.history[-1]["messages"]`) before/after

## Requirements

- `OPENAI_API_KEY` set
- network access
- `uv` installed

## Run

```bash
uv sync
uv run python scripts/live_optimizer_audit.py
```

## Output location

Per run:

```text
artifacts/optimizer_audit/YYYYMMDD-HHMMSS/
  summary.json
  bootstrap_report.json
  miprov2_report.json
  gepa_report.json   # only when GEPA is available in your DSPy version
```

## What to inspect in JSON

In each report:

- `before.instruction` / `after.instruction`
- `instruction_changed`
- `baseline_accuracy` / `compiled_accuracy`
- `baseline_probe.messages` / `compiled_probe.messages`

The system message includes a marker:

- `TEMPLATE_ADAPTER_AUDIT_MARKER`

That marker confirms the messages came through your `TemplateAdapter` template.

## Example run (recorded)

From one recorded run:

- BootstrapFewShot: `0.40 -> 0.70`, instruction unchanged, demos `0 -> 8`
- MIPROv2: `0.40 -> 0.70`, instruction changed, demos `0 -> 8`
- GEPA: `0.40 -> 0.80`, instruction changed, demos `0 -> 0`

This demonstrates:

1. few-shot optimization path (demos injected), and
2. instruction optimization path (instruction rewritten), and
3. changed LM call payloads in captured message JSON.
