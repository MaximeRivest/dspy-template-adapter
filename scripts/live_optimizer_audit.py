from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy

from dspy_template_adapter import Predict, TemplateAdapter

MODEL = "gpt-4.1-nano"
MARKER = "TEMPLATE_ADAPTER_AUDIT_MARKER"


class NumberClassify(dspy.Signature):
    """Classify n into one label: alpha, beta, gamma, or delta.

    Current heuristic (possibly wrong):
    - alpha for multiples of 5
    - beta for even numbers
    - gamma for odd multiples of 3
    - delta otherwise
    """

    n: int = dspy.InputField()
    label: str = dspy.OutputField()


def gold_label(n: int) -> str:
    if n % 6 == 0:
        return "alpha"
    if n % 2 == 0:
        return "beta"
    if n % 3 == 0:
        return "gamma"
    return "delta"


def to_examples(numbers: list[int]) -> list[dspy.Example]:
    return [dspy.Example(n=n, label=gold_label(n)).with_inputs("n") for n in numbers]


def normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    for label in ("alpha", "beta", "gamma", "delta"):
        if label in s:
            return label
    return s.split()[0] if s else ""


def metric(gold, pred, trace=None):
    return normalize_label(pred.label) == gold.label


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    pred_label = normalize_label(pred.label)
    ok = pred_label == gold.label
    score = 1.0 if ok else 0.0

    if pred_name is None:
        return score

    feedback = (
        "Correct."
        if ok
        else (
            f"Incorrect for n={gold.n}: expected '{gold.label}', got '{pred.label}'. "
            "Use this exact rule: alpha if divisible by 6; beta if divisible by 2 only; "
            "gamma if divisible by 3 only; delta otherwise."
        )
    )
    return {"score": score, "feedback": feedback}


def make_adapter() -> TemplateAdapter:
    return TemplateAdapter(
        messages=[
            {
                "role": "system",
                "content": (
                    f"[{MARKER}]\n"
                    "{instruction}\n\n"
                    "Return exactly one lowercase label: alpha, beta, gamma, or delta."
                ),
            },
            {"role": "user", "content": "n: {n}"},
        ],
        parse_mode="full_text",
    )


def json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return str(x)


def predictor_state(program) -> dict[str, Any]:
    named = list(program.named_predictors())
    if not named:
        sig = getattr(program, "signature", None)
        return {
            "predictor_name": None,
            "instruction": getattr(sig, "instructions", None),
            "demo_count": len(getattr(program, "demos", []) or []),
        }

    name, pred = named[0]
    sig = getattr(pred, "signature", None)
    return {
        "predictor_name": name,
        "instruction": getattr(sig, "instructions", None),
        "demo_count": len(getattr(pred, "demos", []) or []),
    }


def eval_accuracy(program, lm: dspy.LM, dataset: list[dspy.Example]) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    correct = 0
    with dspy.context(lm=lm):
        for ex in dataset:
            pred = program(**ex.inputs())
            pred_label = normalize_label(pred.label)
            ok = pred_label == ex.label
            correct += int(ok)
            rows.append({
                "n": ex.n,
                "gold": ex.label,
                "pred_raw": pred.label,
                "pred_norm": pred_label,
                "correct": ok,
            })
    return correct / max(len(dataset), 1), rows


def capture_probe_call(program, lm: dspy.LM, n: int) -> dict[str, Any]:
    lm.history.clear()
    with dspy.context(lm=lm):
        pred = program(n=n)

    entry = lm.history[-1] if lm.history else {}
    return {
        "probe_n": n,
        "prediction_raw": getattr(pred, "label", str(pred)),
        "prediction_norm": normalize_label(getattr(pred, "label", str(pred))),
        "messages": json_safe(entry.get("messages", [])),
        "outputs": json_safe(entry.get("outputs")),
        "kwargs": json_safe(entry.get("kwargs", {})),
    }


@dataclass
class AuditResult:
    name: str
    baseline_acc: float
    compiled_acc: float
    instruction_changed: bool
    demo_count_before: int
    demo_count_after: int
    report_path: str


def run_bootstrap(trainset, valset, outdir: Path) -> AuditResult:
    lm = dspy.LM(MODEL, temperature=0, max_tokens=300)
    student = Predict(NumberClassify, adapter=make_adapter())

    before = predictor_state(student)
    baseline_acc, baseline_rows = eval_accuracy(student, lm, valset)
    baseline_probe = capture_probe_call(student, lm, n=30)

    tele = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=8,
        max_rounds=2,
    )
    with dspy.context(lm=lm):
        compiled = tele.compile(student, trainset=trainset)

    after = predictor_state(compiled)
    compiled_acc, compiled_rows = eval_accuracy(compiled, lm, valset)
    compiled_probe = capture_probe_call(compiled, lm, n=30)

    report = {
        "optimizer": "BootstrapFewShot",
        "model": MODEL,
        "marker": MARKER,
        "before": before,
        "after": after,
        "instruction_changed": before["instruction"] != after["instruction"],
        "baseline_accuracy": baseline_acc,
        "compiled_accuracy": compiled_acc,
        "baseline_eval_rows": baseline_rows,
        "compiled_eval_rows": compiled_rows,
        "baseline_probe": baseline_probe,
        "compiled_probe": compiled_probe,
    }

    path = outdir / "bootstrap_report.json"
    path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False))

    return AuditResult(
        name="bootstrap",
        baseline_acc=baseline_acc,
        compiled_acc=compiled_acc,
        instruction_changed=report["instruction_changed"],
        demo_count_before=before["demo_count"],
        demo_count_after=after["demo_count"],
        report_path=str(path),
    )


def run_miprov2(trainset, valset, outdir: Path) -> AuditResult:
    lm = dspy.LM(MODEL, temperature=0, max_tokens=500)
    student = Predict(NumberClassify, adapter=make_adapter())

    before = predictor_state(student)
    baseline_acc, baseline_rows = eval_accuracy(student, lm, valset)
    baseline_probe = capture_probe_call(student, lm, n=30)

    tele = dspy.MIPROv2(
        metric=metric,
        prompt_model=lm,
        task_model=lm,
        auto=None,
        num_candidates=6,
        max_bootstrapped_demos=3,
        max_labeled_demos=8,
        num_threads=1,
        seed=13,
        verbose=False,
    )
    with dspy.context(lm=lm):
        compiled = tele.compile(
            student,
            trainset=trainset,
            valset=valset,
            num_trials=4,
            minibatch=False,
            requires_permission_to_run=False,
        )

    after = predictor_state(compiled)
    compiled_acc, compiled_rows = eval_accuracy(compiled, lm, valset)
    compiled_probe = capture_probe_call(compiled, lm, n=30)

    report = {
        "optimizer": "MIPROv2",
        "model": MODEL,
        "marker": MARKER,
        "before": before,
        "after": after,
        "instruction_changed": before["instruction"] != after["instruction"],
        "baseline_accuracy": baseline_acc,
        "compiled_accuracy": compiled_acc,
        "baseline_eval_rows": baseline_rows,
        "compiled_eval_rows": compiled_rows,
        "baseline_probe": baseline_probe,
        "compiled_probe": compiled_probe,
    }

    path = outdir / "miprov2_report.json"
    path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False))

    return AuditResult(
        name="miprov2",
        baseline_acc=baseline_acc,
        compiled_acc=compiled_acc,
        instruction_changed=report["instruction_changed"],
        demo_count_before=before["demo_count"],
        demo_count_after=after["demo_count"],
        report_path=str(path),
    )


def run_gepa(trainset, valset, outdir: Path) -> AuditResult | None:
    if not hasattr(dspy, "GEPA"):
        return None

    lm = dspy.LM(MODEL, temperature=0, max_tokens=500)
    student = Predict(NumberClassify, adapter=make_adapter())

    before = predictor_state(student)
    baseline_acc, baseline_rows = eval_accuracy(student, lm, valset)
    baseline_probe = capture_probe_call(student, lm, n=30)

    tele = dspy.GEPA(
        metric=gepa_metric,
        max_metric_calls=30,
        reflection_lm=lm,
        skip_perfect_score=False,
        num_threads=1,
        use_wandb=False,
        use_mlflow=False,
        track_stats=False,
        seed=13,
    )

    with dspy.context(lm=lm):
        compiled = tele.compile(student, trainset=trainset, valset=valset)

    after = predictor_state(compiled)
    compiled_acc, compiled_rows = eval_accuracy(compiled, lm, valset)
    compiled_probe = capture_probe_call(compiled, lm, n=30)

    report = {
        "optimizer": "GEPA",
        "model": MODEL,
        "marker": MARKER,
        "before": before,
        "after": after,
        "instruction_changed": before["instruction"] != after["instruction"],
        "baseline_accuracy": baseline_acc,
        "compiled_accuracy": compiled_acc,
        "baseline_eval_rows": baseline_rows,
        "compiled_eval_rows": compiled_rows,
        "baseline_probe": baseline_probe,
        "compiled_probe": compiled_probe,
    }

    path = outdir / "gepa_report.json"
    path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False))

    return AuditResult(
        name="gepa",
        baseline_acc=baseline_acc,
        compiled_acc=compiled_acc,
        instruction_changed=report["instruction_changed"],
        demo_count_before=before["demo_count"],
        demo_count_after=after["demo_count"],
        report_path=str(path),
    )


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    train_numbers = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 15, 18, 20, 22, 27, 28, 33]
    val_numbers = [7, 14, 16, 21, 24, 25, 26, 30, 35, 36]

    trainset = to_examples(train_numbers)
    valset = to_examples(val_numbers)

    outdir = Path("artifacts") / "optimizer_audit" / datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[AuditResult] = []

    print(f"Running optimizer audit with model={MODEL}")
    print(f"Output dir: {outdir}")

    results.append(run_bootstrap(trainset, valset, outdir))
    results.append(run_miprov2(trainset, valset, outdir))
    gepa_res = run_gepa(trainset, valset, outdir)
    if gepa_res is not None:
        results.append(gepa_res)

    summary = {
        r.name: {
            "baseline_accuracy": r.baseline_acc,
            "compiled_accuracy": r.compiled_acc,
            "instruction_changed": r.instruction_changed,
            "demo_count_before": r.demo_count_before,
            "demo_count_after": r.demo_count_after,
            "report_path": r.report_path,
        }
        for r in results
    }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Summary ===")
    for r in results:
        print(
            f"- {r.name}: baseline={r.baseline_acc:.2f}, compiled={r.compiled_acc:.2f}, "
            f"instruction_changed={r.instruction_changed}, demos {r.demo_count_before}->{r.demo_count_after}"
        )
        print(f"  report: {r.report_path}")

    print(f"\nSummary JSON: {summary_path}")


if __name__ == "__main__":
    main()
