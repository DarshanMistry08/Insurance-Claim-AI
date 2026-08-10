"""
scripts/generate_report.py

Reads data/eval_results.json and renders a clean Markdown evaluation report,
saved as EVAL_REPORT.md in the project root.

Usage:
    python scripts/generate_report.py
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH  = PROJECT_ROOT / "data" / "eval_results.json"
OUTPUT_PATH = PROJECT_ROOT / "EVAL_REPORT.md"

FIELD_DESCRIPTIONS = {
    "claimant_name": "Name of the claimant as read from the document",
    "incident_type": "Type/category of incident",
    "incident_date": "Date the incident occurred",
    "amount_claimed": "Total amount claimed (numeric)",
}

def em_badge(em: float) -> str:
    pct = em * 100
    if pct >= 70: return f"🟢 **{pct:.1f}%**"
    if pct >= 40: return f"🟡 **{pct:.1f}%**"
    return f"🔴 **{pct:.1f}%**"

def f1_badge(f1: float) -> str:
    if f1 >= 0.7: return f"🟢 {f1:.3f}"
    if f1 >= 0.4: return f"🟡 {f1:.3f}"
    return f"🔴 {f1:.3f}"

def flag_badge(acc: float) -> str:
    pct = acc * 100
    if pct >= 80: return f"🟢 {pct:.1f}%"
    if pct >= 50: return f"🟡 {pct:.1f}%"
    return f"🔴 {pct:.1f}%"

def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run evaluate.py first.")
        return

    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    agg = data["aggregate"]
    doc_results = data["doc_results"]
    eval_fields = data["evaluated_fields"]
    num_docs = data["num_docs"]
    overall = agg["OVERALL"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []

    # ── Header ──
    lines.append("# Claims AI Pipeline — Evaluation Report")
    lines.append(f"\n> Generated: {ts} | Documents evaluated: {num_docs} | Model: `naver-clova-ix/donut-base-finetuned-docvqa` + `llama3.2` via Ollama")
    lines.append("\n---\n")

    # ── Executive Summary ──
    lines.append("## Executive Summary\n")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    lines.append(f"| Overall Exact Match | {em_badge(overall['exact_match'])} |")
    lines.append(f"| Overall Token F1 | {f1_badge(overall['token_f1'])} |")
    lines.append(f"| Flag Accuracy (clean claims) | {flag_badge(overall['flag_accuracy'])} |")
    lines.append(f"| Documents Evaluated | {num_docs} |")
    lines.append(f"| Fields Evaluated | {', '.join(f'`{f}`' for f in eval_fields)} |")
    lines.append("")

    # ── Per-field table ──
    lines.append("## Per-Field Extraction Accuracy\n")
    lines.append("| Field | Description | Exact Match | Token F1 |")
    lines.append("|-------|-------------|:-----------:|:--------:|")
    for field in eval_fields:
        m = agg[field]
        desc = FIELD_DESCRIPTIONS.get(field, "")
        lines.append(f"| `{field}` | {desc} | {em_badge(m['exact_match'])} | {f1_badge(m['token_f1'])} |")
    lines.append("")

    # ── Interpretation ──
    lines.append("## Interpretation\n")
    lines.append("These metrics reflect **zero-shot DocVQA performance** on FUNSD scanned forms. FUNSD documents are not insurance claim forms — they are generic scanned business documents, so the extraction is deliberately challenging (no fine-tuning was performed).\n")

    best_field = max(eval_fields, key=lambda f: agg[f]["exact_match"])
    worst_field = min(eval_fields, key=lambda f: agg[f]["exact_match"])
    lines.append(f"- **Best performing field**: `{best_field}` ({agg[best_field]['exact_match']*100:.1f}% EM)")
    lines.append(f"- **Hardest field**: `{worst_field}` ({agg[worst_field]['exact_match']*100:.1f}% EM)")
    lines.append(f"- **Flag accuracy**: The pipeline correctly avoided raising spurious flags on clean synthetic claims {overall['flag_accuracy']*100:.0f}% of the time.")
    lines.append("")

    # ── Per-document breakdown ──
    lines.append("## Per-Document Breakdown\n")
    for dr in doc_results:
        doc_id = dr["doc_id"]
        if dr.get("pipeline_error"):
            lines.append(f"### `{doc_id}` — ⚠️ Pipeline Error\n")
            lines.append(f"```\n{dr['pipeline_error']}\n```\n")
            continue

        em_scores = [s["exact_match"] for s in dr["field_scores"].values()]
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
        icon = "✅" if avg_em >= 0.5 else "⚠️" if avg_em >= 0.25 else "❌"
        lines.append(f"### `{doc_id}` {icon} — Avg EM: {avg_em*100:.0f}%\n")

        lines.append("| Field | Gold | Predicted | EM | F1 |")
        lines.append("|-------|------|-----------|:--:|:--:|")
        for field, s in dr["field_scores"].items():
            check = "✓" if s["exact_match"] else "✗"
            gold_display = s["gold"] if s["gold"] else "*(empty)*"
            pred_display = s["pred"] if s["pred"] else "*(empty)*"
            lines.append(f"| `{field}` | {gold_display} | {pred_display} | {check} | {s['token_f1']:.2f} |")

        if dr["flags_raised"]:
            lines.append(f"\n**Flags raised:** {', '.join(f'`{fl}`' for fl in dr['flags_raised'])}")
        else:
            lines.append("\n**Flags raised:** None ✅")

        lines.append(f"\n**Fraud score:** {dr['fraud_score'] or 'N/A'}")
        lines.append(f"\n**Is duplicate:** {'Yes ⚠️' if dr['is_duplicate'] else 'No ✅'}")
        lines.append("")

    # ── Limitations ──
    lines.append("## Limitations & Next Steps\n")
    lines.append("| Limitation | Recommended Fix |")
    lines.append("|------------|-----------------|")
    lines.append("| Golden labels are synthetic (not hand-labelled) | Manually annotate 20-30 real claim forms |")
    lines.append("| Donut model not fine-tuned on insurance forms | Fine-tune on 100+ annotated claim images |")
    lines.append("| FUNSD documents are generic, not insurance claims | Use real insurance form dataset (e.g., CORD) |")
    lines.append("| Ollama fraud classification is zero-shot | Add few-shot examples to system prompt |")
    lines.append("| Evaluation only covers 4 of 10 fields | Add address, policy number, adjuster to eval |")
    lines.append("")

    # ── How to reproduce ──
    lines.append("## How to Reproduce\n")
    lines.append("```bash")
    lines.append("# 1. Start services")
    lines.append("docker compose up -d")
    lines.append("")
    lines.append("# 2. Run evaluation")
    lines.append("python scripts/evaluate.py --limit 5")
    lines.append("")
    lines.append("# 3. Generate this report")
    lines.append("python scripts/generate_report.py")
    lines.append("```\n")

    report = "\n".join(lines)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"[OK] Report written to: {OUTPUT_PATH}")
    print(f"     {len(doc_results)} docs | Overall EM: {overall['exact_match']*100:.1f}% | Flag Acc: {overall['flag_accuracy']*100:.0f}%")

if __name__ == "__main__":
    main()
