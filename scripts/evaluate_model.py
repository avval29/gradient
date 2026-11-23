"""Compare base vs LoRA-fine-tuned model on SQuAD validation questions (ROUGE / BLEU).

Usage: python scripts/evaluate_model.py --adapter models/gradient-qg-lora --n 100
Writes docs/results.md so the numbers can go in your README / resume.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradient.evaluate import score  # noqa: E402
from gradient.generator import HFBackend  # noqa: E402
from gradient.qa import qg_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--adapter", default="models/gradient-qg-lora")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="docs/results.md")
    args = ap.parse_args()

    from datasets import load_dataset

    rows = load_dataset("rajpurkar/squad", split=f"validation[:{args.n}]")
    contexts = [r["context"] for r in rows]
    refs = [r["question"] for r in rows]

    results, samples = {}, {}
    for name, adapter in [("base", None), ("fine-tuned (LoRA)", args.adapter)]:
        print("evaluating", name)
        backend = HFBackend(args.model, max_new_tokens=64, adapter=adapter)
        preds = [backend.generate(qg_prompt(c)) for c in contexts]
        results[name] = score(preds, refs)
        samples[name] = preds[:3]

    lines = [f"# Evaluation: question generation ({args.model}, {args.n} SQuAD validation rows)", "",
             "| Model | ROUGE-1 | ROUGE-L | BLEU |", "|---|---|---|---|"]
    for name, s in results.items():
        lines.append(f"| {name} | {s['rouge1']:.1f} | {s['rougeL']:.1f} | {s['bleu']:.1f} |")
    lines += ["", "Each context has one reference question, but many valid questions exist, so absolute",
              "scores are low. Compare models against each other, not against 100.", "", "## Sample outputs", ""]
    for i in range(3):
        lines.append(f"**Reference:** {refs[i]}  ")
        for name in samples:
            lines.append(f"**{name}:** {samples[name][i]}  ")
        lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
