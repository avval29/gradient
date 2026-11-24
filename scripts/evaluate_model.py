"""Compare base vs LoRA-fine-tuned model on SQuAD validation passages (ROUGE / BLEU).

Passages are sampled at random (seeded) from the whole validation set, and every generated
question is scored against ALL reference questions written for that passage.

Usage: python scripts/evaluate_model.py --adapter models/gradient-qg-lora --n 100
Writes docs/results.md so the numbers can go in your README / resume.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradient.evaluate import score_multi  # noqa: E402
from gradient.generator import HFBackend  # noqa: E402
from gradient.qa import qg_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--adapter", default="models/gradient-qg-lora")
    ap.add_argument("--n", type=int, default=100, help="number of distinct passages")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="docs/results.md")
    args = ap.parse_args()

    import random

    from datasets import load_dataset

    groups: dict[str, list[str]] = {}
    for r in load_dataset("rajpurkar/squad", split="validation"):
        groups.setdefault(r["context"], []).append(r["question"])
    contexts = random.Random(args.seed).sample(sorted(groups), args.n)
    refs = [groups[c] for c in contexts]

    results, samples = {}, {}
    for name, adapter in [("base", None), ("fine-tuned (LoRA)", args.adapter)]:
        print("evaluating", name)
        backend = HFBackend(args.model, max_new_tokens=64, adapter=adapter)
        preds = [backend.generate(qg_prompt(c)) for c in contexts]
        results[name] = score_multi(preds, refs)
        samples[name] = preds[:3]

    lines = [f"# Evaluation: question generation ({args.model}, {args.n} random SQuAD validation passages, seed {args.seed})", "",
             "| Model | ROUGE-1 | ROUGE-L | BLEU |", "|---|---|---|---|"]
    for name, s in results.items():
        lines.append(f"| {name} | {s['rouge1']:.1f} | {s['rougeL']:.1f} | {s['bleu']:.1f} |")
    lines += ["", "Each generated question is scored against all reference questions for its passage",
              "(best ROUGE match; multi-reference BLEU). Overlap metrics reward word match, not question",
              "quality, so also read the samples below.", "", "## Sample outputs", ""]
    for i in range(3):
        lines.append(f"**Passage starts:** {contexts[i][:90]}...  ")
        lines.append(f"**Reference examples:** {' | '.join(refs[i][:2])}  ")
        for name in samples:
            lines.append(f"**{name}:** {samples[name][i]}  ")
        lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
