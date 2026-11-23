"""Measure how often each model writes questions/answers that the passage does not support.

Usage: python scripts/evaluate_grounding.py --adapter models/gradient-qg-lora --n 100
Writes docs/grounding.md.

"Pass rate" is the share of generated question+answer pairs the groundedness filter keeps.
The rest would have been shown to a teacher without the filter.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradient.data import sample_squad_passages  # noqa: E402
from gradient.generator import HFBackend  # noqa: E402
from gradient.grounding import MIN_ANSWER_SUPPORT, MIN_QUESTION_SUPPORT, check_grounding  # noqa: E402
from gradient.qa import answer_prompt, qg_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--adapter", default="models/gradient-qg-lora")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="docs/grounding.md")
    args = ap.parse_args()

    contexts = sample_squad_passages(args.n, args.seed)
    rows, rejected_examples = [], {}
    for name, adapter in [("base", None), ("fine-tuned (LoRA)", args.adapter)]:
        print("evaluating", name)
        backend = HFBackend(args.model, max_new_tokens=64, adapter=adapter)
        q_ok = a_ok = both = 0
        bad = []
        for c in contexts:
            q = backend.generate(qg_prompt(c))
            a = backend.generate(answer_prompt(q, c))
            v = check_grounding(q, a, c)
            q_ok += v.question_ok
            a_ok += v.answer_ok
            both += v.ok
            if not v.ok and len(bad) < 5:
                bad.append((q, a, v))
        n = len(contexts)
        rows.append((name, 100 * q_ok / n, 100 * a_ok / n, 100 * both / n))
        rejected_examples[name] = bad

    lines = [f"# Groundedness ({args.model}, {args.n} random SQuAD validation passages, seed {args.seed})", "",
             f"Question supported = at least {MIN_QUESTION_SUPPORT:.0%} of its content words appear in the passage; "
             f"answer supported = at least {MIN_ANSWER_SUPPORT:.0%}.", "",
             "| Model | Questions supported | Answers supported | Pass both (kept) |", "|---|---|---|---|"]
    lines += [f"| {n} | {q:.0f}% | {a:.0f}% | {b:.0f}% |" for n, q, a, b in rows]
    lines += ["", "Pairs failing the check are dropped by Gradient and never shown to the teacher.",
              "This is a lexical check: it catches off-passage content, not every wrong answer.", "",
              "## Examples the filter rejected", ""]
    for name, bad in rejected_examples.items():
        lines.append(f"**{name}**")
        for q, a, v in bad:
            lines.append(f"- Q: {q} | A: {a} (question support {v.question_support:.2f}, answer support {v.answer_support:.2f})")
        lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
