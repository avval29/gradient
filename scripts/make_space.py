"""Build a self-contained folder for a Hugging Face Space (Gradio) demo of Gradient.

Usage:
  python scripts/make_space.py --out space_build                          # lite demo: no model, quiz questions are crude
  python scripts/make_space.py --out space_build --adapter USER/gradient-qg-lora   # full demo: FLAN-T5 + your LoRA adapter

The demo runs in GRADIENT_DEMO mode: nothing visitors type is saved, and photo upload is off.
Only the app, the gradient package and the sample passages are copied (no tests, logs, models or private data).
"""
import argparse
import shutil
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ["numpy>=1.24", "scikit-learn>=1.3", "pillow>=10.0"]
FULL = ["transformers>=4.40", "peft>=0.10", "--extra-index-url https://download.pytorch.org/whl/cpu", "torch"]


def build_space(out, adapter=None, gradio_version=None, model="google/flan-t5-base", root=ROOT):
    out, root = Path(out), Path(root)
    if out.exists():
        shutil.rmtree(out)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(root / "gradient", out / "gradient", ignore=ignore)
    shutil.copytree(root / "data" / "sample", out / "data" / "sample", ignore=ignore)

    env = ['os.environ.setdefault("GRADIENT_DEMO", "1")']
    if adapter:
        env += ['os.environ.setdefault("GRADIENT_BACKEND", "hf")', f'os.environ.setdefault("GRADIENT_MODEL", "{model}")',
                f'os.environ.setdefault("GRADIENT_ADAPTER", "{adapter}")']
    header = "# Demo defaults added by scripts/make_space.py\nimport os\n" + "\n".join(env) + "\n\n"
    (out / "app.py").write_text(header + (root / "app.py").read_text(encoding="utf-8"), encoding="utf-8")

    (out / "requirements.txt").write_text("\n".join(CORE + (FULL if adapter else [])) + "\n", encoding="utf-8")

    gv = gradio_version or version("gradio")  # match the Gradio you tested with
    mode = "FLAN-T5 with a LoRA adapter" if adapter else "offline lite mode (no language model)"
    (out / "README.md").write_text(f"""---
title: Gradient
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: {gv}
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Grounded quizzes and teacher-approved draft grading
---

# Gradient (demo)

An AI co-pilot for teachers: quizzes grounded in curriculum text, and rubric-based draft grading where the
teacher approves every grade. This demo runs in {mode}. Gradient does not persist visitor inputs; the hosting platform may process operational data under its own policies.
Source code, tests and evaluation results: https://github.com/avval29/gradient
""", encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="space_build")
    ap.add_argument("--adapter", default=None, help="Hub id of the LoRA adapter, e.g. user/gradient-qg-lora")
    ap.add_argument("--model", default="google/flan-t5-base")
    args = ap.parse_args(argv)
    out = build_space(args.out, args.adapter, model=args.model)
    print("Space folder built at", out)


if __name__ == "__main__":
    main()
