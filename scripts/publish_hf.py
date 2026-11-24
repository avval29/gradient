"""Publish the LoRA adapter and the demo Space to your Hugging Face account.

Usage:
  export HF_TOKEN=...      # a WRITE-access token for your own account; never paste it into chats or commit it
  python scripts/publish_hf.py --user YOUR_HF_USERNAME
Options: --space-name gradient --adapter-name gradient-qg-lora --adapter-dir models/gradient-qg-lora
If the adapter folder is missing, only the lite demo Space is published.

The adapter was trained on SQuAD (CC BY-SA 4.0); the model card says so.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_space import build_space  # noqa: E402

CARD = """---
base_model: google/flan-t5-base
library_name: peft
license: cc-by-sa-4.0
datasets:
- rajpurkar/squad
---

# gradient-qg-lora

LoRA adapter for `google/flan-t5-base`, trained for question generation and answering as part of the Gradient
project (https://github.com/avval29/gradient).

- Method: LoRA (r=8, alpha=16, dropout 0.05, targets q and v; about 0.36% of parameters), 1,000 SQuAD training rows
  (2,000 examples), 1 epoch, one run.
- Prompt format: `generate question: <passage>` and `answer question: question: <q> context: <passage>`.
- Measured on 100 random SQuAD validation passages (multi-reference, single run): ROUGE-L 43.1 -> 45.4, BLEU 18.1 -> 25.0
  versus the base model. Overlap metrics reward word match, not question quality.
- Training data: SQuAD v1.1 (CC BY-SA 4.0), so this adapter is shared under the same licence.
- Limits: English only; small training budget; use the generated questions as drafts for a teacher to review.
"""


def publish(user, space_name="gradient", adapter_name="gradient-qg-lora", adapter_dir="models/gradient-qg-lora",
            token=None, api=None, workdir=None):
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)  # also reads HF_TOKEN or a cached login
    workdir = Path(workdir or tempfile.mkdtemp(prefix="gradient_space_"))
    adapter_id = None
    adapter_dir = Path(adapter_dir)
    if adapter_dir.exists():
        adapter_id = f"{user}/{adapter_name}"
        staged = workdir / "adapter"
        shutil.copytree(adapter_dir, staged)
        (staged / "README.md").write_text(CARD, encoding="utf-8")
        api.create_repo(adapter_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(staged), repo_id=adapter_id, repo_type="model")
    space_id = f"{user}/{space_name}"
    space_dir = build_space(workdir / "space", adapter=adapter_id)
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(space_dir), repo_id=space_id, repo_type="space")
    return f"https://huggingface.co/spaces/{space_id}", adapter_id


def main(argv=None, api=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--space-name", default="gradient")
    ap.add_argument("--adapter-name", default="gradient-qg-lora")
    ap.add_argument("--adapter-dir", default="models/gradient-qg-lora")
    args = ap.parse_args(argv)
    url, adapter = publish(args.user, args.space_name, args.adapter_name, args.adapter_dir, api=api)
    print("Adapter:", adapter or "none (lite demo)")
    print("Space:  ", url, "(the first build takes several minutes)")


if __name__ == "__main__":
    main()
