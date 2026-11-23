"""Fine-tune FLAN-T5 with LoRA on SQuAD for question generation + answering.

Smoke test (~1-2 min):  python scripts/finetune_lora.py --model google/flan-t5-small --max-train 64
Real run:               python scripts/finetune_lora.py --max-train 1000 --epochs 1
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradient.data import build_examples  # noqa: E402


def pick_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--out", default="models/gradient-qg-lora")
    ap.add_argument("--max-train", type=int, default=1000, help="number of SQuAD rows (each gives 2 examples)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_dataset("rajpurkar/squad", split=f"train[:{args.max_train}]")
    examples = build_examples(rows)
    print(f"{len(rows)} SQuAD rows -> {len(examples)} training examples")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q", "v"]))
    model.print_trainable_parameters()

    device = pick_device(args.device)
    print("device:", device)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), args.batch):
            batch = examples[i:i + args.batch]
            enc = tok([e["input"] for e in batch], max_length=384, truncation=True,
                      padding=True, return_tensors="pt")
            labels = tok([e["target"] for e in batch], max_length=64, truncation=True,
                         padding=True, return_tensors="pt").input_ids
            labels[labels == tok.pad_token_id] = -100  # ignore padding in the loss
            loss = model(input_ids=enc.input_ids.to(device),
                         attention_mask=enc.attention_mask.to(device),
                         labels=labels.to(device)).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch + 1} step {step} loss {loss.item():.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    print("saved LoRA adapter to", args.out)


if __name__ == "__main__":
    main()
