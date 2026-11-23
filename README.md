# Gradient

> An AI co-pilot for teachers: RAG-grounded lesson plans and worksheets that cut prep time, with the teacher always in the loop.

**Status:** Phase 3 in progress (see [roadmap](docs/roadmap.md)).

## Problem
Teachers spend a large share of their week on grading, worksheet creation and lesson planning. Existing tools are mostly English-first and not aligned to Indian curricula.

## Design principles
1. **Grounded, not guessed.** Content is generated only from retrieved curriculum text. If nothing relevant is found, Gradient refuses instead of inventing facts.
2. **Teacher in the loop.** Gradient drafts; the teacher approves or edits. It never auto-finalizes a grade.
3. **Swappable LLM backend.** Retrieval and prompting are decoupled from the model, so you can move from FLAN-T5 to a fine-tuned LLaMA/LoRA model without rewriting the app.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                # run tests
python app.py         # launch the Gradio UI
```

## Using a real model and your own notes (Phase 2)
```bash
pip install -r requirements-llm.txt
# put your own .txt / .md / .pdf notes in data/private/ (it is gitignored)
GRADIENT_DATA=data/private GRADIENT_RETRIEVER=dense GRADIENT_BACKEND=hf python app.py
```
| Variable | Options | Default |
|---|---|---|
| `GRADIENT_RETRIEVER` | `tfidf`, `dense` | `tfidf` |
| `GRADIENT_BACKEND` | `echo`, `hf` | `echo` |
| `GRADIENT_MODEL` | any HF seq2seq model id | `google/flan-t5-base` |
| `GRADIENT_DATA` | folder of curriculum files | `data/sample` |

## Fine-tuning and evaluation (Phase 3)
Gradient's quiz mode writes one question and answer per retrieved passage. To teach FLAN-T5 that task,
fine-tune it with LoRA on SQuAD, then measure the gain:
```bash
python scripts/finetune_lora.py --model google/flan-t5-small --max-train 64   # quick smoke test
python scripts/finetune_lora.py --max-train 1000                              # real run
python scripts/evaluate_model.py --adapter models/gradient-qg-lora --n 100    # writes docs/results.md
GRADIENT_BACKEND=hf GRADIENT_ADAPTER=models/gradient-qg-lora python app.py    # use the adapter
```
Results are in [docs/results.md](docs/results.md) once you have run the evaluation.

## Project layout
```
gradient/
  rag.py         # chunking + TF-IDF retriever
  embeddings.py  # dense (sentence-transformers) retriever
  ingest.py      # .txt/.md/.pdf loader
  generator.py   # prompts, grounding check, LLM backends (+ LoRA adapter loading)
  qa.py          # quiz mode: question + answer per passage
  data.py        # SQuAD -> training examples
  evaluate.py    # ROUGE / BLEU
  config.py      # env-driven factory
scripts/          # finetune_lora.py, evaluate_model.py
app.py           # Gradio UI
data/sample/     # placeholder curriculum text (replace with your own NCERT notes)
tests/
docs/roadmap.md
```
