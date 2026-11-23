# Gradient

> An AI co-pilot for teachers: RAG-grounded lesson plans and worksheets that cut prep time, with the teacher always in the loop.

**Status:** Phase 1 in progress (see [roadmap](docs/roadmap.md)).

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

## Project layout
```
gradient/
  rag.py         # chunking + retrieval (TF-IDF in Phase 1)
  generator.py   # prompt building, grounding check, LLM backends
app.py           # Gradio UI
data/sample/     # placeholder curriculum text (replace with your own NCERT notes)
tests/
docs/roadmap.md
```
