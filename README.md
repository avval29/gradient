# Gradient

[![tests](https://github.com/avval29/gradient/actions/workflows/tests.yml/badge.svg)](https://github.com/avval29/gradient/actions/workflows/tests.yml)

> An AI co-pilot for teachers: RAG-grounded lesson plans and worksheets that cut prep time, with the teacher always in the loop.

**Status:** Phase 4a done (typed-answer grading); handwriting OCR next (see [roadmap](docs/roadmap.md)).

## Results at a glance
All measured on my own machine; details, caveats and reproduction commands are in the sections below and in `docs/`.

| What | Result |
|---|---|
| LoRA fine-tune of FLAN-T5-base for question generation (0.36% of parameters trained) | ROUGE-L 43.1 -> 45.4, BLEU 18.1 -> 25.0 on 100 random SQuAD passages (single run) |
| Groundedness filter (question/answer supported by the passage) | kept 90% of base-model pairs vs 99% of fine-tuned pairs (lexical check) |
| Answer grading on 2,273 real student answers, unseen questions, 10 splits | Pearson 0.54 +- 0.02, QWK 0.44 +- 0.03 (answer-length baseline 0.11 / 0.03; two human graders 0.56 / 0.51) |

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

## Groundedness check
Every generated question and answer is verified against the passage it came from; pairs with
content words missing from the passage are dropped and counted (shown in the app status line).
```bash
python scripts/evaluate_grounding.py --adapter models/gradient-qg-lora --n 100   # writes docs/grounding.md
```
It is a lexical check: it catches invented entities and off-passage answers, not every wrong answer.

## Grading assistant (draft only)
The **Grade (draft)** tab compares a typed student answer with a teacher-written rubric and suggests a
score with evidence (found / missing terms, matching sentence). The teacher must approve or edit before
anything is saved, and every correction is logged to `data/private/grading_log.jsonl` (gitignored: it
contains student work).

Rubric format, one criterion per line: `points | name | description | keywords` (use `|` inside the
keywords for alternatives). Matching ignores plurals and tolerates small spelling mistakes. Add
`GRADIENT_GRADER_SEMANTIC=1` for embedding-based matching of paraphrases.

Uncertain drafts (borderline evidence, very short answers) are flagged for review. Known limit: wording
the rubric did not anticipate can be under-credited, which is why the teacher stays in the loop.

**Measuring accuracy:** put real student answers and the teacher's marks in a CSV (`answer,score`) and run
```bash
python scripts/evaluate_grading.py --rubric your_rubric.json --csv your_graded_answers.csv --out docs/grading_eval.md
```
`examples/` contains hand-written demo data only; do not quote accuracy from it.

## Benchmarking the grader on real data
`scripts/evaluate_mohler.py` scores Gradient's matching engine on the public Mohler short-answer dataset
(2,273 real student answers to computer-science questions, each marked 0-5 by two graders). Questions are split
into a development half and a disjoint test half; calibration is fitted on the development half only and every
reported number is measured on the unseen test questions. It is compared with two human graders agreeing with each
other, an "always predict the median" baseline, and an answer-length baseline.
```bash
python scripts/evaluate_mohler.py --n-splits 10 --out docs/mohler_eval.md   # mean ± sd over 10 random question splits
python scripts/evaluate_mohler.py --n-splits 10 --semantic --out docs/mohler_semantic.md   # also test meaning-based similarity
```
Findings: raw keyword-coverage scores rank answers about as well as two humans agree (Pearson ~0.5) but are
badly scaled (they under-score by about two points), so they are calibrated before use. Meaning-based similarity
(`--semantic`) alone did worse than keywords (QWK 0.37 vs 0.44), and combining them gave no clear gain, so semantic
matching stays optional. Rubrics are auto-built
from each question's reference answer, so this measures the matching engine, not what a teacher-written rubric could
achieve. Data: Mohler, Bunescu and Mihalcea (2011), via
[nkazi/MohlerASAG](https://huggingface.co/datasets/nkazi/MohlerASAG) (CC-BY-4.0).

## Handwritten answers
In the **Grade (draft)** tab, upload a photo of a handwritten answer. Gradient cuts the page into text lines,
reads each with [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten), and puts the text in the answer
box with low-confidence lines flagged. **The teacher corrects the text before it is graded**, so a reading
mistake never silently lowers a mark.
```bash
python scripts/read_handwriting.py photo.jpg --segment-only --save-lines /tmp/lines   # check the line cutting (no model)
python scripts/read_handwriting.py photo.jpg                                          # transcribe
python scripts/evaluate_ocr.py --dir data/private/handwriting --rubric your_rubric.json --out docs/ocr_eval.md
```
Limits: TrOCR is trained on single lines of English handwriting; line cutting assumes dark ink on lighter paper, a
roughly straight page and clear gaps between lines. Keep photos of real students' work out of git
(`data/private/` is ignored). Set `GRADIENT_OCR_MODEL` to try another model.

## Feedback loop and monitoring
Teachers can rate generated drafts (thumbs up/down with a comment), and every approved grade is logged with the
teacher's correction. `scripts/analyze_feedback.py` turns those logs into a report: how often teachers change the
suggested score, in which direction, whether the review flags catch the drafts that needed changing, which questions
get corrected most (a sign of a weak rubric), and, with 30+ graded answers, a suggested calibration. Suggestions are
shown to a human and never applied automatically.
```bash
python scripts/analyze_feedback.py --out docs/feedback_report.md
```
Logs live in `data/private/` (gitignored) because they contain student work; the report contains aggregates only.

## Project layout
```
gradient/
  rag.py         # chunking + TF-IDF retriever
  embeddings.py  # dense (sentence-transformers) retriever
  ingest.py      # .txt/.md/.pdf loader
  generator.py   # prompts, grounding check, LLM backends (+ LoRA adapter loading)
  qa.py          # quiz mode: question + answer per passage
  asag.py        # Mohler benchmark helpers (grouping, auto-rubric, scoring)
  ocr.py         # handwriting photo -> text lines -> TrOCR, with confidence
  feedback.py    # logs + summaries of ratings and teacher corrections
  grading.py     # rubric-based draft grading, teacher approval, feedback log
  grounding.py   # question/answer support check against the source passage
  data.py        # SQuAD -> training examples
  evaluate.py    # ROUGE / BLEU
  config.py      # env-driven factory
scripts/          # finetune_lora.py, evaluate_model.py, evaluate_grounding.py, evaluate_grading.py, evaluate_mohler.py, read_handwriting.py, evaluate_ocr.py, analyze_feedback.py
examples/         # demo rubric and answers (not a benchmark)
app.py           # Gradio UI
data/sample/     # placeholder curriculum text (replace with your own NCERT notes)
tests/
docs/roadmap.md
```
