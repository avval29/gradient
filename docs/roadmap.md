# Roadmap

| Phase | Goal | Course link | Status |
|---|---|---|---|
| 1 | RAG-grounded worksheet / lesson-plan generator + Gradio UI | Module 3 (RAG), Module 4 (Gradio) | Done (offline backend) |
| 2 | Real LLM backend (FLAN-T5), dense embeddings, load real curriculum notes | Lab 1, Module 2 (HF) | Code done; tune on real notes |
| 3 | LoRA fine-tune for question generation/answering; ROUGE/BLEU base-vs-tuned evaluation | Labs 2, 3, 4 | Training and evaluation results recorded; see docs/results.md |
| 3b | Groundedness filter on generated quizzes + measured rejection rate | Module 3 (RAG), Interpretable AI | Evaluation results recorded; see docs/grounding.md |
| 4a | Rubric-based draft grading of typed answers, teacher approval, override log | Module 3 (evaluation), Interpretable AI | Done |
| 4a-eval | Benchmark the grader on the Mohler dataset vs human agreement and a median baseline | Module 3 (evaluation) | Benchmark completed; see docs/mohler_eval.md |
| 4b | Handwritten answer photo -> line cutting -> TrOCR -> teacher correction -> grader | Module 1, Lab 3 | Code implemented; real-photo testing found segmentation issues that need improvement |
| 5 | Voice dictation of lesson notes (STT) + regional-language output | Lab 4 (TTS/STT) | Planned |
| 6 | Feedback loop (thumbs + teacher corrections), monitoring report | Module 4 | Code done; needs real usage data |
| 7 | Public demo on a Hugging Face Space (demo mode, no data saved) | Module 4 (deployment) | Code done; publish and add the link |

## Principles
- Never auto-finalize a grade.
- Refuse rather than hallucinate when the topic isn't in the corpus.
