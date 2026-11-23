# Roadmap

| Phase | Goal | Course link | Status |
|---|---|---|---|
| 1 | RAG-grounded worksheet / lesson-plan generator + Gradio UI | Module 3 (RAG), Module 4 (Gradio) | Done (offline backend) |
| 2 | Real LLM backend (FLAN-T5), dense embeddings, load real curriculum notes | Lab 1, Module 2 (HF) | Code done; tune on real notes |
| 3 | LoRA fine-tune for question generation/answering; ROUGE/BLEU base-vs-tuned evaluation | Labs 2, 3, 4 | Code done; run training + record results |
| 3b | Groundedness filter on generated quizzes + measured rejection rate | Module 3 (RAG), Interpretable AI | Code done; run evaluate_grounding |
| 4a | Rubric-based draft grading of typed answers, teacher approval, override log | Module 3 (evaluation), Interpretable AI | Code done; needs real graded data |
| 4b | Handwritten answer photo -> OCR (e.g. TrOCR) -> grader | Module 1, Lab 3 | Next |
| 5 | Voice dictation of lesson notes (STT) + regional-language output | Lab 4 (TTS/STT) | Planned |
| 6 | Feedback loop (flag bad drafts), logging, monitoring | Module 4 | Planned |

## Principles
- Never auto-finalize a grade.
- Refuse rather than hallucinate when the topic isn't in the corpus.
