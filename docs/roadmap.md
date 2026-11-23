# Roadmap

| Phase | Goal | Course link | Status |
|---|---|---|---|
| 1 | RAG-grounded worksheet / lesson-plan generator + Gradio UI | Module 3 (RAG), Module 4 (Gradio) | Done (offline backend) |
| 2 | Real LLM backend (FLAN-T5), dense embeddings, load real curriculum notes | Lab 1, Module 2 (HF) | Code done; tune on real notes |
| 3 | Fine-tune on worksheet/rubric data with PEFT/LoRA; BLEU/ROUGE evaluation | Labs 2, 4 | Planned |
| 4 | Handwritten answer OCR + rubric-based draft grading (teacher approves) | Module 1, Lab 3 | Planned |
| 5 | Voice dictation of lesson notes (STT) + regional-language output | Lab 4 (TTS/STT) | Planned |
| 6 | Feedback loop (flag bad drafts), logging, monitoring | Module 4 | Planned |

## Principles
- Never auto-finalize a grade.
- Refuse rather than hallucinate when the topic isn't in the corpus.
