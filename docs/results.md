# Evaluation: question generation (google/flan-t5-base, 100 random SQuAD validation passages, seed 42)

| Model | ROUGE-1 | ROUGE-L | BLEU |
|---|---|---|---|
| base | 46.6 | 43.1 | 18.1 |
| fine-tuned (LoRA) | 49.3 | 45.4 | 25.0 |

Each generated question is scored against all reference questions for its passage
(best ROUGE match; multi-reference BLEU). Overlap metrics reward word match, not question
quality, so also read the samples below.

## Sample outputs

**Passage starts:** Frederick William, Elector of Brandenburg, invited Huguenots to settle in his realms, and ...  
**Reference examples:** What German ruler invited Huguenot immigration? | What German poet was descended from Huguenots?  
**base:** Who invited Huguenots to settle in his realms?  
**fine-tuned (LoRA):** Who was the last Prime Minister of the German Democratic Republic?  

**Passage starts:** After this, Huguenots (with estimates ranging from 200,000 to 1,000,000) fled to surroundi...  
**Reference examples:** Which central European country had a Calvinist ruler? | After Huguenots fled France, their last remaining bastion was where?  
**base:** How many years did it take French troops years to hunt down and destroy all the bands of Camisards?  
**fine-tuned (LoRA):** When did the Camisards riot against the Catholic Church?  

**Passage starts:** One of the most prominent Huguenot refugees in the Netherlands was Pierre Bayle. He starte...  
**Reference examples:** Who was a prominent Huguenot in Holland? | Where did he begin teaching?  
**base:** Who was Pierre Bayle's teacher?  
**fine-tuned (LoRA):** Who was Pierre Bayle's teacher in Rotterdam?  
