# Evaluation: question generation (google/flan-t5-base, 100 SQuAD validation rows)

| Model | ROUGE-1 | ROUGE-L | BLEU |
|---|---|---|---|
| base | 27.4 | 25.8 | 7.8 |
| fine-tuned (LoRA) | 31.6 | 29.5 | 9.9 |

Each context has one reference question, but many valid questions exist, so absolute
scores are low. Compare models against each other, not against 100.

## Sample outputs

**Reference:** Which NFL team represented the AFC at Super Bowl 50?  
**base:** Which team won Super Bowl 50?  
**fine-tuned (LoRA):** How many games did the Denver Broncos win in Super Bowl 50?  

**Reference:** Which NFL team represented the NFC at Super Bowl 50?  
**base:** Which team won Super Bowl 50?  
**fine-tuned (LoRA):** How many games did the Denver Broncos win in Super Bowl 50?  

**Reference:** Where did Super Bowl 50 take place?  
**base:** Which team won Super Bowl 50?  
**fine-tuned (LoRA):** How many games did the Denver Broncos win in Super Bowl 50?  
