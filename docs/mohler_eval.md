# Mohler short-answer grading (2273 answers, 81 questions)

Scores are 0-5. Gradient and the baseline are compared with the average of the two human graders.

| System | MAE (lower is better) | Pearson r | Within 1 point | QWK |
|---|---|---|---|---|
| Human: grader 2 vs grader 1 (vs each other) | 0.89 | 0.55 | 79% | 0.49 |
| Baseline: always predict 4.5 | 0.81 | n/a | 80% | 0.00 |
| Gradient: keyword coverage from reference answer | 2.22 | 0.51 | 28% | 0.17 |

Data: Mohler, Bunescu and Mihalcea (2011), via https://huggingface.co/datasets/nkazi/MohlerASAG (CC-BY-4.0).
Rubrics are auto-built from each question's reference answer; teacher-written rubrics may do better.
No thresholds were tuned on this data.