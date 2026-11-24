# Mohler short-answer grading: 1161 test answers on 41 unseen questions

Calibrations were fitted on 1112 answers from 40 other questions (seed 42). Scores are 0-5, compared with the average of the two human graders.

| System | MAE (lower is better) | Pearson r | Within 1 point | QWK |
|---|---|---|---|---|
| Human: grader 2 vs grader 1 | 0.89 | 0.51 | 79% | 0.46 |
| Baseline: always predict 4.5 (dev median) | 0.85 | n/a | 79% | 0.00 |
| Baseline: answer length (calibrated) | 0.94 | 0.16 | 88% | 0.06 |
| Gradient: raw keyword-coverage score | 2.25 | 0.50 | 28% | 0.17 |
| Gradient: calibrated on dev questions | 0.73 | 0.53 | 86% | 0.41 |

Data: Mohler, Bunescu and Mihalcea (2011), via https://huggingface.co/datasets/nkazi/MohlerASAG (CC-BY-4.0).
Rubrics are auto-built from each question's reference answer; teacher-written rubrics may do better.
The human row compares two individual graders, a noisier target than the two-grader average used for the other rows.