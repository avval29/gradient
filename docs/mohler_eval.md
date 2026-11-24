# Mohler short-answer grading: mean ± sd over 10 random question splits

Each split fits calibrations on about 40 questions and tests on the other ~41 (seeds 42-51). Splits reuse the same 81 questions, so they are not independent: the spread shows sensitivity to the split, not uncertainty about new subjects. Scores are 0-5, compared with the average of the two human graders.

| System | MAE (lower is better) | Pearson r | Within 1 point | QWK |
|---|---|---|---|---|
| Human: grader 2 vs grader 1 | 0.85 ± 0.06 | 0.56 ± 0.05 | 80 ± 1% | 0.51 ± 0.05 |
| Baseline: always predict the dev median | 0.80 ± 0.05 | n/a | 81 ± 2% | 0.00 ± 0.00 |
| Baseline: answer length (calibrated) | 0.90 ± 0.04 | 0.11 ± 0.05 | 91 ± 2% | 0.03 ± 0.02 |
| Gradient: raw keyword-coverage score | 2.21 ± 0.07 | 0.51 ± 0.02 | 28 ± 2% | 0.17 ± 0.02 |
| Gradient: calibrated on dev questions | 0.66 ± 0.03 | 0.54 ± 0.02 | 88 ± 2% | 0.44 ± 0.03 |

Data: Mohler, Bunescu and Mihalcea (2011), via https://huggingface.co/datasets/nkazi/MohlerASAG (CC-BY-4.0).
Rubrics are auto-built from each question's reference answer; teacher-written rubrics may do better.
The human row compares two individual graders, a noisier target than the two-grader average used for the other rows.
'Within 1 point' flatters any guess near the top because marks are skewed high; rely on correlation and QWK.