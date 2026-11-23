# Groundedness (google/flan-t5-base, 100 random SQuAD validation passages, seed 42)

Question supported = at least 50% of its content words appear in the passage; answer supported = at least 80%.

| Model | Questions supported | Answers supported | Pass both (kept) |
|---|---|---|---|
| base | 99% | 91% | 90% |
| fine-tuned (LoRA) | 100% | 99% | 99% |

Pairs failing the check are dropped by Gradient and never shown to the teacher.
This is a lexical check: it catches off-passage content, not every wrong answer.

## Examples the filter rejected

**base**
- Q: How many years after Saudi and Kuwaiti governments declared Japan a "nonfriendly" country did Japan receive a 5% production cut? | A: 2 (question support 0.85, answer support 0.00)
- Q: How many years after Luther advised kindness toward the Jews in That Jesus Christ was Born a Jew and also aimed to convert them to Christianity did Eric Metaxas claim that Luther's attitude toward Jews "unraveled along with his health?" | A: 10 (question support 0.96, answer support 0.00)
- Q: How many years passed between the first variant entering Europe through the port of Marseille and spreading through France over the next two years, eventually reaching England in the spring of 1349, where it spread through the country in three epidemics? | A: 2 (question support 0.88, answer support 0.00)
- Q: How many in percent weren't married couples? | A: 98.8 (question support 0.50, answer support 0.50)
- Q: What is Stephen Eilmann's opinion on law breaking? | A: public civil disobedience rather than simply covert lawbreaking (question support 0.40, answer support 1.00)

**fine-tuned (LoRA)**
- Q: How many percent of households were not made up of individuals? | A: 99.7 (question support 0.80, answer support 0.50)
