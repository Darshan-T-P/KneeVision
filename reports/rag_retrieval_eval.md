# RAG retrieval evaluation (top-4, 26 queries)

| Configuration | grade_hit | hit | recall | MRR |
|---|---|---|---|---|
| A: base corpus, original retriever | 0.88 | 0.88 | 0.60 | 0.71 |
| B: +AAOS, original retriever | 0.35 | 0.77 | 0.50 | 0.67 |
| C: +AAOS, current retriever (+context match) | 1.00 | 1.00 | 0.89 | 1.00 |

## Topic-specific labels only (20 queries with context-specific targets)

| Configuration | topic_hit | topic_recall | topic_MRR |
|---|---|---|---|
| A: base corpus, original retriever | 0.40 | 0.27 | 0.27 |
| B: +AAOS, original retriever | 0.80 | 0.73 | 0.71 |
| C: +AAOS, current retriever (+context match) | 0.85 | 0.78 | 0.67 |

_A cannot retrieve AAOS chunks, so its topic scores are capped for AAOS-labelled queries._

## Config C: queries whose topic-specific target was NOT retrieved

- KL1 [weight] "BMI 33, mild morning stiffness" — wanted ['kl0_kl1_normal_doubtful.md::Recommendations', 'general_exercise_and_lifestyle.md::Weight management', 'aaos_oak3_recommendations.md::Weight Loss Intervention']; got ['kl0_kl1_normal_doubtful.md::Recommendations', 'aaos_oak3_recommendations.md::Partial Meniscectomy', 'aaos_oak3_recommendations.md::Oral/Dietary Supplements', 'aaos_oak3_recommendations.md::Acupuncture']
- KL2 [exercise] "asks about tai chi, yoga or swimming instead of the gym" — wanted ['kl2_mild.md::Recommendations (core treatments)', 'general_exercise_and_lifestyle.md::Exercise', 'aaos_oak3_recommendations.md::Supervised Exercise']; got ['kl2_mild.md::Recommendations (core treatments)', 'kl2_mild.md::Not yet indicated at this stage', 'aaos_oak3_recommendations.md::Partial Meniscectomy', 'aaos_oak3_recommendations.md::Oral/Dietary Supplements']
- KL2 [topical] "prefers a gel or cream rather than tablets for knee pain" — wanted ['kl2_mild.md::Recommendations (core treatments)', 'aaos_oak3_recommendations.md::Topical Treatments']; got ['kl2_mild.md::Recommendations (core treatments)', 'aaos_oak3_recommendations.md::Acupuncture', 'aaos_oak3_recommendations.md::Canes', 'aaos_oak3_recommendations.md::Percutaneous Electrical Nerve Stimulation/Pulsed Electromagnetic Field']
