# RAG retrieval evaluation (top-4, 14 queries)

| Configuration | grade_hit | hit | recall | MRR |
|---|---|---|---|---|
| A: base corpus, original retriever | 0.79 | 0.86 | 0.45 | 0.64 |
| B: +AAOS, original retriever | 0.29 | 0.71 | 0.36 | 0.62 |
| C: +AAOS, current retriever (+context match) | 1.00 | 1.00 | 0.76 | 0.96 |

## Topic-specific labels only (14 queries with context-specific targets)

| Configuration | topic_hit | topic_recall | topic_MRR |
|---|---|---|---|
| A: base corpus, original retriever | 0.29 | 0.18 | 0.20 |
| B: +AAOS, original retriever | 0.50 | 0.46 | 0.46 |
| C: +AAOS, current retriever (+context match) | 0.71 | 0.61 | 0.51 |

_A cannot retrieve AAOS chunks, so its topic scores are capped for AAOS-labelled queries._

## Config C: queries whose topic-specific target was NOT retrieved

- KL2 [weight] "retired teacher, heavy build, doctor says she should shed some kilos" — wanted ['kl2_mild.md::Recommendations (core treatments)', 'general_exercise_and_lifestyle.md::Weight management', 'aaos_oak3_recommendations.md::Weight Loss Intervention']; got ['kl2_mild.md::Recommendations (core treatments)', 'kl2_mild.md::Not yet indicated at this stage', 'general_exercise_and_lifestyle.md::Self-management and education', 'general_exercise_and_lifestyle.md::Exercise']
- KL1 [sleep] "office worker, trouble sleeping because the knee aches" — wanted ['kl0_kl1_normal_doubtful.md::Recommendations', 'general_exercise_and_lifestyle.md::Sleep and stress']; got ['kl0_kl1_normal_doubtful.md::Recommendations', 'general_exercise_and_lifestyle.md::Exercise', 'general_exercise_and_lifestyle.md::Weight management', 'general_exercise_and_lifestyle.md::Self-management and education']
- KL4 [arthroscopy] "previous meniscus tear, now asks about keyhole surgery to clean the joint" — wanted ['kl4_severe.md::Recommendations', 'aaos_oak3_recommendations.md::Lavage/Debridement', 'aaos_oak3_recommendations.md::Partial Meniscectomy']; got ['kl4_severe.md::Recommendations', 'kl4_severe.md::What this system should NOT do', 'general_exercise_and_lifestyle.md::Exercise', 'general_exercise_and_lifestyle.md::Weight management']
- KL3 [stress] "feels sad and stressed about losing mobility" — wanted ['kl3_moderate.md::Recommendations', 'general_exercise_and_lifestyle.md::Sleep and stress']; got ['kl3_moderate.md::Recommendations', 'general_exercise_and_lifestyle.md::Exercise', 'general_exercise_and_lifestyle.md::Weight management', 'general_exercise_and_lifestyle.md::Self-management and education']
