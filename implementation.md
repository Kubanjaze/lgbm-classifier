# Phase 38 — LightGBM HTS Hit Classifier

**Version:** 1.1 | **Tier:** Standard | **Date:** 2026-03-26

## Goal
Train a LightGBM binary classifier to rank compounds by probability of being active (pIC50 ≥ 7.0).
Evaluate with LOO-CV: ROC-AUC, PR-AUC, EF@10%.

CLI: `python main.py --input data/compounds.csv --threshold 7.0`

Outputs: lgbm_scores.csv, roc_pr_plot.png, ef_bar.png

## Model
- ECFP4: radius=2, nBits=2048, useChirality=True
- `LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15, random_state=42)`
- LOO-CV: predict_proba on held-out compound
- Metrics: ROC-AUC, PR-AUC (AP), EF@5%, EF@10%, EF@20%

## Outputs
- lgbm_scores.csv: compound_name, family, y_true, lgbm_score
- roc_pr_plot.png: ROC + PR curves (same format as Phase 36)
- ef_bar.png: EF@K bar chart

## Logic
- Load compounds.csv, compute ECFP4 fingerprints, define active = pIC50 >= threshold
- LOO-CV: for each compound, train LGBMClassifier on remaining 44, get predict_proba on held-out
- Compute ROC-AUC, PR-AUC (average_precision_score) from LOO probability scores
- Compute EF@5%, EF@10%, EF@20% from ranked LOO scores
- Plot ROC + PR curves (2-panel), EF@K bar chart
- Save lgbm_scores.csv with per-compound probability scores

## Key Concepts
- LightGBM `LGBMClassifier` (n_estimators=100, learning_rate=0.1, num_leaves=15)
- ECFP4 Morgan fingerprints (radius=2, nBits=2048, useChirality=True)
- LOO-CV for small-sample classification evaluation
- Enrichment Factor at multiple cutoffs (reusing Phase 35 formula)

## Verification Checklist
- [x] ROC-AUC = 0.44 (below random 0.5, confirming overfitting)
- [x] PR-AUC = 0.72 (marginally above 0.667 hit rate baseline)
- [x] EF@10% = 0.75x (below random enrichment of 1.0x)
- [x] lgbm_scores.csv, roc_pr_plot.png, ef_bar.png saved to output/
- [x] 30/45 actives at pIC50 >= 7.0

## Risks
- LightGBM with 2048 features and 44 training samples = extreme overfitting risk (confirmed by results)
- Gradient boosting is not suited for high-dimensional sparse features with very small n; RF is preferred

## Actual Results (v1.1)

| Metric | Value | vs Baseline |
|---|---|---|
| ROC-AUC | 0.4422 | < 0.500 random |
| PR-AUC | 0.7157 | > 0.667 hit rate |
| EF@5% | 0.00× | — |
| EF@10% | 0.75× | — |
| EF@20% | 1.17× | — |

**Key insight:** LightGBM with 2048-feature ECFP4 and 44 training samples = severe overfitting. ROC-AUC falls below random in LOO-CV. PR-AUC marginally above hit rate. Compare to RF (Phase 37) which handles high-dim sparse features better due to implicit feature selection. Small datasets favor RF over gradient boosting.
