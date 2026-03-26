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

## Actual Results (v1.1)

| Metric | Value | vs Baseline |
|---|---|---|
| ROC-AUC | 0.4422 | < 0.500 random |
| PR-AUC | 0.7157 | > 0.667 hit rate |
| EF@5% | 0.00× | — |
| EF@10% | 0.75× | — |
| EF@20% | 1.17× | — |

**Key insight:** LightGBM with 2048-feature ECFP4 and 44 training samples = severe overfitting. ROC-AUC falls below random in LOO-CV. PR-AUC marginally above hit rate. Compare to RF (Phase 37) which handles high-dim sparse features better due to implicit feature selection. Small datasets favor RF over gradient boosting.
