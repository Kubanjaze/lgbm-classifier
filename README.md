# lgbm-classifier — Phase 38

LightGBM binary classifier for HTS hit ranking on ECFP4 fingerprints.
Evaluated with LOO-CV: ROC-AUC, PR-AUC, EF@K.

## Usage

```bash
PYTHONUTF8=1 python main.py --input data/compounds.csv --threshold 7.0
```

## Outputs

| File | Description |
|---|---|
| `output/lgbm_scores.csv` | LOO-CV probability scores per compound |
| `output/roc_pr_plot.png` | ROC + PR curves |
| `output/ef_bar.png` | EF@K bar chart |
