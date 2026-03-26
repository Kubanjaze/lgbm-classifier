import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from lightgbm import LGBMClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
RDLogger.DisableLog("rdApp.*")

FAMILY_COLORS = {"benz": "#4C72B0", "naph": "#DD8452", "ind": "#55A868",
                 "quin": "#C44E52", "pyr": "#8172B2", "bzim": "#937860", "other": "#808080"}

def load_compounds(path, threshold):
    df = pd.read_csv(path)
    records, n_bad = [], 0
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None: n_bad += 1; continue
        try:
            pic50 = float(row["pic50"])
        except (KeyError, ValueError):
            continue
        if np.isnan(pic50): continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        fam = str(row["compound_name"]).split("_")[0]
        records.append({"compound_name": str(row["compound_name"]),
                        "family": fam if fam in FAMILY_COLORS else "other",
                        "pic50": pic50,
                        "active": int(pic50 >= threshold),
                        "fp": list(fp)})
    print(f"  {len(records)} valid ({n_bad} skipped)")
    return pd.DataFrame(records)

def compute_ef(y_true, y_score, k_frac):
    y_true = np.array(y_true); y_score = np.array(y_score)
    n = len(y_true); k = max(1, int(np.round(n * k_frac)))
    total_hits = y_true.sum()
    if total_hits == 0: return {"ef": float("nan"), "k": k, "hits_topk": 0, "k_frac": k_frac}
    order = np.argsort(y_score)[::-1]
    hits_topk = y_true[order[:k]].sum()
    ef = (hits_topk / k) / (total_hits / n)
    return {"ef": round(float(ef), 4), "k": k, "hits_topk": int(hits_topk), "k_frac": k_frac}

def plot_roc_pr(y_true, y_score, hit_rate, output_path):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"LightGBM (AUC={roc_auc:.3f})")
    ax_roc.plot([0,1],[0,1],"k--",lw=1,label="Random (0.500)")
    ax_roc.set_xlabel("FPR",fontsize=11); ax_roc.set_ylabel("TPR",fontsize=11)
    ax_roc.set_title("ROC Curve",fontsize=13,fontweight="bold")
    ax_roc.legend(fontsize=9); ax_roc.spines["top"].set_visible(False); ax_roc.spines["right"].set_visible(False)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    ax_pr.plot(rec, prec, color="#DD8452", lw=2, label=f"LightGBM (AP={pr_auc:.3f})")
    ax_pr.axhline(hit_rate, color="k", linestyle="--", lw=1, label=f"Random ({hit_rate:.3f})")
    ax_pr.set_xlabel("Recall",fontsize=11); ax_pr.set_ylabel("Precision",fontsize=11)
    ax_pr.set_title("Precision-Recall Curve",fontsize=13,fontweight="bold")
    ax_pr.legend(fontsize=9); ax_pr.spines["top"].set_visible(False); ax_pr.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()
    return roc_auc, pr_auc

def plot_ef_bars(ef_results, output_path):
    labels = [f"Top {int(r['k_frac']*100)}%\n(K={r['k']})" for r in ef_results]
    efs = [r["ef"] for r in ef_results]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, efs, color="#4C72B0", edgecolor="white")
    ax.axhline(1.0, color="#808080", linestyle="--", lw=1.5)
    for bar, ef in zip(bars, efs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03, f"{ef:.2f}x",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("EF@K", fontsize=11)
    ax.set_title("LightGBM EF@K (LOO-CV)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading: {args.input}")
    df = load_compounds(args.input, args.threshold)
    X = np.array(df["fp"].tolist())
    y = df["active"].values
    n_active = y.sum()
    hit_rate = n_active / len(y)
    print(f"  Actives: {n_active}/{len(y)} ({100*hit_rate:.1f}%)")

    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15,
                          random_state=42, verbose=-1)
    loo = LeaveOneOut()
    scores = np.zeros(len(y))
    print("Running LOO-CV...")
    for train_idx, test_idx in loo.split(X):
        lgbm.fit(X[train_idx], y[train_idx])
        scores[test_idx] = lgbm.predict_proba(X[test_idx])[0, 1]

    out_df = pd.DataFrame({"compound_name": df["compound_name"], "family": df["family"],
                           "y_true": y, "lgbm_score": scores})
    out_df.to_csv(os.path.join(args.output_dir, "lgbm_scores.csv"), index=False)
    print(f"Saved: {args.output_dir}/lgbm_scores.csv")

    roc_auc, pr_auc = plot_roc_pr(y, scores, hit_rate,
                                   os.path.join(args.output_dir, "roc_pr_plot.png"))
    print(f"Saved: {args.output_dir}/roc_pr_plot.png")

    ef_results = [compute_ef(y, scores, f) for f in [0.05, 0.10, 0.20]]
    plot_ef_bars(ef_results, os.path.join(args.output_dir, "ef_bar.png"))
    print(f"Saved: {args.output_dir}/ef_bar.png")

    print(f"\n--- LightGBM LOO-CV Results (threshold={args.threshold}) ---")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    for r in ef_results:
        print(f"  EF@{int(r['k_frac']*100)}%:   {r['ef']:.2f}x  (K={r['k']}, hits={r['hits_topk']})")
    print("\nDone.")

if __name__ == "__main__":
    main()
