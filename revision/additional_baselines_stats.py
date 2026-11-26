"""
Statistics for the additional baselines (Section 6.5.3, Table 9) of
"Evaluating Arabic readability under cross-level ambiguity" (electronics-4630086).

Uses the saved test-set predictions of the additional baselines (produced by
additional_baselines_colab.py) and of the original models, and reports:
  - Accuracy, Macro-F1, AOD, NM@1 and QWK for every additional baseline (Table 9);
  - L4 recall and number of L4 predictions for the two AraBERT schedules;
  - group-level paired comparisons (1,000-resample cluster bootstrap, seed 42;
    10,000-permutation sign-flip test, seed 2026), Holm-adjusted over the five
    comparisons reported in Section 6.5.3.

Usage:
    python revision/additional_baselines_stats.py --pred_dir predictions --out_dir results
Requirements: numpy, pandas, scikit-learn.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, recall_score

LABELS = [3, 4, 5]
FILES = {
    "Character TF-IDF SVM": "svm_recheck", "Logistic Regression": "logistic_regression",
    "AraBERT (Two-Stage)": "arabert_two_stage", "SOAR-Linear": "soar_linear",
    "AraBERT (Single-Stage)": "arabert_single_stage", "Char. SVD-256 + LR": "char_svd_lr",
    "Frozen AraBERT + LR": "arabert_frozen_lr", "SOAR-Linear (FT)": "soar_ft_linear",
    "SOAR-SVM (FT)": "soar_ft_svm", "SOAR-Ordinal (FT)": "soar_ft_ordinal"}
COMPARISONS = [("AraBERT (Two-Stage)", "AraBERT (Single-Stage)"),
               ("Character TF-IDF SVM", "SOAR-Linear (FT)"), ("Character TF-IDF SVM", "SOAR-SVM (FT)"),
               ("Character TF-IDF SVM", "SOAR-Ordinal (FT)"), ("Logistic Regression", "SOAR-SVM (FT)")]


def metrics(y, p):
    d = np.abs(y - p)
    return dict(Accuracy=np.mean(y == p), MacroF1=f1_score(y, p, labels=LABELS, average="macro", zero_division=0),
                AOD=d.mean(), NM1=np.mean(d <= 1), QWK=cohen_kappa_score(y, p, labels=LABELS, weights="quadratic"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred_dir", required=True); ap.add_argument("--out_dir", default="results")
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(exist_ok=True, parents=True)
    base = pd.read_csv(Path(a.pred_dir) / "test_full_predictions_svm_recheck.csv").set_index("id")
    ids = base.index; y = base.gold.values.astype(int)
    pred = {}
    for n, slug in FILES.items():
        p = pd.read_csv(Path(a.pred_dir) / f"test_full_predictions_{slug}.csv").set_index("id").loc[ids]
        assert (p.gold.values == y).all(), n
        pred[n] = p.pred.values.astype(int)
    tab = pd.DataFrame([dict(model=n, **metrics(y, p)) for n, p in pred.items()]).round(4)
    tab.to_csv(out / "6_additional_baselines_table9.csv", index=False)
    l4 = {n: dict(l4_predictions=int((pred[n] == 4).sum()), l4_recall=round(recall_score(y, pred[n], labels=[4], average=None)[0], 4))
          for n in ["AraBERT (Single-Stage)", "AraBERT (Two-Stage)"]}
    json.dump(l4, open(out / "6_arabert_l4.json", "w"), indent=1)
    _, gidx = np.unique(base.group_id.values, return_inverse=True)
    MEM = np.array([np.where(gidx == g)[0] for g in range(gidx.max() + 1)]); G = len(MEM)
    rng = np.random.default_rng(42); samples = [MEM[rng.integers(0, G, G)].ravel() for _ in range(1000)]
    signs = np.random.default_rng(2026).choice([-1, 1], size=(10000, G))
    rows = []
    for A, B in COMPARISONS:
        ca, cb = (pred[A] == y).astype(int), (pred[B] == y).astype(int)
        dg = (ca[MEM].sum(1) - cb[MEM].sum(1)).astype(float)
        p = (1 + (np.abs(signs @ dg) >= abs(dg.sum())).sum()) / (len(signs) + 1)
        bd = np.array([[np.mean(ca[ix]) - np.mean(cb[ix]),
                        f1_score(y[ix], pred[A][ix], labels=LABELS, average="macro") - f1_score(y[ix], pred[B][ix], labels=LABELS, average="macro")]
                       for ix in samples])
        rows.append(dict(A=A, B=B, d_acc=ca.mean() - cb.mean(), acc_lo=np.percentile(bd[:, 0], 2.5), acc_hi=np.percentile(bd[:, 0], 97.5),
                         d_f1=metrics(y, pred[A])["MacroF1"] - metrics(y, pred[B])["MacroF1"],
                         f1_lo=np.percentile(bd[:, 1], 2.5), f1_hi=np.percentile(bd[:, 1], 97.5), p=p))
    PR = pd.DataFrame(rows); o = np.argsort(PR.p.values); adj = np.empty(len(o)); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (len(o) - k) * PR.p.values[i])); adj[i] = run
    PR["p_holm"] = adj; PR.round(4).to_csv(out / "6_additional_comparisons.csv", index=False)
    print(tab.to_string(index=False)); print(json.dumps(l4, indent=1)); print(PR.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
