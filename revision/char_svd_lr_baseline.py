"""
Char. SVD-256 + LR baseline (Section 5.3.5, Table 9) of
"Evaluating Arabic readability under cross-level ambiguity" (electronics-4630086).

Trains Logistic Regression on the standardised 256-dimensional TruncatedSVD reduction of the
character TF-IDF representation, i.e. the surface branch of SOAR on its own, and saves its
test-set predictions. CPU only (about 2 minutes).

Usage:
    python revision/char_svd_lr_baseline.py --data_dir data --out_dir predictions
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 42


def normalize_text(t):  # identical to the original pipeline
    t = str(t)
    t = re.sub(r"[ؗ-ًؚ-ْ]", "", t)
    t = t.replace("ـ", "")
    t = re.sub(r"[إأآا]", "ا", t)
    t = t.replace("ى", "ي")
    return re.sub(r"\s+", " ", t).strip()


def load_split(data_dir, split):
    w = pd.read_csv(Path(data_dir) / f"{split}.tsv", sep="\t", dtype=str, keep_default_na=False,
                    na_filter=False, engine="python")
    w.columns = [c.strip() for c in w.columns]
    rows = [dict(id=f"{split}_row_{i}_L{lv}", group_id=f"{split}_row_{i}", label=lv, text=normalize_text(t))
            for lv in [3, 4, 5] for i, t in enumerate(w[f"L{lv}"])]
    d = pd.DataFrame(rows)
    d = d[d.text.str.len() > 0].reset_index(drop=True)
    d["same_all"] = (d.groupby("text").label.transform("nunique") > 1).astype(int)
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", default="predictions")
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    train, test = load_split(a.data_dir, "train"), load_split(a.data_dir, "test")
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True, norm="l2")
    C_tr = vec.fit_transform(train.text); C_te = vec.transform(test.text)
    svd = TruncatedSVD(n_components=min(256, C_tr.shape[0] - 1, C_tr.shape[1] - 1), random_state=SEED)
    S_tr = svd.fit_transform(C_tr); S_te = svd.transform(C_te)
    sc = StandardScaler().fit(S_tr)
    lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED)
    lr.fit(sc.transform(S_tr), train.label)
    df = test[["id", "group_id", "label", "same_all"]].rename(columns={"label": "gold"})
    df["pred"] = lr.predict(sc.transform(S_te))
    df.to_csv(out / "test_full_predictions_char_svd_lr.csv", index=False)
    print("Char. SVD-256 + LR test accuracy:", round((df.gold == df.pred).mean(), 4))


if __name__ == "__main__":
    main()
