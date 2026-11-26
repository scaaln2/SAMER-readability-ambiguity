"""
Revision analyses for "Evaluating Arabic readability under cross-level ambiguity:
an ordinal analysis of the SAMER corpus" (electronics-4630086).

Reproduces every analysis added in the revision from the official SAMER files and the
saved test-set prediction files of the original experiments:

  1. Cross-partition overlap audit                      (Section 5.1, Table 3)
  2. Passage-group types and similarity-error analysis  (Sections 3.3, 6.3.2, Table 6)
  3. Sources of ambiguity and effect of normalisation   (Section 3.1)
  4. Group-level bootstrap and permutation tests        (Sections 5.6, 6.3.1, 6.5, Table 8, Tables A1-A2)
  5. Threshold sensitivity                              (Appendix B, Table A3)
  6. Sensitivity to texts shared across passage groups  (Sections 5.6.3, 6.5.1)

Usage:
    python revision_analyses.py --data_dir data --pred_dir predictions --out_dir results

data_dir must contain train.tsv, dev.tsv, test.tsv (official SAMER partitions).
pred_dir must contain test_full_predictions_<model>.csv (columns: id, group_id, gold, pred),
plus test_same0/1_predictions_arabert_two_stage.csv for the AraBERT subset results.
Requirements: numpy, pandas, scipy, scikit-learn.
"""
import argparse, re, itertools, json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, cohen_kappa_score

LABELS = [3, 4, 5]
MODELS = {"Character TF-IDF SVM": "svm_recheck", "Logistic Regression": "logistic_regression",
          "AraBERT (Two-Stage)": "arabert_two_stage", "Ordinal Logistic Regression": "ordinal_logreg",
          "SOAR-Linear": "soar_linear", "SOAR-SVM": "soar_svm", "SOAR-Ordinal": "soar_ordinal"}


def normalize_text(t):  # identical to the original pipeline
    t = str(t)
    t = re.sub(r"[ؗ-ًؚ-ْ]", "", t)
    t = t.replace("ـ", "")
    t = re.sub(r"[إأآا]", "ا", t)
    t = t.replace("ى", "ي")
    return re.sub(r"\s+", " ", t).strip()


def load(data_dir):
    rows, wide = [], {}
    for s in ["train", "dev", "test"]:
        w = pd.read_csv(Path(data_dir) / f"{s}.tsv", sep="\t", dtype=str, keep_default_na=False,
                        na_filter=False, engine="python")
        w.columns = [c.strip() for c in w.columns]
        wide[s] = w
        for i, r in w.iterrows():
            for lv in LABELS:
                rows.append(dict(split=s, group=f"{s}_row_{i}", id=f"{s}_row_{i}_L{lv}", level=lv,
                                 raw=r[f"L{lv}"], text=normalize_text(r[f"L{lv}"]),
                                 novel=r["Novel"], chapter=r["Chapter"]))
    L = pd.DataFrame(rows)
    return L[L.text.str.len() > 0].reset_index(drop=True), wide


# ------------------------------------------------------------------ 1
def overlap_audit(L, wide):
    out = {}
    ch = {s: set(wide[s].Novel + "|" + wide[s].Chapter) for s in wide}
    out["shared_chapters"] = {f"{a}-{b}": len(ch[a] & ch[b]) for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]}
    out["novels_per_split"] = {s: wide[s].Novel.nunique() for s in wide}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        A, B = L[L.split == a], L[L.split == b]
        forms = set(A.text)
        out[f"exact_{a}-{b}"] = dict(shared_forms=len(set(B.text) & forms), affected=int(B.text.isin(forms).sum()),
                                     pct=round(100 * B.text.isin(forms).mean(), 2))
        a_u = sorted(forms); uq = sorted(set(B.text) - forms)
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True).fit(a_u)
        nn = NearestNeighbors(n_neighbors=50, metric="cosine").fit(vec.transform(a_u))
        _, idx = nn.kneighbors(vec.transform(uq))
        best = {t: max(SequenceMatcher(None, t, a_u[j], autojunk=False).ratio() for j in r) for t, r in zip(uq, idx)}
        s = B.text.map(lambda t: best.get(t, 0.0))
        out[f"near_{a}-{b}"] = {f">={th}": round(100 * (s >= th).mean(), 2) for th in [0.90, 0.95]}
    tr = L[L.split == "train"]; te = L[L.split == "test"]
    lab = tr.groupby("text")["level"].agg(lambda x: set(x)); maj = tr.groupby("text")["level"].agg(lambda x: x.value_counts().idxmax())
    seen = te.text.isin(lab.index)
    out["affected_multi_label_in_train_pct"] = round(100 * np.mean([len(lab[t]) > 1 for t in te.text[seen]]), 2)
    out["majority_label_baseline_acc_pct"] = round(100 * np.mean([maj[t] == y for t, y in zip(te.text[seen], te["level"][seen])]), 2)
    w = te.groupby("text")["level"].transform("nunique") > 1
    c = L.groupby("text")["level"].transform("nunique") > 1
    out["test_ambiguous_within_vs_corpus"] = [round(100 * w.mean(), 2), round(100 * c[te.index].mean(), 2), int((w != c[te.index]).sum())]
    return out, seen


# ------------------------------------------------------------------ 2
def group_table(L):
    te = L[L.split == "test"]
    w = te.pivot(index="group", columns="level", values="text")
    sim = lambda a, b: SequenceMatcher(None, a, b).ratio()  # default settings, as in the original analysis
    w["s"] = [np.mean([sim(r[3], r[4]), sim(r[4], r[5]), sim(r[3], r[5])]) for _, r in w.iterrows()]
    w["n_id"] = (w[3] == w[4]).astype(int) + (w[4] == w[5]).astype(int) + (w[3] == w[5]).astype(int)
    w["type"] = np.select([w.n_id == 3, w.n_id >= 1, w.s >= 0.90], ["exact_triple", "partial_dup", "near_dup"], "low")
    return w[["s", "n_id", "type"]]


def load_preds(pred_dir, base_ids):
    P = {}
    for n, slug in MODELS.items():
        p = pd.read_csv(Path(pred_dir) / f"test_full_predictions_{slug}.csv").set_index("id").loc[base_ids]
        P[n] = p
    return P


def similarity_analysis(P, G):
    rows, rho = [], []
    for n, p in P.items():
        m = p.join(G, on="group_id"); m["e"] = m.gold != m.pred; m["d"] = (m.gold - m.pred).abs()
        for t, g in m.groupby("type"):
            rows.append(dict(model=n, type=t, n=len(g), error=round(100 * g.e.mean(), 2), aod=round(g.d.mean(), 3)))
        gg = m.groupby("group_id").agg(s=("s", "first"), e=("e", "mean"), d=("d", "mean"), t=("type", "first"), k=("n_id", "first"))
        r = dict(model=n, rho_all=spearmanr(gg.s, gg.e)[0])
        x = gg[gg.t != "exact_triple"]; r.update(rho_noexact_err=spearmanr(x.s, x.e)[0], rho_noexact_aod=spearmanr(x.s, x.d)[0])
        x = gg[gg.k == 0]; r.update(rho_noident_err=spearmanr(x.s, x.e)[0], p_noident_err=spearmanr(x.s, x.e)[1])
        rho.append(r)
    return pd.DataFrame(rows), pd.DataFrame(rho)


# ------------------------------------------------------------------ 3
def ambiguity_sources(L):
    out = {}
    for col in ["text", "raw"]:
        X = L.copy(); X["t"] = X[col] if col == "text" else X[col].map(lambda s: re.sub(r"\s+", " ", str(s)).strip())
        lab = X.groupby("t")["level"].agg(lambda s: set(s)); X["amb"] = X.t.map(lambda t: len(lab[t]) > 1)
        c = X.groupby(["t", "level"]).size().unstack(fill_value=0)
        out[col] = dict(unique=X.t.nunique(), ambiguous=int(X.amb.sum()), pct=round(100 * X.amb.mean(), 2),
                        ceiling=round(100 * c.max(axis=1).sum() / len(X), 2))
    X = L.copy()
    X["within"] = X.groupby(["group", "text"])["level"].transform("nunique") > 1
    tg = defaultdict(lambda: defaultdict(set))
    for t, g, lv in zip(X.text, X.group, X["level"]): tg[t][g].add(lv)
    X["cross"] = [any(g2 != g and (ls - {lv}) for g2, ls in tg[t].items()) for t, g, lv in zip(X.text, X.group, X["level"])]
    out["within_only"] = int((X.within & ~X.cross).sum()); out["both"] = int((X.within & X.cross).sum())
    out["cross_only"] = int((~X.within & X.cross).sum())
    xg = X[X.cross]  # instances involved in cross-group ambiguity (both + cross-only)
    out["cross_group_instances"] = int(len(xg))
    out["cross_group_instances_le3_tokens_pct"] = round(100 * (xg.text.str.split().str.len() <= 3).mean(), 1)
    c0 = X.groupby(["text", "level"]).size().unstack(fill_value=0)
    out["correct_classifiable_corpus"] = int(c0.max(axis=1).sum())
    c = X.groupby(["group", "text", "level"]).size().unstack(fill_value=0)
    out["correct_classifiable_within_group_only"] = int(c.max(axis=1).sum())
    out["ceiling_within_group_only"] = round(100 * c.max(axis=1).sum() / len(X), 2)
    return out


# ------------------------------------------------------------------ 4
def metric_vec(y, p):
    d = np.abs(y - p)
    return [np.mean(y == p), f1_score(y, p, labels=LABELS, average="macro", zero_division=0), d.mean(),
            np.mean(d <= 1), cohen_kappa_score(y, p, labels=LABELS, weights="quadratic")]


def group_statistics(P, sub_arabert, n_boot=1000, n_perm=10000):
    base = next(iter(P.values()))
    y = base.gold.values.astype(int); amb = base.same_all.values.astype(int)
    _, gidx = np.unique(base.group_id.values, return_inverse=True)
    mem = defaultdict(list)
    for i, g in enumerate(gidx): mem[g].append(i)
    MEM = np.array([mem[g] for g in range(len(mem))]); G = len(MEM)
    rng = np.random.default_rng(42); samples = [MEM[rng.integers(0, G, G)].ravel() for _ in range(n_boot)]
    names = ["Accuracy", "Macro-F1", "AOD", "NM@1", "QWK"]; boot = {}; ci = []
    pred = {n: p.pred.values.astype(int) for n, p in P.items()}
    for n in P:
        b = np.array([metric_vec(y[ix], pred[n][ix]) for ix in samples]); boot[n] = b
        pt = metric_vec(y, pred[n])
        for k, m in enumerate(names):
            ci.append(dict(model=n, metric=m, point=pt[k], lo=np.percentile(b[:, k], 2.5), hi=np.percentile(b[:, k], 97.5)))
    corr = {n: (pred[n] == y).astype(int) for n in P}
    signs = np.random.default_rng(2026).choice([-1, 1], size=(n_perm, G))
    pairs = []
    for a, b in itertools.combinations(P, 2):
        dg = (corr[a][MEM].sum(1) - corr[b][MEM].sum(1)).astype(float)
        p = (1 + (np.abs(signs @ dg) >= abs(dg.sum())).sum()) / (n_perm + 1)
        da = boot[a][:, 0] - boot[b][:, 0]; df = boot[a][:, 1] - boot[b][:, 1]
        pairs.append(dict(A=a, B=b, d_acc=corr[a].mean() - corr[b].mean(), acc_lo=np.percentile(da, 2.5), acc_hi=np.percentile(da, 97.5),
                          d_f1=metric_vec(y, pred[a])[1] - metric_vec(y, pred[b])[1], f1_lo=np.percentile(df, 2.5), f1_hi=np.percentile(df, 97.5), p=p))
    PR = pd.DataFrame(pairs); o = np.argsort(PR.p.values); adj = np.empty(len(o)); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (len(o) - k) * PR.p.values[i])); adj[i] = run
    PR["p_holm"] = adj
    gaps = []
    for n in P:
        pp = sub_arabert if n == "AraBERT (Two-Stage)" else pred[n]; c = pp == y
        bs = [c[ix][amb[ix] == 0].mean() - c[ix][amb[ix] == 1].mean() for ix in samples]
        gaps.append(dict(model=n, gap=c[amb == 0].mean() - c[amb == 1].mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    gs = corr["Character TF-IDF SVM"][MEM].sum(1); pbar = corr["Character TF-IDF SVM"].mean()
    icc = (gs.var() / (3 * pbar * (1 - pbar)) - 1) / 2
    return pd.DataFrame(ci), PR, pd.DataFrame(gaps), icc


# ------------------------------------------------------------------ 5
def threshold_sensitivity(P, G):
    rows = []
    for t in [0.80, 0.85, 0.90, 0.93, 0.95]:
        for n, p in P.items():
            m = p.join(G, on="group_id"); m = m[m.n_id == 0]; e = m.gold != m.pred
            rows.append(dict(threshold=t, model=n, n_low=m[m.s < t].group_id.nunique(), n_high=m[m.s >= t].group_id.nunique(),
                             err_low=round(100 * e[m.s < t].mean(), 1), err_high=round(100 * e[m.s >= t].mean(), 1)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 6
def shared_text_sensitivity(L, P, sub_arabert, n_boot=1000, n_perm=10000):
    """Repeat the group-level tests with connected components of passage groups
    linked by shared normalised texts as resampling units."""
    T = L[L.split == "test"]
    tg = T.groupby("text")["group"].agg(lambda s: set(s)); shared = tg[tg.map(len) > 1]
    par = {g: g for g in T.group.unique()}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for gs in shared.values:
        gl = sorted(gs)
        for g in gl[1:]: par[find(g)] = find(gl[0])
    comp = {g: find(g) for g in par}
    stats = dict(shared_texts=len(shared), shared_instances=int(T.text.isin(shared.index).sum()),
                 groups_touched=len(set().union(*shared.values)) if len(shared) else 0,
                 shared_texts_le3_tokens=int(sum(len(t.split()) <= 3 for t in shared.index)),
                 units=len(set(comp.values())), largest_unit=int(pd.Series(comp).value_counts().iloc[0]))
    base = next(iter(P.values())); y = base.gold.values.astype(int); amb = base.same_all.values.astype(int)
    _, u = np.unique(base.group_id.map(comp).values, return_inverse=True); K = u.max() + 1
    M = [np.where(u == k)[0] for k in range(K)]
    rng = np.random.default_rng(42); S = [np.concatenate([M[j] for j in rng.integers(0, K, K)]) for _ in range(n_boot)]
    signs = np.random.default_rng(2026).choice([-1, 1], size=(n_perm, K))
    C = {n: (p.pred.values.astype(int) == y).astype(int) for n, p in P.items()}
    rows = []
    for a, b in itertools.combinations(P, 2):
        d = C[a] - C[b]; dk = np.bincount(u, weights=d, minlength=K)
        p = (1 + (np.abs(signs @ dk) >= abs(dk.sum())).sum()) / (n_perm + 1)
        bs = [d[ix].mean() for ix in S]
        rows.append(dict(A=a, B=b, d_acc=d.mean(), acc_lo=np.percentile(bs, 2.5), acc_hi=np.percentile(bs, 97.5), p=p))
    PR = pd.DataFrame(rows); o = np.argsort(PR.p.values); adj = np.empty(len(o)); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (len(o) - k) * PR.p.values[i])); adj[i] = run
    PR["p_holm"] = adj
    gaps = []
    for n in P:
        pp = sub_arabert if n == "AraBERT (Two-Stage)" else P[n].pred.values.astype(int); c = pp == y
        bs = [c[ix][amb[ix] == 0].mean() - c[ix][amb[ix] == 1].mean() for ix in S]
        gaps.append(dict(model=n, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    return stats, PR, pd.DataFrame(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", default="results")
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(exist_ok=True, parents=True)
    L, wide = load(a.data_dir)
    audit, _ = overlap_audit(L, wide); json.dump(audit, open(out / "1_overlap_audit.json", "w"), indent=1)
    G = group_table(L); G.to_csv(out / "2_test_group_types.csv")
    base = pd.read_csv(Path(a.pred_dir) / "test_full_predictions_svm_recheck.csv").id
    P = load_preds(a.pred_dir, base)
    tab, rho = similarity_analysis(P, G); tab.to_csv(out / "2_error_by_group_type.csv", index=False); rho.to_csv(out / "2_spearman.csv", index=False)
    json.dump(ambiguity_sources(L), open(out / "3_ambiguity_sources.json", "w"), indent=1)
    sub = pd.concat([pd.read_csv(Path(a.pred_dir) / f"test_{s}_predictions_arabert_two_stage.csv") for s in ["same0", "same1"]]).set_index("id").loc[base].pred.values
    ci, pairs, gaps, icc = group_statistics(P, sub)
    ci.to_csv(out / "4_group_bootstrap_ci.csv", index=False); pairs.to_csv(out / "4_pairwise_tests.csv", index=False)
    gaps.to_csv(out / "4_ambiguity_gaps.csv", index=False); json.dump({"icc_svm": icc}, open(out / "4_icc.json", "w"))
    threshold_sensitivity(P, G).to_csv(out / "5_threshold_sensitivity.csv", index=False)
    st, pr, gp = shared_text_sensitivity(L, P, sub)
    json.dump(st, open(out / "6_shared_text_units.json", "w"), indent=1)
    pr.to_csv(out / "6_pairwise_tests_merged_units.csv", index=False); gp.to_csv(out / "6_ambiguity_gaps_merged_units.csv", index=False)
    print("Done. Results written to", out)


if __name__ == "__main__":
    main()
