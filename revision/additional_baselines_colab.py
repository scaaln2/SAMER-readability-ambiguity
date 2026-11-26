# ============================================================
# Task 6 (revision): additional baselines for R3.4, R3.5, R2.3
# Run in Google Colab with a GPU (Runtime > Change runtime type > T4 GPU).
# Upload train.tsv, dev.tsv, test.tsv to /content/sample_data/ first.
#
# Produces (in /content/task6_outputs/):
#   A) AraBERT single-stage fine-tuning  -> test_full_predictions_arabert_single_stage.csv
#   B) Frozen AraBERT [CLS] + LR          -> test_full_predictions_arabert_frozen_lr.csv
#   C) SOAR with the fine-tuned encoder   -> test_full_predictions_soar_ft_{linear,svm,ordinal}.csv
#   (The CPU-only Char. SVD-256 + LR baseline is trained by char_svd_lr_baseline.py.)
#   + task6_summary.csv, task6_environment.json
# All settings mirror the original notebook (seed 42, max_len 256, CLS pooling).
# Expected runtime on a T4: about 60-90 minutes.
# ============================================================
import os, re, json, random, subprocess, sys, platform
from pathlib import Path
import numpy as np, pandas as pd

for pkg in ["transformers", "datasets", "accelerate"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

SEED = 42
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
MAX_LEN = 256
LABELS = [3, 4, 5]
DATA = Path("/content/sample_data")
OUT = Path("/content/task6_outputs"); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed()

# ---------- data (same normalisation and ids as the original notebook) ----------
def normalize_text(t):
    t = str(t)
    t = re.sub(r"[ؗ-ًؚ-ْ]", "", t)
    t = t.replace("ـ", "")
    t = re.sub(r"[إأآا]", "ا", t)
    t = t.replace("ى", "ي")
    return re.sub(r"\s+", " ", t).strip()

def load_split(split):
    w = pd.read_csv(DATA / f"{split}.tsv", sep="\t", dtype=str, keep_default_na=False,
                    na_filter=False, engine="python")
    w.columns = [c.strip() for c in w.columns]
    rows = []
    for lv in [3, 4, 5]:
        for i, t in enumerate(w[f"L{lv}"]):
            rows.append(dict(id=f"{split}_row_{i}_L{lv}", group_id=f"{split}_row_{i}",
                             label=lv, text=normalize_text(t)))
    d = pd.DataFrame(rows)
    d = d[d.text.str.len() > 0].reset_index(drop=True)
    d["ambiguous"] = (d.groupby("text").label.transform("nunique") > 1).astype(int)
    return d

train, dev, test = load_split("train"), load_split("dev"), load_split("test")
print({k: len(v) for k, v in [("train", train), ("dev", dev), ("test", test)]})

def metrics(y, p):
    y, p = np.asarray(y, int), np.asarray(p, int); d = np.abs(y - p)
    return dict(acc=accuracy_score(y, p),
                macro_f1=f1_score(y, p, labels=LABELS, average="macro", zero_division=0),
                aod=d.mean(), nm1=(d <= 1).mean(),
                qwk=cohen_kappa_score(y, p, labels=LABELS, weights="quadratic"))

summary = []
def save_preds(name, slug, pred):
    df = test[["id", "group_id", "label", "ambiguous", "text"]].rename(columns={"label": "gold"}).copy()
    df["pred"] = np.asarray(pred, int)
    df.to_csv(OUT / f"test_full_predictions_{slug}.csv", index=False)
    row = dict(model=name, **metrics(df.gold, df.pred))
    for k, sub in [("same0", df[df.ambiguous == 0]), ("same1", df[df.ambiguous == 1])]:
        m = metrics(sub.gold, sub.pred); row.update({f"{k}_{a}": b for a, b in m.items()})
    summary.append(row)
    print(name, {k: round(v, 4) for k, v in row.items() if k != "model"})

# ============================================================
# A) AraBERT single-stage fine-tuning (same hyper-parameters as Stage 1)
# ============================================================
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
l2i = {3: 0, 4: 1, 5: 2}; i2l = {v: k for k, v in l2i.items()}
def to_ds(df):
    ds = Dataset.from_pandas(pd.DataFrame({"text": df.text, "labels": df.label.map(l2i)}))
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True,
                  remove_columns=["text"])
tr_ds, dv_ds, te_ds = to_ds(train), to_ds(dev), to_ds(test)

set_seed()
clf = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
import inspect, math
n_steps = math.ceil(len(tr_ds) / 16) * 3
args = dict(output_dir=str(OUT / "run_single"), learning_rate=2e-5, num_train_epochs=3,
            weight_decay=0.01, max_grad_norm=1.0,
            per_device_train_batch_size=16, per_device_eval_batch_size=32,
            logging_steps=100, report_to="none", seed=SEED, fp16=torch.cuda.is_available(),
            save_strategy="no", load_best_model_at_end=False)
sig = set(inspect.signature(TrainingArguments.__init__).parameters)
# warm-up = 10% of training steps (warmup_ratio in older versions, warmup_steps in newer ones)
if "warmup_ratio" in sig: args["warmup_ratio"] = 0.1
else: args["warmup_steps"] = int(round(0.1 * n_steps))
if "eval_strategy" in sig: args["eval_strategy"] = "epoch"
elif "evaluation_strategy" in sig: args["evaluation_strategy"] = "epoch"
args = {k: v for k, v in args.items() if k in sig}
print("TrainingArguments used:", args)
targs = TrainingArguments(**args)
trainer = Trainer(model=clf, args=targs, train_dataset=tr_ds, eval_dataset=dv_ds,
                  data_collator=DataCollatorWithPadding(tok))
trainer.train()
logits = trainer.predict(te_ds).predictions
save_preds("AraBERT (Single-Stage)", "arabert_single_stage", [i2l[i] for i in logits.argmax(-1)])
FT_DIR = OUT / "arabert_single_stage_model"
trainer.model.save_pretrained(FT_DIR); tok.save_pretrained(FT_DIR)
del trainer, clf; torch.cuda.empty_cache()

# ============================================================
# Embedding helper ([CLS] of last hidden layer, as in SOAR)
# ============================================================
@torch.no_grad()
def embed(model_dir, texts, bs=32):
    enc_model = AutoModel.from_pretrained(model_dir).to(DEVICE).eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt").to(DEVICE)
        out.append(enc_model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
    del enc_model; torch.cuda.empty_cache()
    return np.vstack(out)

# ============================================================
# B) Frozen pretrained AraBERT [CLS] + Logistic Regression (contextual branch alone)
# ============================================================
E_tr, E_dv, E_te = (embed(MODEL_NAME, d.text.tolist()) for d in (train, dev, test))
sc_b = StandardScaler().fit(E_tr)
lr_b = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED)
lr_b.fit(sc_b.transform(E_tr), train.label)
save_preds("Frozen AraBERT + LR", "arabert_frozen_lr", lr_b.predict(sc_b.transform(E_te)))

# ============================================================
# C) SOAR with the fine-tuned (single-stage) encoder
# ============================================================
F_tr, F_dv, F_te = (embed(str(FT_DIR), d.text.tolist()) for d in (train, dev, test))
vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=50000,
                      sublinear_tf=True, norm="l2")
C_tr = vec.fit_transform(train.text); C_dv = vec.transform(dev.text); C_te = vec.transform(test.text)
svd = TruncatedSVD(n_components=min(256, C_tr.shape[0] - 1, C_tr.shape[1] - 1), random_state=SEED)
S_tr = svd.fit_transform(C_tr); S_dv = svd.transform(C_dv); S_te = svd.transform(C_te)
sc_c, sc_f = StandardScaler().fit(S_tr), StandardScaler().fit(F_tr)
X_tr = np.hstack([sc_c.transform(S_tr), sc_f.transform(F_tr)]).astype("float32")
X_dv = np.hstack([sc_c.transform(S_dv), sc_f.transform(F_dv)]).astype("float32")
X_te = np.hstack([sc_c.transform(S_te), sc_f.transform(F_te)]).astype("float32")
y_tr, y_dv = train.label.values, dev.label.values

m = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=SEED).fit(X_tr, y_tr)
save_preds("SOAR-Linear (fine-tuned encoder)", "soar_ft_linear", m.predict(X_te))
m = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=5000).fit(X_tr, y_tr)
save_preds("SOAR-SVM (fine-tuned encoder)", "soar_ft_svm", m.predict(X_te))

class CumulativeOrdinal:
    def __init__(self, C): self.C = C
    def fit(self, X, y):
        self.m = [LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced",
                                     solver="lbfgs", random_state=SEED).fit(X, (y > t).astype(int)) for t in [3, 4]]
        return self
    def predict(self, X):
        p = np.vstack([mm.predict_proba(X)[:, 1] for mm in self.m]).T
        p[:, 1] = np.minimum(p[:, 1], p[:, 0])
        return 3 + (p >= 0.5).sum(1)
best = None
for C in [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]:          # same grid and criterion as the original SOAR-Ordinal
    mm = CumulativeOrdinal(C).fit(X_tr, y_tr); pd_ = mm.predict(X_dv)
    score = (f1_score(y_dv, pd_, labels=LABELS, average="macro"), -np.abs(y_dv - pd_).mean())
    if best is None or score > best[0]: best = (score, C, mm)
print("SOAR-Ordinal (fine-tuned) best C on dev:", best[1])
save_preds(f"SOAR-Ordinal (fine-tuned encoder, C={best[1]})", "soar_ft_ordinal", best[2].predict(X_te))

# ============================================================
pd.DataFrame(summary).to_csv(OUT / "task6_summary.csv", index=False)
env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
       "transformers": __import__("transformers").__version__, "sklearn": __import__("sklearn").__version__,
       "numpy": np.__version__, "pandas": pd.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}
json.dump(env, open(OUT / "task6_environment.json", "w"), indent=1)
print(pd.DataFrame(summary).round(4).to_string())

# zip the results (without model weights) for download
import shutil
keep = OUT / "to_send"; keep.mkdir(exist_ok=True)
for f in OUT.glob("*.csv"): shutil.copy(f, keep)
shutil.copy(OUT / "task6_environment.json", keep)
shutil.make_archive("/content/task6_results", "zip", keep)
print("Download /content/task6_results.zip")
try:
    from google.colab import files; files.download("/content/task6_results.zip")
except Exception:
    pass
