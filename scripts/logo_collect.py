"""Collect leave-one-generator-out results without the images.

Usage: python logo_collect.py results_v3

Uses the saved fold predictions in <res>/logo/pred_<generator>_seed0.npz (written by run_logo.py) and, for folds whose
networks were not trained, fits only the CLIP + logistic-regression detector from the cached CLIP features
(<res>/clip_feats_all.npz) and the cluster-level splits in <res>/metadata_v2.csv. Network entries of such folds stay empty.
Writes logo_results.csv, logo_summary.csv and figs/fig_logo.png exactly as run_logo.py does.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1]; seed = 0
meta = pd.read_csv(f"{R}/metadata_v2.csv", dtype=str, keep_default_na=False)
meta["y"] = (meta.label == "fake").astype(int); meta["held_out"] = pd.to_numeric(meta.held_out, errors="coerce").fillna(0).astype(int)
z = np.load(f"{R}/clip_feats_all.npz", allow_pickle=True); FEAT = dict(zip(z["ids"], z["feats"]))
F = lambda df: np.stack([FEAT[i] for i in df.id])
cf = meta[meta.origin == "cf"]
gens = sorted(cf[cf.y == 1].generator.unique())
reals_test = meta[(meta.split_v2 == "test_seen") & (meta.y == 0)].reset_index(drop=True)

def clip_fold(tr, va, te):
    sc = StandardScaler().fit(F(tr)); best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(F(tr)), tr.y)
        auc = roc_auc_score(va.y, lr.predict_proba(sc.transform(F(va)))[:, 1])
        if best is None or auc > best[0]: best = (auc, C, lr)
    return best[2].predict_proba(sc.transform(F(te)))[:, 1]

rows = []
for g in gens:
    tr = cf[(cf.split_v2 == "train") & (cf.generator != g)].reset_index(drop=True)
    va = cf[(cf.split_v2 == "val") & (cf.generator != g)].reset_index(drop=True)
    te = pd.concat([cf[cf.generator == g], reals_test]).reset_index(drop=True); y = te.y.values
    f = f"{R}/logo/pred_{g}_seed{seed}.npz"; P = {}
    if os.path.exists(f):
        zz = np.load(f); P = {k: zz[k] for k in zz.files}
        assert len(P["CLIP+LR"]) == len(y), (g, len(P["CLIP+LR"]), len(y))
    else:
        P["CLIP+LR"] = clip_fold(tr, va, te)
    for m in ("SimpleCNN", "ResNet-50", "CLIP+LR"):
        base = dict(generator=g, held_out_in_main_study=int(cf[cf.generator == g].held_out.max()), model=m, n_fake=int((y == 1).sum()), n_real=int((y == 0).sum()))
        if m in P:
            p = P[m]; base.update(tpr_heldout=float((p[y == 1] >= 0.5).mean()), auc_heldout=float(roc_auc_score(y, p)), acc_heldout=float(((p >= 0.5).astype(int) == y).mean()), tnr_real=float((p[y == 0] < 0.5).mean()))
        else:
            base.update(tpr_heldout=np.nan, auc_heldout=np.nan, acc_heldout=np.nan, tnr_real=np.nan)
        rows.append(base)
L = pd.DataFrame(rows); L.to_csv(f"{R}/logo_results.csv", index=False)
S = L.dropna(subset=["auc_heldout"]).groupby("model").agg(tpr_heldout_mean=("tpr_heldout", "mean"), tpr_heldout_std=("tpr_heldout", "std"), tpr_heldout_min=("tpr_heldout", "min"),
        auc_heldout_mean=("auc_heldout", "mean"), auc_heldout_std=("auc_heldout", "std"), auc_heldout_min=("auc_heldout", "min"), n_generators=("generator", "count")).reset_index()
S.to_csv(f"{R}/logo_summary.csv", index=False)
print(L.pivot_table(index="generator", columns="model", values="auc_heldout", dropna=False).round(3).to_string()); print(); print(S.round(3).to_string(index=False))

MODELS = ["SimpleCNN", "ResNet-50", "CLIP+LR"]; w = 0.27
order = L[L.model == "CLIP+LR"].sort_values("auc_heldout").generator.tolist()
fig, ax = plt.subplots(figsize=(9, 4.2))
for i, m in enumerate(MODELS):
    d = L[L.model == m].set_index("generator").loc[order]; ax.bar(np.arange(len(order)) + i * w, d.auc_heldout.fillna(0), w, label=m)
ax.set_xticks(np.arange(len(order)) + w); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8); ax.set_ylim(0.4, 1.02)
ax.set_ylabel("AUC when the generator is held out"); ax.set_title("Leave-one-generator-out: each generator held out in turn (one seed)"); ax.grid(axis="y", alpha=0.3); ax.legend(frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(f"{R}/figs/fig_logo.png", dpi=200); plt.close()
