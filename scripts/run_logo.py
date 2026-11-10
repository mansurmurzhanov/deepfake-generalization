"""Leave-one-generator-out (LOGO) over all 12 Community Forensics generators (review round 2, point 1).

Usage:
    python run_logo.py --zip dataset.zip --root <extract dir> --res results_v3 --workers 2 --seed 0

For every generator g (9 training + 3 held-out): train on the cluster-level training split of the other 11 generators
(the real images stay as they are), select the epoch / C on the validation split without g, and evaluate on all 125
images of g against the real images of test_seen. Reported per generator and detector: detection rate at threshold 0.5
and AUC versus the test_seen real images. Networks are trained once per fold (one seed); CLIP + logistic regression is
refitted per fold on cached features. Resumable: every fold's predictions are saved to <res>/logo/pred_<gen>_<model>.npz.
Output: logo_results.csv (per generator), logo_summary.csv (mean +- std across the 12 generator choices), figs/fig_logo.png.
"""
import os, sys, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torchvision import models
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

seed = 0
if "--seed" in sys.argv:
    k = sys.argv.index("--seed"); seed = int(sys.argv[k + 1]); del sys.argv[k:k + 2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_v2_local as base   # parses --zip/--root/--res/--workers itself


def main():
    CFG, log = base.CFG, base.log
    OUT = f"{CFG['RES']}/logo"; os.makedirs(OUT, exist_ok=True)

    S = base.prepare(); meta, FEAT = S["meta"], S["FEAT"]
    cf = meta[meta.origin == "cf"]
    gens = sorted(cf[cf.y == 1].generator.unique())
    reals_test = meta[(meta.split_v2 == "test_seen") & (meta.y == 0)].reset_index(drop=True)
    log("LOGO over", len(gens), "generators:", gens)

    def fold_sets(g):
        tr = cf[(cf.split_v2 == "train") & (cf.generator != g)].reset_index(drop=True)
        va = cf[(cf.split_v2 == "val") & (cf.generator != g)].reset_index(drop=True)
        te = pd.concat([cf[cf.generator == g], reals_test]).reset_index(drop=True)   # all 125 images of g + 225 real test images
        return tr, va, te

    rows = []
    for g in gens:
        tr, va, te = fold_sets(g)
        f = f"{OUT}/pred_{g}_seed{seed}.npz"
        if os.path.exists(f):
            z = np.load(f); P = {k: z[k] for k in z.files}; log(f"{g}: loaded")
        else:
            P = {}
            base.set_seed(seed); m = base.train(base.SimpleCNN(), f"logo_cnn_{g}", CFG["EPOCHS_CNN"], CFG["LR_CNN"], seed, tr, va, OUT)
            P["SimpleCNN"] = base.predict(m, te); del m
            base.set_seed(seed); m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Linear(m.fc.in_features, 1)
            m = base.train(m, f"logo_resnet_{g}", CFG["EPOCHS_RESNET"], CFG["LR_RESNET"], seed, tr, va, OUT)
            P["ResNet-50"] = base.predict(m, te); del m
            sc, clf, C, _ = base.fit_clip_lr(FEAT, tr, va, verbose=False)
            P["CLIP+LR"] = clf.predict_proba(sc.transform(np.stack([FEAT[i] for i in te.id])))[:, 1]
            P["clip_C"] = np.array([C]); np.savez(f, **P)
        y = te.y.values
        for mname in ("SimpleCNN", "ResNet-50", "CLIP+LR"):
            p = P[mname]
            rows.append(dict(generator=g, held_out_in_main_study=int(cf[cf.generator == g].held_out.max()), model=mname, n_fake=int((y == 1).sum()), n_real=int((y == 0).sum()),
                             tpr_heldout=float((p[y == 1] >= 0.5).mean()), auc_heldout=float(roc_auc_score(y, p)), acc_heldout=float(((p >= 0.5).astype(int) == y).mean()),
                             tnr_real=float((p[y == 0] < 0.5).mean())))
        log(f"{g}: " + ", ".join(f"{r['model']} tpr={r['tpr_heldout']:.2f} auc={r['auc_heldout']:.3f}" for r in rows[-3:]))

    R = pd.DataFrame(rows); R.to_csv(f"{CFG['RES']}/logo_results.csv", index=False)
    # compare with the detection rate of the same generator when it was seen in training (main study, per_generator.csv)
    try:
        PG = pd.read_csv(f"{CFG['RES']}/per_generator.csv", keep_default_na=False)
        seen = PG[(PG.kind == "fake") & (PG.test == "test_seen")][["model", "generator", "mean", "auc_mean"]].rename(columns={"mean": "tpr_seen", "auc_mean": "auc_seen"})
        R = R.merge(seen, on=["model", "generator"], how="left"); R.to_csv(f"{CFG['RES']}/logo_results.csv", index=False)
    except Exception as e:
        log("per_generator.csv not merged:", e)
    summ = R.groupby("model").agg(tpr_heldout_mean=("tpr_heldout", "mean"), tpr_heldout_std=("tpr_heldout", "std"), tpr_heldout_min=("tpr_heldout", "min"),
                                  auc_heldout_mean=("auc_heldout", "mean"), auc_heldout_std=("auc_heldout", "std"), auc_heldout_min=("auc_heldout", "min"),
                                  n_generators=("generator", "count")).reset_index()
    summ.to_csv(f"{CFG['RES']}/logo_summary.csv", index=False)
    log("\n" + R.pivot_table(index="generator", columns="model", values=["tpr_heldout", "auc_heldout"]).round(3).to_string())
    log("\n" + summ.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.2)); MODELS = ["SimpleCNN", "ResNet-50", "CLIP+LR"]; w = 0.27
    order = R[R.model == "CLIP+LR"].sort_values("auc_heldout").generator.tolist()
    for i, m in enumerate(MODELS):
        d = R[R.model == m].set_index("generator").loc[order]
        ax.bar(np.arange(len(order)) + i * w, d.auc_heldout, w, label=m)
    ax.set_xticks(np.arange(len(order)) + w); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8); ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUC when the generator is held out"); ax.set_title("Leave-one-generator-out: each generator held out in turn (one seed)"); ax.grid(axis="y", alpha=0.3); ax.legend(frameon=False, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{CFG['RES']}/figs/fig_logo.png", dpi=200); plt.close()
    json.dump(dict(seed=seed, generators=gens, minutes=(time.time() - base.T0) / 60), open(f"{OUT}/logo_config.json", "w"), indent=1)
    log("LOGO DONE")


if __name__ == "__main__":   # required: DataLoader workers are spawned and re-import this file
    main()
