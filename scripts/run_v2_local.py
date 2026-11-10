"""Local port of 04_revision.ipynb (deepfake generalization study), revised for review round 2.

Same procedure as the Colab notebook plus the round-2 changes: near-duplicate clusters joined by dHash (Hamming <= 8) OR
CLIP cosine similarity (>= 0.95) within the same generator / real source, cluster-level splits (seed 42),
SimpleCNN and ResNet-50 x N seeds, CLIP + logistic regression, mean +- std over seeds, 95% percentile bootstrap CIs
resampled by unit (real/generated pair on test_own, cluster on the CF test sets), threshold/calibration analysis on
test_own, per-generator rates, figures.

Usage:
    python run_v2_local.py --zip dataset.zip --root <extract dir> --res results_v3 --workers 2 --seeds 0 1 2 3 4

Device: CUDA if present, else Apple MPS, else CPU. On CUDA/MPS CLIP ViT-L/14 is used, on CPU ViT-B/16.
Resumable: predictions of every (model, seed) are saved to <res>/pred_<model>_seed<k>.npz and reused.
"""
import argparse, collections, contextlib, json, os, random, shutil, time, zipfile
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True); ap.add_argument("--root", required=True); ap.add_argument("--res", required=True)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args, _ = ap.parse_known_args()

CFG = dict(
    ROOT=args.root, RES=args.res, SEEDS=list(args.seeds), SPLIT_SEED=42, HASH_THR=8, COS_THR=0.95,
    IMG=224, BS=32, WORKERS=args.workers,
    EPOCHS_CNN=10, LR_CNN=1e-3, EPOCHS_RESNET=5, LR_RESNET=1e-4,
    CLIP_ID="openai/clip-vit-large-patch14", N_BOOT=1000, N_CAL=200, N_CAL_REPEATS=20,
)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if DEV == "cpu":
    CFG.update(CLIP_ID="openai/clip-vit-base-patch16"); torch.set_num_threads(os.cpu_count() or 2)
CLIP_DTYPE = torch.float16 if DEV == "cuda" else torch.float32
USE_AMP = DEV == "cuda"
print("device:", DEV, "| CLIP:", CFG["CLIP_ID"], "| torch", torch.__version__, flush=True)
for d in (CFG["RES"], f"{CFG['RES']}/figs", f"{CFG['RES']}/splits_v2"):
    os.makedirs(d, exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{(time.time()-T0)/60:6.1f} min]", *a, flush=True)


def clip_features(meta):
    """CLIP image embeddings for every image (cached in RES/clip_feats_all.npz, keyed by image id; independent of the splits)."""
    feat_path = f"{CFG['RES']}/clip_feats_all.npz"
    if os.path.exists(feat_path):
        z = np.load(feat_path, allow_pickle=True); FE = dict(zip(z["ids"], z["feats"]))
        if all(i in FE for i in meta.id): return FE
    from transformers import CLIPModel, CLIPImageProcessor
    clip = CLIPModel.from_pretrained(CFG["CLIP_ID"], torch_dtype=CLIP_DTYPE).to(DEV).eval()
    proc = CLIPImageProcessor.from_pretrained(CFG["CLIP_ID"])
    feats = []
    with torch.no_grad():
        for i in range(0, len(meta), CFG["BS"]):
            ims = [Image.open(p).convert("RGB").resize((256, 256)) for p in meta.path.iloc[i:i + CFG["BS"]]]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(DEV, CLIP_DTYPE)
            out = clip.get_image_features(pixel_values=px)
            f = out if torch.is_tensor(out) else (out.image_embeds if getattr(out, "image_embeds", None) is not None else out.pooler_output)
            feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
            if (i // CFG["BS"]) % 20 == 0: log(f"CLIP feats {i}/{len(meta)}")
    feats = np.concatenate(feats); np.savez(feat_path, ids=meta.id.values, feats=feats); del clip
    return dict(zip(meta.id.values, feats))


def prepare():
    """Sections 1-2 of the notebook: data, near-duplicate clusters (dHash OR CLIP cosine), cluster-level splits."""
    if not os.path.exists(f"{CFG['ROOT']}/metadata.csv"):
        with zipfile.ZipFile(args.zip) as z: z.extractall(CFG["ROOT"])
    meta = pd.read_csv(f"{CFG['ROOT']}/metadata.csv", dtype=str).fillna("")
    meta["y"] = (meta.label == "fake").astype(int)
    meta["path"] = CFG["ROOT"] + "/images/" + meta.filename
    meta.loc[(meta.origin == "cf") & (meta.y == 1), "fake_type"] = "generated"
    meta["held_out"] = pd.to_numeric(meta.held_out, errors="coerce").fillna(0).astype(int)
    meta["group"] = np.where(meta.y == 1, meta.generator, "real:" + meta.source)
    meta = meta.reset_index(drop=True)
    log("images:", len(meta))

    def dhash(p, s=8):
        im = Image.open(p).convert("L").resize((s + 1, s), Image.LANCZOS)
        a = np.asarray(im, dtype=np.int16)
        return (a[:, 1:] > a[:, :-1]).ravel().astype(np.uint8)
    bits = np.stack([dhash(p) for p in meta.path]).astype(np.int32)
    D = 64 - (bits @ bits.T + (1 - bits) @ (1 - bits).T); np.fill_diagonal(D, 99)
    parent = list(range(len(meta)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    cf_idx = np.where(meta.origin == "cf")[0]
    grp = meta.group.values; origin = meta.origin.values; old_split = meta.split.values
    near = np.argwhere(np.triu(D <= CFG["HASH_THR"]))
    for i, j in near:
        if origin[i] == "cf" and origin[j] == "cf" and grp[i] == grp[j]:
            parent[find(i)] = find(j)
    dhash_cluster = np.array([find(i) for i in range(len(meta))])
    # second criterion (review round 2): CLIP cosine similarity >= COS_THR within the same generator / real source
    FE = clip_features(meta); Xf = np.stack([FE[i] for i in meta.id]); Xf = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
    Scf = Xf[cf_idx] @ Xf[cf_idx].T; np.fill_diagonal(Scf, 0)
    clip_pairs = [(cf_idx[a], cf_idx[b]) for a, b in zip(*np.where(np.triu(Scf >= CFG["COS_THR"], 1))) if grp[cf_idx[a]] == grp[cf_idx[b]]]
    clip_pairs_new = sum(1 for i, j in clip_pairs if dhash_cluster[i] != dhash_cluster[j])
    old_cross_clip = int(sum(1 for i, j in clip_pairs if old_split[i] != old_split[j]))
    for i, j in clip_pairs: parent[find(i)] = find(j)
    meta["cluster"] = [find(i) for i in range(len(meta))]
    cs = meta[meta.origin == "cf"].groupby("cluster").size()
    old_cross = int(sum(1 for i, j in near if origin[i] == "cf" and origin[j] == "cf" and old_split[i] != old_split[j]))
    old_cross_same_gen = int(sum(1 for i, j in near if origin[i] == "cf" and origin[j] == "cf" and grp[i] == grp[j] and old_split[i] != old_split[j]))
    multi = {}
    for g, d in meta[meta.origin == "cf"].groupby("group"):
        s = d.groupby("cluster").size(); v = int(s[s > 1].sum())
        if v > 0: multi[g] = v
    report = dict(
        hash_threshold=CFG["HASH_THR"], clip_cosine_threshold=CFG["COS_THR"],
        clip_pairs_same_group=int(len(clip_pairs)), clip_pairs_not_in_same_dhash_cluster=int(clip_pairs_new), v1_cross_split_clip_pairs=old_cross_clip,
        cf_images=int(len(cf_idx)), cf_clusters=int(len(cs)),
        cf_images_in_multi_clusters=int(cs[cs > 1].sum()), largest_cluster=int(cs.max()),
        exact_duplicate_pairs=int(sum(1 for i, j in np.argwhere(np.triu(D == 0)) if origin[i] == "cf" and origin[j] == "cf")),
        near_duplicate_pairs_cf_any_group=int(sum(1 for i, j in near if origin[i] == "cf" and origin[j] == "cf")),
        near_duplicate_pairs_cf_same_group=int(sum(1 for i, j in near if origin[i] == "cf" and origin[j] == "cf" and grp[i] == grp[j])),
        near_duplicate_pairs_own=int(sum(1 for i, j in near if origin[i] == "own" and origin[j] == "own")),
        near_duplicate_pairs_cf_own=int(sum(1 for i, j in near if origin[i] != origin[j])),
        v1_cross_split_near_duplicate_pairs=old_cross, v1_cross_split_near_duplicate_pairs_same_group=old_cross_same_gen,
        multi_cluster_images_by_group=multi,
    )
    log(json.dumps(report, indent=1, ensure_ascii=False))

    # cluster-level splits: whole cluster -> one split, 70/15/15 per group as closely as the cluster sizes allow
    meta["split_v2"] = meta.split
    cf = meta[(meta.origin == "cf") & (meta.held_out == 0)]
    for g, d in cf.groupby("group"):
        clusters = d.groupby("cluster").size().sample(frac=1, random_state=CFG["SPLIT_SEED"])
        n = len(d); targets = {"train": 0.70 * n, "val": 0.15 * n, "test_seen": 0.15 * n}; filled = {k: 0 for k in targets}
        for c, size in clusters.items():
            k = max(targets, key=lambda k: (targets[k] - filled[k]) / targets[k])
            meta.loc[(meta.cluster == c) & (meta.group == g), "split_v2"] = k; filled[k] += size
    meta.loc[(meta.origin == "cf") & (meta.held_out == 1), "split_v2"] = "test_unseen"
    meta.loc[meta.origin == "own", "split_v2"] = "test_own"
    sv2 = meta.split_v2.values
    new_cross = int(sum(1 for i, j in near if origin[i] == "cf" and origin[j] == "cf" and sv2[i] != sv2[j]))
    new_cross_same = int(sum(1 for i, j in near if origin[i] == "cf" and origin[j] == "cf" and grp[i] == grp[j] and sv2[i] != sv2[j]))
    report["v2_cross_split_near_duplicate_pairs"] = new_cross
    report["v2_cross_split_near_duplicate_pairs_same_group"] = new_cross_same
    report["v2_cross_split_clip_pairs_same_group"] = int(sum(1 for i, j in clip_pairs if sv2[i] != sv2[j]))
    report["images_moved_between_splits"] = int((meta.split != meta.split_v2).sum())
    counts = meta.groupby(["split_v2", "label"]).size().unstack(fill_value=0)
    report["split_counts_v2"] = {s: {l: int(counts.loc[s, l]) for l in counts.columns} for s in counts.index}
    log("cross-split near-duplicate pairs (dHash): v1", old_cross, "-> v2", new_cross, "| CLIP pairs crossing: v1", old_cross_clip, "-> v2", report["v2_cross_split_clip_pairs_same_group"], "| moved:", report["images_moved_between_splits"])
    log("\n" + counts.to_string())
    json.dump(report, open(f"{CFG['RES']}/dedup_report.json", "w"), indent=1, ensure_ascii=False)
    meta.drop(columns=["path"]).to_csv(f"{CFG['RES']}/metadata_v2.csv", index=False)
    for s, d in meta.groupby("split_v2"):
        d.drop(columns=["path"]).to_csv(f"{CFG['RES']}/splits_v2/{s}.csv", index=False)

    train_df = meta[meta.split_v2 == "train"].reset_index(drop=True)
    val_df = meta[meta.split_v2 == "val"].reset_index(drop=True)
    EVAL = {
        "test_seen": meta[meta.split_v2 == "test_seen"].reset_index(drop=True),
        "test_unseen": pd.concat([meta[meta.split_v2 == "test_unseen"], meta[(meta.split_v2 == "test_seen") & (meta.y == 0)]]).reset_index(drop=True),
        "test_own": meta[meta.split_v2 == "test_own"].reset_index(drop=True),
    }
    log("train:", len(train_df), "val:", len(val_df), {k: len(v) for k, v in EVAL.items()})
    return dict(meta=meta, D=D, near=near, report=report, train_df=train_df, val_df=val_df, EVAL=EVAL, FEAT=FE)


# ------------------------------------------------------------------ shared functions
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = T.Compose([T.Resize(256), T.RandomCrop(CFG["IMG"]), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(CFG["IMG"]), T.ToTensor(), T.Normalize(MEAN, STD)])
class ImgDS(torch.utils.data.Dataset):
    def __init__(self, df, tf): self.p, self.y, self.tf = df.path.tolist(), df.y.values.astype(np.float32), tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.tf(Image.open(self.p[i]).convert("RGB")), self.y[i]
def loader(df, tf, shuffle=False, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(ImgDS(df, tf), batch_size=CFG["BS"], shuffle=shuffle, generator=g,
                                       num_workers=CFG["WORKERS"], pin_memory=(DEV == "cuda"), persistent_workers=CFG["WORKERS"] > 0)
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if DEV == "cuda": torch.cuda.manual_seed_all(s)
    if DEV == "mps": torch.mps.manual_seed(s)
def amp(): return torch.autocast("cuda") if USE_AMP else contextlib.nullcontext()

def metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p); pred = (p >= thr).astype(int)
    out = dict(n=len(y), acc=accuracy_score(y, pred), bacc=balanced_accuracy_score(y, pred) if len(set(y)) == 2 else np.nan)
    out["auc"] = roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
    out["ap"] = average_precision_score(y, p) if len(set(y)) == 2 else np.nan
    out["tpr"] = pred[y == 1].mean() if (y == 1).any() else np.nan
    out["tnr"] = 1 - pred[y == 0].mean() if (y == 0).any() else np.nan
    return out

@torch.no_grad()
def predict(model, df):
    model.eval(); out = []
    for x, _ in loader(df, eval_tf):
        with amp():
            out.append(torch.sigmoid(model(x.to(DEV)).float().squeeze(1)).cpu())
    return torch.cat(out).numpy()

def train(model, name, epochs, lr, seed, train_df=None, val_df=None, log_dir=None):
    train_df = train_df if train_df is not None else globals()["train_df"]; val_df = val_df if val_df is not None else globals()["val_df"]
    set_seed(seed); model.to(DEV)
    pos_w = torch.tensor([float((train_df.y == 0).sum()) / max(float((train_df.y == 1).sum()), 1.0)], dtype=torch.float32, device=DEV)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP) if USE_AMP else None
    tl = loader(train_df, train_tf, shuffle=True, seed=seed)
    best_auc, best_state, logrows = -1, None, []
    for ep in range(epochs):
        model.train(); tot = 0; t0 = time.time()
        for x, y in tl:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad(set_to_none=True)
            with amp():
                loss = crit(model(x).squeeze(1).float(), y)
            if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
            tot += loss.item() * len(y)
        sched.step()
        vm = metrics(val_df.y, predict(model, val_df))
        logrows.append(dict(seed=seed, epoch=ep + 1, train_loss=tot / len(train_df), val_auc=vm["auc"], val_acc=vm["acc"], sec=time.time() - t0))
        log(f"{name} seed{seed} ep{ep+1}/{epochs}: loss={tot/len(train_df):.4f} val_auc={vm['auc']:.4f} val_acc={vm['acc']:.4f} ({time.time()-t0:.0f}s)")
        if vm["auc"] > best_auc: best_auc, best_state = vm["auc"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pd.DataFrame(logrows).to_csv(f"{log_dir or CFG['RES']}/trainlog_{name}_seed{seed}.csv", index=False)
    return model

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2))
        self.f = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 1))
    def forward(self, x): return self.h(self.f(x))

def fit_clip_lr(FEAT, train_df, val_df, Cs=(0.01, 0.1, 1.0, 10.0), verbose=True):
    """Logistic regression on frozen CLIP features; C chosen on validation AUC. Returns (scaler, clf, bestC, val_auc_by_C)."""
    F = lambda df: np.stack([FEAT[i] for i in df.id])
    sc = StandardScaler().fit(F(train_df)); best = None; by_C = {}
    for C in Cs:
        lr_ = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(F(train_df)), train_df.y)
        auc = roc_auc_score(val_df.y, lr_.predict_proba(sc.transform(F(val_df)))[:, 1]); by_C[C] = auc
        if verbose: log(f"CLIP+LR C={C}: val_auc={auc:.4f}")
        if best is None or auc > best[0]: best = (auc, C, lr_)
    return sc, best[2], best[1], by_C

PRED = collections.defaultdict(dict)
def pred_file(name, seed): return f"{CFG['RES']}/pred_{name}_seed{seed}.npz"
def save_pred(name, seed): np.savez(pred_file(name, seed), **PRED[name][seed])
def load_pred(name, seed):
    p = pred_file(name, seed)
    if os.path.exists(p):
        z = np.load(p); PRED[name][seed] = {k: z[k] for k in z.files}; log(f"{name} seed{seed}: loaded from {p}"); return True
    return False
def run_and_store(model, name, seed):
    PRED[name][seed] = {"val": predict(model, val_df), **{k: predict(model, v) for k, v in EVAL.items()}}
    save_pred(name, seed)


if __name__ == "__main__":
    globals().update(prepare())
    # -------------------------------------------------------------- SimpleCNN and ResNet-50 x seeds
    for seed in CFG["SEEDS"]:
        if not load_pred("SimpleCNN", seed):
            set_seed(seed); m = train(SimpleCNN(), "simple_cnn", CFG["EPOCHS_CNN"], CFG["LR_CNN"], seed)
            run_and_store(m, "SimpleCNN", seed); del m
        if not load_pred("ResNet-50", seed):
            set_seed(seed); m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Linear(m.fc.in_features, 1)
            m = train(m, "resnet50", CFG["EPOCHS_RESNET"], CFG["LR_RESNET"], seed)
            run_and_store(m, "ResNet-50", seed); del m
    log("networks done:", {k: sorted(v) for k, v in PRED.items()})

    # -------------------------------------------------------------- CLIP + LR
    sc, clf, bestC, clip_val = fit_clip_lr(FEAT, train_df, val_df); log("CLIP+LR chosen C =", bestC)
    F = lambda df: np.stack([FEAT[i] for i in df.id])
    PRED["CLIP+LR"][0] = {"val": clf.predict_proba(sc.transform(F(val_df)))[:, 1], **{k: clf.predict_proba(sc.transform(F(v)))[:, 1] for k, v in EVAL.items()}}
    save_pred("CLIP+LR", 0)

    # -------------------------------------------------------------- metrics, mean +- std, bootstrap CI (unit resampling)
    MODELS = ["SimpleCNN", "ResNet-50", "CLIP+LR"]
    rows = []
    for m in MODELS:
        for s, d in PRED[m].items():
            for t, p in d.items():
                if t == "val": continue
                rows.append(dict(model=m, seed=s, test=t, **metrics(EVAL[t].y, p)))
                if t == "test_own":
                    for ft in ("t2i", "i2i"):
                        mk = (EVAL[t].fake_type == ft).values
                        rows.append(dict(model=m, seed=s, test=f"test_own/{ft}", n=int(mk.sum()), tpr=float((p[mk] >= 0.5).mean())))
    per_seed = pd.DataFrame(rows); per_seed.to_csv(f"{CFG['RES']}/metrics_per_seed.csv", index=False)

    rng = np.random.default_rng(0)
    KEYS = ("acc", "bacc", "auc", "ap", "tpr", "tnr")
    def boot_units(t):
        """Resampling units: on test_own every real photograph together with the generated image made from it (t2i via its
        caption, i2i via its pixels) is one unit (1,000 pairs); on the CF test sets a near-duplicate cluster is one unit."""
        df = EVAL[t]
        key = np.where(df.y.values == 1, df.original_id.values, df.id.values) if t == "test_own" else df.cluster.values
        _, inv = np.unique(key, return_inverse=True)
        return [np.where(inv == u)[0] for u in range(inv.max() + 1)]
    UNITS = {t: boot_units(t) for t in EVAL}
    log("bootstrap units:", {t: len(u) for t, u in UNITS.items()})
    def boot_ci(m, t):
        y = EVAL[t].y.values; P = [PRED[m][s][t] for s in PRED[m]]; vals = {k: [] for k in KEYS}; units = UNITS[t]
        for _ in range(CFG["N_BOOT"]):
            idx = np.concatenate([units[u] for u in rng.integers(0, len(units), len(units))])
            if len(set(y[idx])) < 2: continue
            ms = [metrics(y[idx], p[idx]) for p in P]
            for k in KEYS: vals[k].append(np.mean([mm[k] for mm in ms]))
        return {k: np.percentile(v, [2.5, 97.5]) for k, v in vals.items()}
    summ = []
    for m in MODELS:
        for t in EVAL:
            d = per_seed[(per_seed.model == m) & (per_seed.test == t)]
            row = dict(model=m, test=t, n_seeds=len(d), n=int(d.n.iloc[0])); ci = boot_ci(m, t)
            for k in KEYS:
                row[f"{k}_mean"] = d[k].mean(); row[f"{k}_std"] = d[k].std(ddof=0) if len(d) > 1 else 0.0; row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = ci[k]
            if t == "test_own":
                for ft in ("t2i", "i2i"):
                    dd = per_seed[(per_seed.model == m) & (per_seed.test == f"test_own/{ft}")]
                    row[f"tpr_{ft}_mean"] = dd.tpr.mean(); row[f"tpr_{ft}_std"] = dd.tpr.std(ddof=0) if len(dd) > 1 else 0.0
            summ.append(row)
    summary = pd.DataFrame(summ); summary.to_csv(f"{CFG['RES']}/metrics_summary.csv", index=False)
    pd.set_option("display.width", 250)
    show = summary.copy()
    for k in ("acc", "auc", "ap", "tpr", "tnr"):
        show[k] = show.apply(lambda r: f"{r[k+'_mean']:.3f}±{r[k+'_std']:.3f} [{r[k+'_ci_lo']:.3f},{r[k+'_ci_hi']:.3f}]", axis=1)
    log("\n" + show[["model", "test", "n_seeds", "acc", "auc", "ap", "tpr", "tnr"]].to_string(index=False))
    log("\nTPR on test_own by subtype:\n" + show[show.test == "test_own"][["model", "tpr_t2i_mean", "tpr_t2i_std", "tpr_i2i_mean", "tpr_i2i_std"]].round(3).to_string(index=False))

    # -------------------------------------------------------------- ranking vs threshold on test_own
    def best_thr(y, p):
        ts = np.unique(np.round(p, 3)); ba = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in ts]
        return float(ts[int(np.argmax(ba))])
    cal_rows = []
    yo = EVAL["test_own"].y.values; yv = val_df.y.values
    for m in MODELS:
        for s, d in PRED[m].items():
            p, pv = d["test_own"], d["val"]
            t_val = best_thr(yv, pv); t_orc = best_thr(yo, p)
            accs = []; rng2 = np.random.default_rng(s)
            for r in range(CFG["N_CAL_REPEATS"]):
                pos = rng2.choice(np.where(yo == 1)[0], CFG["N_CAL"] // 2, replace=False); neg = rng2.choice(np.where(yo == 0)[0], CFG["N_CAL"] // 2, replace=False)
                cal = np.concatenate([pos, neg]); rest = np.setdiff1d(np.arange(len(yo)), cal)
                t_cal = best_thr(yo[cal], p[cal]); accs.append(accuracy_score(yo[rest], (p[rest] >= t_cal).astype(int)))
            cal_rows.append(dict(model=m, seed=s, auc=roc_auc_score(yo, p),
                acc_thr05=accuracy_score(yo, (p >= 0.5).astype(int)), bacc_thr05=balanced_accuracy_score(yo, (p >= 0.5).astype(int)),
                tpr_thr05=float((p[yo == 1] >= 0.5).mean()), tnr_thr05=float((p[yo == 0] < 0.5).mean()),
                frac_fake_below05=float((p[yo == 1] < 0.5).mean()),
                thr_val=t_val, acc_thr_val=accuracy_score(yo, (p >= t_val).astype(int)),
                acc_thr_cal200_mean=np.mean(accs), acc_thr_cal200_std=np.std(accs),
                thr_oracle=t_orc, acc_thr_oracle=accuracy_score(yo, (p >= t_orc).astype(int)),
                median_p_real=float(np.median(p[yo == 0])), median_p_fake=float(np.median(p[yo == 1]))))
    cal = pd.DataFrame(cal_rows); cal.to_csv(f"{CFG['RES']}/calibration_test_own.csv", index=False)
    cal_summary = cal.drop(columns="seed").groupby("model").agg(["mean", "std"]); cal_summary.to_csv(f"{CFG['RES']}/calibration_summary.csv")
    log("\n" + cal.groupby("model")[["auc", "acc_thr05", "bacc_thr05", "tpr_thr05", "tnr_thr05", "frac_fake_below05", "acc_thr_val", "acc_thr_cal200_mean", "acc_thr_oracle", "thr_val", "thr_oracle", "median_p_real", "median_p_fake"]].mean().round(3).to_string())

    # -------------------------------------------------------------- per generator and figures
    rows = []
    for m in MODELS:
        for s, d in PRED[m].items():
            for t, p in d.items():
                if t == "val": continue
                df = EVAL[t].copy(); df["p"] = p
                for (y, g), gdf in df.groupby(["y", "group"]):
                    rows.append(dict(model=m, seed=s, test=t, kind="fake" if y == 1 else "real", generator=g.replace("real:", ""),
                                     held_out=int(gdf.held_out.max()), n=len(gdf), rate=float((gdf.p >= 0.5).mean()) if y == 1 else float((gdf.p < 0.5).mean()),
                                     auc_vs_reals=(roc_auc_score(np.r_[np.ones(len(gdf)), np.zeros((df.y == 0).sum())], np.r_[gdf.p.values, df.p[df.y == 0].values]) if y == 1 else np.nan)))
    G = pd.DataFrame(rows).drop_duplicates(["model", "seed", "generator", "kind", "test"])
    Gs = G.groupby(["model", "test", "kind", "generator", "held_out", "n"]).agg(mean=("rate", "mean"), std=("rate", "std"), auc_mean=("auc_vs_reals", "mean"), auc_std=("auc_vs_reals", "std")).reset_index()
    Gs["std"] = Gs["std"].fillna(0.0); Gs["auc_std"] = Gs["auc_std"].fillna(0.0)
    Gs.to_csv(f"{CFG['RES']}/per_generator.csv", index=False)
    log("\n" + Gs[Gs.kind == "fake"].pivot_table(index=["test", "held_out", "generator"], columns="model", values="mean").round(3).to_string())

    sets = ["test_seen", "test_unseen", "test_own"]; labels = ["seen\ngenerators", "unseen\ngenerators", "our set\n(COCO + SDXL-Turbo)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(axes, ["auc", "acc"], ["ROC-AUC", "Accuracy (thr = 0.5)"]):
        w = 0.8 / len(MODELS)
        for i, m in enumerate(MODELS):
            r = summary[summary.model == m].set_index("test").loc[sets]
            mean = r[f"{metric}_mean"].values; err = np.vstack([mean - r[f"{metric}_ci_lo"].values, r[f"{metric}_ci_hi"].values - mean])
            ax.bar(np.arange(3) + i * w, mean, w, yerr=err, capsize=3, label=m)
            for j, v in enumerate(mean): ax.text(j + i * w, r[f"{metric}_ci_hi"].values[j] + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(np.arange(3) + w); ax.set_xticklabels(labels); ax.set_ylim(0.4, 1.08); ax.set_title(title + " (95% bootstrap CI)"); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, loc="lower left"); plt.tight_layout(); plt.savefig(f"{CFG['RES']}/figs/fig_main_v2.png", dpi=200); plt.close()

    g = Gs[(Gs.kind == "fake") & (Gs.model == "CLIP+LR")].sort_values(["test", "mean"])
    plt.figure(figsize=(9, 4.5)); colors = ["#4C72B0" if t == "test_seen" else "#DD8452" if t == "test_unseen" else "#55A868" for t in g.test]
    plt.barh(g.generator + "  (" + g.test.str.replace("test_", "") + ")", g["mean"], color=colors)
    plt.xlabel("fraction of fakes detected (thr = 0.5)"); plt.title("CLIP+LR: detection rate per generator"); plt.xlim(0, 1.05)
    plt.tight_layout(); plt.savefig(f"{CFG['RES']}/figs/fig_per_generator_v2.png", dpi=200); plt.close()

    plt.figure(figsize=(6.5, 4)); ts = np.linspace(0.01, 0.99, 99)
    for m in MODELS:
        p = np.mean([PRED[m][s]["test_own"] for s in PRED[m]], axis=0)
        plt.plot(ts, [accuracy_score(yo, (p >= t).astype(int)) for t in ts], label=m)
    plt.axvline(0.5, color="gray", ls="--", lw=1); plt.xlabel("decision threshold"); plt.ylabel("accuracy on our set"); plt.ylim(0.4, 0.9)
    plt.title("Accuracy on test-own as a function of the threshold"); plt.legend(frameon=False); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{CFG['RES']}/figs/fig_calibration_v2.png", dpi=200); plt.close()

    # -------------------------------------------------------------- predictions, config
    P = []
    for m in MODELS:
        for s, d in PRED[m].items():
            for t, p in d.items():
                df = (val_df if t == "val" else EVAL[t])[["id", "filename", "label", "origin", "generator", "fake_type", "source", "cluster"]].copy()
                df["model"], df["seed"], df["test"], df["prob_fake"] = m, s, t, p; P.append(df)
    pd.concat(P).to_csv(f"{CFG['RES']}/predictions_all.csv", index=False)
    json.dump({**CFG, "bootstrap_units": {t: len(u) for t, u in UNITS.items()}, "device": DEV, "clip_C": bestC, "clip_val_auc_by_C": clip_val,
               "torch": torch.__version__, "cpu": os.uname().machine, "total_minutes": (time.time() - T0) / 60}, open(f"{CFG['RES']}/config.json", "w"), indent=1, default=str)
    log("DONE. results in", CFG["RES"])
