"""Second duplicate check for the revised paper (reviewer round 2, points 3 and 8).

Usage:
    python check_overlap.py results_v2/metadata_v2.csv <image dir> results_v2/clip_feats_all.npz results_v2/overlap_report.json results_v2/figs/fig_overlap_pairs.png

(a) Community Forensics real images (train / val / test_seen) versus the 1,000 COCO photographs of test_own:
    dHash Hamming distance <= 8 and <= 4, and CLIP ViT-L/14 cosine similarity >= 0.95 / 0.90.
(b) Within Community Forensics, same generator or real source: CLIP cosine >= 0.95 pairs and how many of them cross a split_v2 boundary
    (dHash clusters were built at Hamming <= 8; this is the stronger, semantic check).
(c) Sanity: cosine between each image-to-image sample and its source photograph (expected high) and between
    unrelated real pairs (expected low), to put the thresholds in context.
Writes a JSON report and a contact sheet of the closest CF-real / COCO pairs for visual inspection.
"""
import json, sys
import numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

meta_path, img_dir, feat_path, out_json, out_png = sys.argv[1:6]
meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
z = np.load(feat_path, allow_pickle=True); F = dict(zip(z["ids"], z["feats"]))
X = np.stack([F[i] for i in meta.id]); X = X / np.linalg.norm(X, axis=1, keepdims=True)

def dhash(p, s=8):
    im = Image.open(p).convert("L").resize((s + 1, s), Image.LANCZOS); a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel().astype(np.int32)
B = np.stack([dhash(f"{img_dir}/{f}") for f in meta.filename])

real_cf = np.where((meta.origin == "cf") & (meta.label == "real"))[0]
real_own = np.where((meta.origin == "own") & (meta.label == "real"))[0]
D = 64 - (B[real_cf] @ B[real_own].T + (1 - B[real_cf]) @ (1 - B[real_own]).T)   # hamming, cf-real x coco
S = X[real_cf] @ X[real_own].T                                                      # cosine, cf-real x coco
rep = dict(
    cf_real=int(len(real_cf)), coco_real=int(len(real_own)),
    cf_coco_pairs_hamming_le8=int((D <= 8).sum()), cf_coco_pairs_hamming_le4=int((D <= 4).sum()), cf_coco_pairs_hamming_0=int((D == 0).sum()),
    cf_coco_pairs_cosine_ge095=int((S >= 0.95).sum()), cf_coco_pairs_cosine_ge090=int((S >= 0.90).sum()),
    cf_coco_max_cosine=float(S.max()), cf_coco_min_hamming=int(D.min()),
    cf_coco_pairs_both_hamming_le8_and_cosine_ge090=int(((D <= 8) & (S >= 0.90)).sum()),
)
# which CF reals (by split) are involved at hamming <= 8
ii, jj = np.where(D <= 8)
rep["cf_coco_hamming_le8_by_cf_split"] = {k: int(v) for k, v in meta.iloc[real_cf[ii]].split_v2.value_counts().items()}
rep["cf_coco_hamming_le8_pairs"] = [dict(cf=meta.id.iloc[real_cf[i]], cf_split=meta.split_v2.iloc[real_cf[i]], coco=meta.id.iloc[real_own[j]], hamming=int(D[i, j]), cosine=round(float(S[i, j]), 3)) for i, j in zip(ii, jj)]

# (b) within CF, same group, cosine >= 0.95 across split boundaries
cf = np.where(meta.origin == "cf")[0]
grp = np.where(meta.label == "fake", meta.generator, "real:" + meta.source).astype(str)
Scf = X[cf] @ X[cf].T; np.fill_diagonal(Scf, 0)
iu = np.triu_indices(len(cf), 1)
same = grp[cf][iu[0]] == grp[cf][iu[1]]; cross = meta.split_v2.values[cf][iu[0]] != meta.split_v2.values[cf][iu[1]]
sim = Scf[iu]
for thr in (0.95, 0.97):
    m = same & (sim >= thr)
    rep[f"cf_same_group_pairs_cosine_ge{int(thr*100)}"] = int(m.sum())
    rep[f"cf_same_group_pairs_cosine_ge{int(thr*100)}_cross_split"] = int((m & cross).sum())
    by = pd.Series(grp[cf][iu[0]][m & cross]).value_counts()
    rep[f"cf_cross_split_cosine_ge{int(thr*100)}_by_group"] = {k: int(v) for k, v in by.items()}
# clusters at hamming<=8 already share a split; how many cosine>=0.95 same-group pairs are NOT in the same dHash cluster?
cl = meta.cluster.values[cf]
m95 = same & (sim >= 0.95)
rep["cf_same_group_cosine_ge95_pairs_in_different_dhash_clusters"] = int((m95 & (cl[iu[0]] != cl[iu[1]])).sum())

# (c) context: i2i vs source, random real pairs
i2i = meta[meta.fake_type == "i2i"]; idx = {i: k for k, i in enumerate(meta.id)}
cos_i2i = [float(X[idx[r.id]] @ X[idx[r.original_id]]) for r in i2i.itertuples()]
rng = np.random.default_rng(0); a = rng.choice(real_own, 2000); b = rng.choice(real_own, 2000); keep = a != b
cos_rand = (X[a[keep]] * X[b[keep]]).sum(1)
rep["context"] = dict(i2i_vs_source_cosine_median=float(np.median(cos_i2i)), i2i_vs_source_cosine_p10=float(np.percentile(cos_i2i, 10)),
                      random_coco_pairs_cosine_median=float(np.median(cos_rand)), random_coco_pairs_cosine_p99=float(np.percentile(cos_rand, 99)))
json.dump(rep, open(out_json, "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "cf_coco_hamming_le8_pairs"}, indent=1))

# contact sheet: the 8 closest CF-real / COCO pairs by cosine
order = np.dstack(np.unravel_index(np.argsort(-S, axis=None)[:8], S.shape))[0]
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for k, (i, j) in enumerate(order):
    axes[0, k].imshow(Image.open(f"{img_dir}/{meta.filename.iloc[real_cf[i]]}").convert("RGB")); axes[0, k].set_title(f"CF real ({meta.split_v2.iloc[real_cf[i]]})", fontsize=8); axes[0, k].axis("off")
    axes[1, k].imshow(Image.open(f"{img_dir}/{meta.filename.iloc[real_own[j]]}").convert("RGB")); axes[1, k].set_title(f"COCO  cos={S[i, j]:.2f} ham={D[i, j]}", fontsize=8); axes[1, k].axis("off")
plt.suptitle("Closest Community Forensics real / COCO pairs by CLIP cosine similarity", fontsize=10); plt.tight_layout(); plt.savefig(out_png, dpi=150); print("written:", out_json, out_png)
