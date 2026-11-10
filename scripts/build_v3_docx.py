"""Build draft v3 of the ELCVIA paper from draft v2 and the results of the revised run.

Usage:
    python build_v3_docx.py --v2 Akhmetova_Murzhanov_ELCVIA_draft_v2_1.docx --res results_v2 \
        --out Akhmetova_Murzhanov_ELCVIA_draft_v3.docx [--github URL]

Everything that the reviewer asked for is written from the files in results_v2/ (no number is typed by hand):
metrics_summary.csv, metrics_per_seed.csv, calibration_test_own.csv, per_generator.csv, dedup_report.json,
config.json, trainlog_*.csv, paired_analysis.csv, figs/*.png.
"""
import argparse, copy, glob, json, re
import numpy as np, pandas as pd
import docx
from docx.shared import Emu
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table

ap = argparse.ArgumentParser()
ap.add_argument("--v2", required=True); ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--github", default="[GitHub URL — insert before submission]")
ap.add_argument("--hardware", default="an Apple M3 laptop GPU (PyTorch MPS backend)")
a = ap.parse_args()
R = a.res

# ----------------------------------------------------------------------------- results
S = pd.read_csv(f"{R}/metrics_summary.csv").set_index(["model", "test"])
PS = pd.read_csv(f"{R}/metrics_per_seed.csv")
CAL = pd.read_csv(f"{R}/calibration_test_own.csv")
CM = CAL.drop(columns="seed").groupby("model").mean(); CS = CAL.drop(columns="seed").groupby("model").std(ddof=0).fillna(0)
PG = pd.read_csv(f"{R}/per_generator.csv", keep_default_na=False, dtype={"generator": str})
DD = json.load(open(f"{R}/dedup_report.json"))
CFG = json.load(open(f"{R}/config.json"))
PA = pd.read_csv(f"{R}/paired_analysis.csv").set_index("model")
LOGS = {m: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/trainlog_{m}_seed*.csv"))]) for m in ("simple_cnn", "resnet50")}
import os as _os
OV = json.load(open(f"{R}/overlap_report.json")) if _os.path.exists(f"{R}/overlap_report.json") else None
LG = pd.read_csv(f"{R}/logo_results.csv") if _os.path.exists(f"{R}/logo_results.csv") else None
LGS = pd.read_csv(f"{R}/logo_summary.csv").set_index("model") if _os.path.exists(f"{R}/logo_summary.csv") else None
NS = len(CFG["SEEDS"]); NSW = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(NS, str(NS))
seed_list = ", ".join(str(x) for x in CFG["SEEDS"][:-1]) + f" and {CFG['SEEDS'][-1]}"
splits = {s: pd.read_csv(f"{R}/splits_v2/{s}.csv", dtype=str) for s in ("train", "val", "test_seen", "test_unseen", "test_own")}
MODELS = ["SimpleCNN", "ResNet-50", "CLIP+LR"]; NAMES = {"SimpleCNN": "SimpleCNN", "ResNet-50": "ResNet-50", "CLIP+LR": "CLIP + linear"}

def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def pct(x): return f"{round(100 * x):d}%"
def ms(m, t, k, nd=3):
    r = S.loc[(m, t)]; return f"{r[f'{k}_mean']:.{nd}f}" + (f" ± {r[f'{k}_std']:.{nd}f}" if m != "CLIP+LR" else "")
def ci(m, t, k, nd=3):
    r = S.loc[(m, t)]; return f"[{r[f'{k}_ci_lo']:.{nd}f}, {r[f'{k}_ci_hi']:.{nd}f}]"
def mean(m, t, k): return S.loc[(m, t)][f"{k}_mean"]
def rng(vals, nd=2):
    vals = sorted(vals); return f"{vals[0]:.{nd}f}–{vals[-1]:.{nd}f}" if abs(vals[0] - vals[-1]) >= 10 ** (-nd) / 2 else f"{vals[0]:.{nd}f}"
def rngpct(vals):
    vals = sorted(vals); return f"{round(100*vals[0])}–{round(100*vals[-1])}%" if round(100*vals[0]) != round(100*vals[-1]) else f"{round(100*vals[0])}%"
def cnt(n): return f"{int(n):,}"
def gen_rate(m, t, g, kind="fake"):
    d = PG[(PG.model == m) & (PG.test == t) & (PG.kind == kind) & (PG.generator == g)]
    assert len(d) == 1, (m, t, g, kind, len(d)); return float(d["mean"].iloc[0]), float(d["std"].iloc[0]), int(d["n"].iloc[0])
def real_rate(m, t):
    d = PG[(PG.model == m) & (PG.test == t) & (PG.kind == "real")]
    assert len(d) >= 1, (m, t); return 1 - float(d["mean"].iloc[0]), float(d["std"].iloc[0]), int(d["n"].iloc[0])  # false alarm rate
def calv(m, k): return float(CM.loc[m, k])
def cals(m, k): return float(CS.loc[m, k])
def cal_ms(m, k, nd=3): return f"{calv(m, k):.{nd}f}" + (f" ± {cals(m, k):.{nd}f}" if m != "CLIP+LR" else "")
def pa(m, k): return float(PA.loc[m, f"{k}_mean"])
def pa_ms(m, k, nd=3): return f"{pa(m, k):.{nd}f}" + (f" ± {PA.loc[m, f'{k}_std']:.{nd}f}" if m != "CLIP+LR" and PA.loc[m, f"{k}_std"] > 0 else "")

clip_variant = "ViT-L/14" if "large" in CFG["CLIP_ID"] else "ViT-B/16"
clip_dim = "768" if clip_variant == "ViT-L/14" else "512"
clip_C = CFG["clip_C"]; clip_val_auc = float(CFG["clip_val_auc_by_C"][str(clip_C)] if str(clip_C) in CFG["clip_val_auc_by_C"] else max(CFG["clip_val_auc_by_C"].values()))
n_real_seen = int((splits["test_seen"].label == "real").sum()); n_fake_seen = int((splits["test_seen"].label == "fake").sum())
n_train = len(splits["train"]); per_gen_seen = round(n_fake_seen / 9)
best_ep = {m: L.loc[L.groupby("seed").val_auc.idxmax()] for m, L in LOGS.items()}
ep_sec = {m: float(L.sec.mean()) for m, L in LOGS.items()}
run_min = sum(float(L.sec.sum()) for L in LOGS.values()) / 60 + 15  # all training epochs (the run was resumed once) + CLIP features and analysis

def best_ep_text(m, label):
    b = best_ep[m]; eps = sorted(set(b.epoch.astype(int))); au = (b.val_auc.min(), b.val_auc.max())
    ep = f"epoch {eps[0]}" if len(eps) == 1 else "epochs " + ", ".join(map(str, eps[:-1])) + f" and {eps[-1]}" if len(eps) > 2 else f"epochs {eps[0]} and {eps[1]}"
    return f"{ep} for {label} (validation AUC {au[0]:.3f}–{au[1]:.3f} over the {NSW} seeds)" if abs(au[0]-au[1]) >= 0.0005 else f"{ep} for {label} (validation AUC {au[0]:.3f})"

# CI overlap of CLIP vs ResNet-50 on test_unseen and test_own (AUC)
def overlap(t, k="auc"):
    c, r = S.loc[("CLIP+LR", t)], S.loc[("ResNet-50", t)]
    return not (c[f"{k}_ci_lo"] > r[f"{k}_ci_hi"] or r[f"{k}_ci_lo"] > c[f"{k}_ci_hi"])
ov_unseen, ov_own = overlap("test_unseen"), overlap("test_own")
ov_text = ("do not overlap on test-unseen or test-own" if not ov_unseen and not ov_own else
           "overlap on test-unseen but not on test-own" if ov_unseen and not ov_own else
           "overlap on test-own but not on test-unseen" if not ov_unseen and ov_own else "overlap on both test-unseen and test-own")
ov_verdict = "reliable for the AUC" if not ov_unseen and not ov_own else "suggestive rather than established"

own_acc = [mean(m, "test_own", "acc") for m in MODELS]; own_auc = [mean(m, "test_own", "auc") for m in MODELS]
tpr_t2i = {m: float(S.loc[(m, "test_own")]["tpr_t2i_mean"]) for m in MODELS}; tpr_i2i = {m: float(S.loc[(m, "test_own")]["tpr_i2i_mean"]) for m in MODELS}
fa_coco = {m: real_rate(m, "test_own")[0] for m in MODELS}; fa_cf = {m: real_rate(m, "test_seen")[0] for m in MODELS}
clip_thr_dir = "below" if calv("CLIP+LR", "thr_oracle") < 0.5 else "above"
gain_cal = {m: round(calv(m, "acc_thr_cal200_mean"), 3) - round(calv(m, "acc_thr05"), 3) for m in MODELS}
val_thr_help = {m: calv(m, "acc_thr_val") - calv(m, "acc_thr05") for m in MODELS}
def help_text(m):
    d = val_thr_help[m]
    return "does not help" if d < 0.01 else ("helps a little" if d < 0.05 else "helps")
# oracle vs test_unseen accuracy for CLIP
clip_gap = mean("CLIP+LR", "test_unseen", "acc") - calv("CLIP+LR", "acc_thr_oracle")
oracle_sentence = (" Even at the best possible threshold the CLIP-based detector stays well below its accuracy on the benchmark "
                   f"({f3(calv('CLIP+LR', 'acc_thr_oracle'))} against {f3(mean('CLIP+LR', 'test_unseen', 'acc'))} on test-unseen), so the drop is not only a calibration problem; "
                   "part of it is a real loss of separability on SDXL-Turbo images, above all on the image-to-image subset.") if clip_gap > 0.05 else ""
paired_close = all(abs(pa(m, "paired_rate") - 0.5) < 0.1 for m in ("SimpleCNN", "ResNet-50"))
GH = a.github
def lg(m, k): return float(LGS.loc[m, k]) if LGS is not None else float("nan")
def lg_gen(m, g):
    d = LG[(LG.model == m) & (LG.generator == gname_lg(g))]; return (float(d.auc_heldout.iloc[0]), float(d.tpr_heldout.iloc[0])) if len(d) else (float("nan"), float("nan"))
def gname_lg(g):
    if LG is None: return g
    cands = LG.generator.unique(); low = {c.lower().replace(" ", "").replace("-", "").replace("_", ""): c for c in cands}
    k = g.lower().replace(" ", "").replace("-", "").replace("_", "")
    return low.get(k, g)
def _fig_main_paper():   # Figure 2 for the paper: same data as figs/fig_main_v2.png, legend placed above the panels
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
    S = pd.read_csv(f"{R}/metrics_summary.csv")
    sets = ["test_seen", "test_unseen", "test_own"]; labels = ["seen\ngenerators", "unseen\ngenerators", "our set\n(COCO + SDXL-Turbo)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4)); w = 0.8 / len(MODELS)
    for ax, metric, title in zip(axes, ["auc", "acc"], ["ROC-AUC", "Accuracy (thr = 0.5)"]):
        for i, m in enumerate(MODELS):
            r = S[S.model == m].set_index("test").loc[sets]; mu = r[f"{metric}_mean"].values
            err = np.vstack([mu - r[f"{metric}_ci_lo"].values, r[f"{metric}_ci_hi"].values - mu])
            ax.bar(np.arange(3) + i * w, mu, w, yerr=err, capsize=3, label=NAMES[m])
            for j, v in enumerate(mu): ax.text(j + i * w, r[f"{metric}_ci_hi"].values[j] + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(np.arange(3) + w); ax.set_xticklabels(labels); ax.set_ylim(0.4, 1.08); ax.set_title(title + " (95% bootstrap CI)"); ax.grid(axis="y", alpha=0.3)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(f"{R}/figs/fig_main_paper.png", dpi=200); plt.close()
_fig_main_paper()
HAVE_LOGO = LG is not None and LGS is not None
if HAVE_LOGO:   # compact version of the LOGO figure for the paper (the full one stays in figs/fig_logo.png)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as _plt
    _order = LG[LG.model == "CLIP+LR"].sort_values("auc_heldout").generator.tolist(); _w = 0.27
    _fig, _ax = _plt.subplots(figsize=(10, 2.7))
    for _i, _m in enumerate(MODELS):
        _d = LG[LG.model == _m].set_index("generator").loc[_order]
        _ax.bar(np.arange(len(_order)) + _i * _w, _d.auc_heldout.fillna(0), _w, label=NAMES[_m])
    _ax.set_xticks(np.arange(len(_order)) + _w); _ax.set_xticklabels(_order, rotation=25, ha="right", fontsize=8); _ax.set_ylim(0.4, 1.02)
    _ax.set_ylabel("held-out AUC", fontsize=9); _ax.grid(axis="y", alpha=0.3); _h, _l = _ax.get_legend_handles_labels(); _fig.legend(_h, _l, frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    _plt.tight_layout(rect=(0, 0, 1, 0.93)); _plt.savefig(f"{R}/figs/fig_logo_paper.png", dpi=200); _plt.close()
if HAVE_LOGO:
    LGv = LG.dropna(subset=["auc_heldout"])
    PRETTY = {"StyleGANXL": "StyleGAN-XL", "StyleSANXL": "StyleSAN-XL", "glide": "GLIDE", "tamingTransformers": "Taming Transformers", "VQDiffusion": "VQ-Diffusion", "ProjectedGAN": "Projected GAN"}
    lg_hard = {m: [PRETTY.get(g, g) for g in LGv[LGv.model == m].sort_values("auc_heldout").generator.tolist()[:2]] for m in MODELS}
    n_net = int(LGv[LGv.model == "ResNet-50"].generator.nunique()); n_clip = int(LGv[LGv.model == "CLIP+LR"].generator.nunique())
    miss_net = [PRETTY.get(g, g) for g in sorted(set(LG.generator) - set(LGv[LGv.model == "ResNet-50"].generator))]
    net_scope = "all 12" if n_net == 12 else f"{n_net} of the 12"
    net_on = "On the same 12 folds" if n_net == 12 else f"On the {n_net} of the 12 folds run for the networks"
    net_tail = "" if n_net == 12 else f" on the {n_net} of the 12 folds run for it"
    upto = "" if n_net == 12 else "up to "
    lg_max_r = float(LGv[LGv.model == "ResNet-50"].auc_heldout.max())
    lg_min = {m: lg(m, "auc_heldout_min") for m in MODELS}
    lg_nlow = {m: int((LGv[LGv.model == m].auc_heldout < 0.9).sum()) for m in MODELS}
    lg_n = int(lg("CLIP+LR", "n_generators"))
    lg_ratio = {m: round(float(LGv[LGv.model == m].tpr_heldout.mean()), 2) for m in MODELS}

# ----------------------------------------------------------------------------- texts
TITLE = "Detection of Deepfakes and AI-Generated Images: A Review of Methods and a Small-Scale Empirical Study of Generalization to Unseen Generators"

ABSTRACT = ("Images produced by generative models are easy to make and hard to tell from real photographs, and detectors that are almost perfect on the generators they were trained on often fail on new ones. First, we review the main families of detection methods (artifact-based CNNs, frequency-based methods, detectors built on large pre-trained vision models and diffusion-specific methods) and the benchmarks used to evaluate them. "
    f"Second, we report a small-scale empirical study of generalization. We build a dataset of 5,000 images: 3,000 images from the Community Forensics dataset (1,500 real and 1,500 generated by 12 generators drawn at random from those available in the smaller files of the release) and 2,000 images that we collected ourselves (1,000 real photographs from COCO and 1,000 images generated or partially re-synthesized with SDXL-Turbo). We train three detectors that are often used as baselines: a small CNN trained from scratch, a fine-tuned ResNet-50 and a logistic regression on frozen CLIP {clip_variant} features. Each detector is tested at three distances from the training data: on new images from the nine training generators, on three held-out generators and on our own set, which uses a different source of real photographs and a modern diffusion generator. Near-duplicate images are kept on one side of every split boundary, networks are trained with {NSW} seeds, generators are also held out one at a time, and all metrics are reported with 95% bootstrap confidence intervals. "
    f"On seen generators the two pre-trained detectors reach an AUC of {rng([mean('ResNet-50','test_seen','auc'), mean('CLIP+LR','test_seen','auc')])}. On the held-out generators the CLIP-based detector keeps an AUC of {f2(mean('CLIP+LR','test_unseen','auc'))}, while ResNet-50 drops to {f2(mean('ResNet-50','test_unseen','auc'))}, and the drop is concentrated on the two non-GAN generators. On our own set, accuracy at the fixed threshold of 0.5 falls to {rng(own_acc)} for all three detectors, but the ranking information is not lost in the same way: the CLIP-based detector keeps an AUC of {f2(mean('CLIP+LR','test_own','auc'))}, and a threshold chosen on 200 images from the new domain recovers an accuracy of {f2(calv('CLIP+LR','acc_thr_cal200_mean'))}. Image-to-image samples remain the hardest case for every detector. We also discuss pitfalls that make detection results look better than they are (file format, resolution, near-duplicates across splits). Code, splits and all predictions are available at {GH}.")

INTRO2 = ("The second part is a small-scale empirical study. We build a dataset of 5,000 images that combines a subset of the Community Forensics dataset [6], a recent large-scale collection of images from thousands of generators, with 2,000 images that we collected ourselves. Our own part uses a different source of real photographs (COCO [7]) and a modern distilled diffusion generator (SDXL-Turbo [8]) that is not present in the Community Forensics subset we use. On this data we train three detectors that are often used as baselines and test each of them at three distances from the training data: on new images from the training generators, on generators that were held out from training, and on our own set. The difference between the three settings is a direct measure of how much each detector loses as the test data moves away from the training data. "
    f"The study is small: twelve generators from one benchmark, three of them held out in the main protocol and each of them held out in turn in a complementary analysis, one external generator, and about {cnt(round(n_train, -2))} training images. We therefore do not try to rank detector families in general. Our aim is to measure the three-level drop carefully, with repeated runs and confidence intervals, and to separate two things that are often mixed in such results: a detector that no longer orders real and generated images correctly, and a detector whose decision threshold is simply in the wrong place for new data. We also pay attention to three technical details that are easy to overlook and that can make a detector look better than it really is: the file format and the resolution of real and fake images, and near-duplicate images that fall on both sides of a split.")

SEC31 = ("Community Forensics is distributed as parquet files of 1–13 GB each, which is far beyond what we could process on a free Google Colab session. We therefore used the Community Forensics-Small release, which the authors recommend for experiments and which includes real images with redistributable licenses, and we read only files smaller than 1.5 GB. "
    "The selection was done in five steps, all with fixed seeds, in notebook 02 of the repository. (1) From the parquet files smaller than 1.5 GB we read the metadata of every image (label, generator name, prompt) without decoding the images. (2) We listed the generators that had at least 125 generated images in the files read; this list is the population from which we sampled. (3) From this list we drew 12 generators with Python's random.sample and seed 42. (4) From these 12 we drew 3 held-out generators with random.sample again; they never appear in training or validation. (5) Within each of the 12 generators we drew 125 images, and from the real part of the release 1,500 images, with DataFrame.sample(random_state=42). The list of candidate generators, the drawn identifiers and the seed are stored in the repository, so the subset can be rebuilt exactly. Table 1 lists the generators. Because small parquet files mostly belong to models from the Manual subset of Community Forensics (models the authors downloaded and ran by hand), the candidate list and therefore our subset is dominated by GANs; it also contains an autoregressive model (Taming Transformers), a text-conditioned diffusion model (GLIDE), a discrete diffusion model (VQ-Diffusion) and a latent flow-matching model (LFM). We return to this bias in Section 6.")

SEC32 = ("The generated images were produced with SDXL-Turbo [8], a distilled version of SDXL that generates a 512 × 512 image in four denoising steps without classifier-free guidance (guidance scale 0). We used the fp16 weights of stabilityai/sdxl-turbo through the diffusers library (version 0.27 or later, Google Colab, T4 GPU) with the default scheduler of the model repository (EulerAncestralDiscreteScheduler); every image received its own generator seed drawn from Python's random module seeded with 42, so the whole sequence of seeds is reproducible from notebook 01. We used two procedures. For 600 text-to-image samples, we took the first human caption of a random real COCO image, appended the words “, realistic photo”, and generated an image from this prompt with 4 steps. Because the prompts come from the captions of our real photographs, the two classes share the same topics, so the detector cannot separate them by content. For 400 image-to-image samples, we took one of the 400 real COCO photographs of our set whose caption had not been used for a text-to-image prompt, resized it to 512 × 512, and passed it through the SDXL-Turbo image-to-image pipeline with its own caption plus “, realistic photo” as the prompt, guidance scale 0 and strength 0.5. With 4 scheduled steps, a strength of 0.5 means that the input is encoded to the latent space, noised to the middle of the noise schedule, and only the last 2 of the 4 denoising steps are run from that noised latent. The result inherits the layout, the colours and the coarse texture of the original photograph, but every pixel is decoded by the VAE of the model, so the low-level signal is synthetic even though the scene is kept; we call these images partially re-synthesized (Figure 1, bottom row). This simulates “AI-enhanced” photographs, which are common on social media. Every image-to-image sample keeps the identifier of its source photograph (original_id), so our set is paired: 400 of the 1,000 real photographs have a partially re-synthesized version with the opposite label in the same test set, and the other 600 share their caption with a text-to-image sample. We keep this design on purpose, because it asks whether a detector can separate a photograph from its own edit; since the two classes are then not independent samples, Section 5 also reports results without the 400 source photographs and a paired comparison, and the bootstrap resamples whole pairs. Nothing from our set is used in training or validation.")

multi = DD["multi_cluster_images_by_group"]; multi_gens = [g for g in sorted(multi, key=lambda g: -multi[g]) if not g.startswith("real")]
SEC34_A = ("The Community Forensics part was divided into training, validation and test sets within each generator and within the real images (70% / 15% / 15%), so that all three sets have the same composition. The three held-out generators form a separate test set of unseen generators, and our own set is used only for testing.")
OVTXT = (f" We also checked that the external set is independent of the benchmark: between the {cnt(OV['cf_real'])} real images of Community Forensics and the 1,000 COCO photographs there is no pair at Hamming distance 4 or less and no pair with CLIP cosine similarity above 0.95 (maximum {OV['cf_coco_max_cosine']:.2f}; the closest pairs are different photographs of similar scenes), so no COCO photograph is in the training data." if OV else "")
SEC34_B = (f"A random split by image is not enough when a dataset contains near-duplicates. Several generators in Community Forensics-Small, in particular StyleGAN2 and StyleGAN3, produce many almost identical faces, and a detector that sees one of them in training is helped when a near-copy appears in the test set. We checked all 5,000 images with two criteria: a 64-bit difference hash (dHash) at Hamming distance {DD['hash_threshold']} or less, which finds near-copies, and the cosine similarity of CLIP {clip_variant} embeddings at {DD['clip_cosine_threshold']} or more, which also finds look-alikes that differ in small details. Within the same generator or real source the hash criterion gave {cnt(DD['near_duplicate_pairs_cf_same_group'])} pairs ({DD['exact_duplicate_pairs']} of them exact duplicates) and the CLIP criterion {cnt(DD['clip_pairs_same_group'])} pairs, {cnt(DD['clip_pairs_not_in_same_dhash_cluster'])} of which the hash had missed; together they involve {DD['cf_images_in_multi_clusters']} images, "
    + (f"{', '.join(f'{multi[g]} from {g}' for g in multi_gens[:3])}" + (" and a few real images" if any(g.startswith("real") for g in multi) else "") + ". ")
    + f"In our first splits, {DD['v1_cross_split_near_duplicate_pairs_same_group']} hash pairs and {DD['v1_cross_split_clip_pairs']} CLIP pairs crossed a split boundary. We therefore rebuilt the splits at the level of clusters: images of the same generator or real source joined by either criterion form a cluster ({cnt(DD['cf_clusters'])} clusters for {cnt(DD['cf_images'])} images; the largest has {DD['largest_cluster']} images), each cluster was assigned to a single split, and the 70 / 15 / 15 proportion was kept per generator as closely as the cluster sizes allow. After this step no pair of either kind crosses a boundary within a generator or real source, and {cnt(DD['images_moved_between_splits'])} images changed split. "
    + "Our own set is used only for testing, so none of its images or their sources is seen in training; the 400 real/edited pairs inside it are described in Section 3.2 and handled separately in Section 5." + OVTXT + " All results in Section 5 use the new splits; the dedup report and the split files are in the repository. Table 2 summarizes the splits.")
TAB2_CAP = f"Table 2: Data splits after cluster-level de-duplication. The unseen-generator test set contains only generated images; for metrics that need both classes it is combined with the {n_real_seen} real images of the seen-generator test set."
def sp(s, lab): return int((splits[s].label == lab).sum())
TAB2_ROWS = [["train", "CF, 9 generators", cnt(sp("train", "real")), cnt(sp("train", "fake")), cnt(len(splits["train"])), "training"],
             ["val", "CF, 9 generators", cnt(sp("val", "real")), cnt(sp("val", "fake")), cnt(len(splits["val"])), "model selection, threshold"],
             ["test-seen", "CF, 9 generators", cnt(sp("test_seen", "real")), cnt(sp("test_seen", "fake")), cnt(len(splits["test_seen"])), "seen generators"],
             ["test-unseen", "CF, 3 held-out generators", "–", cnt(sp("test_unseen", "fake")), cnt(sp("test_unseen", "fake")), "unseen generators"],
             ["test-own", "COCO + SDXL-Turbo", "1,000", "1,000 (600 t2i, 400 i2i)", "2,000", "new source and generator"]]

SEC41_A = "We compare three detectors that are often used as baselines in the literature and that differ in how much prior knowledge they bring. All three output a single score, the probability that the image is generated, and are trained on the same training split."
SEC41_CLIP = [("CLIP + linear. ", True), (f"The image encoder of CLIP {clip_variant} [23], kept frozen, followed by a logistic regression classifier on the standardized, L2-normalized {clip_dim}-dimensional image embeddings, in the spirit of the universal detector of Ojha et al. [4]. The regularization parameter C was chosen on the validation set among {{0.01, 0.1, 1, 10}}; C = {clip_C:g} gave the best validation AUC ({clip_val_auc:.3f}). This detector has no random component once the features are fixed, so it is fitted once.", False)]

SEC42 = (f"SimpleCNN was trained for 10 epochs with AdamW (learning rate 10⁻³, weight decay 10⁻⁴) and ResNet-50 for 5 epochs with AdamW (learning rate 10⁻⁴, weight decay 10⁻⁴), both with a cosine learning-rate schedule, batch size 32, binary cross-entropy loss with a positive-class weight equal to the real/fake ratio of the training set, and random 224 × 224 crops and horizontal flips as augmentation. Each network was trained {NSW} times with seeds {seed_list}, which control the initialization of the new layers and the order of the training data; for each run the epoch with the best validation AUC was kept: {best_ep_text('simple_cnn', 'SimpleCNN')}, {best_ep_text('resnet50', 'ResNet-50')}. CLIP features were extracted once for all images after the same resizing to 256 pixels. "
    f"The revised experiments were run on {a.hardware}; one epoch took about {round(ep_sec['simple_cnn'])} s for SimpleCNN and {round(ep_sec['resnet50'] / 60)} min for ResNet-50, and the whole protocol ({NSW} seeds, CLIP features, bootstrap) about {int(round(run_min, -1))} minutes. The code, the split files, the configuration actually used (config.json) and the predictions of every detector on every test image are available at {GH}.")

SEC43 = ("Each detector is evaluated on three test sets of increasing difficulty: (1) test-seen, new images from the nine training generators; (2) test-unseen, images from the three held-out generators, combined with the real images of test-seen so that both classes are present; (3) test-own, our COCO + SDXL-Turbo set, which differs from the training data in both the real source and the generator. "
    "We report accuracy at a fixed threshold of 0.5, the area under the ROC curve (AUC), which does not depend on a threshold, and average precision (AP). We also report the detection rate, that is the fraction of generated images classified as generated at threshold 0.5, separately for every generator and for the two kinds of SDXL-Turbo images, and the false alarm rate on real images. The threshold of 0.5 is never tuned on a test set; Section 4.4 describes a separate analysis in which the threshold is varied.")

SEC44 = [[("Repeated runs. ", True), (f"For SimpleCNN and ResNet-50 every metric is the mean over the {NSW} seeds, and we report the standard deviation over seeds. The CLIP-based detector is deterministic, so its standard deviation over seeds is zero and only the bootstrap interval below applies.", False)],
    [("Confidence intervals. ", True), (f"The three test sets are small ({cnt(len(splits['test_seen']))} to 2,000 images), so a difference of a few percentage points may be noise. For every metric we report a 95% percentile bootstrap confidence interval: we resample the test set with replacement 1,000 times, compute the metric for each seed on the resample, average over seeds and take the 2.5th and 97.5th percentiles. Because the test images are not all independent, we resample units rather than images: on our own set a unit is a real photograph together with the generated image made from it (its caption for text-to-image, its pixels for image-to-image), 1,000 units in all; on the Community Forensics test sets a unit is a near-duplicate cluster of Section 3.4. The intervals therefore reflect the size of the test set; the standard deviations reflect the training randomness.", False)],
    [("Ranking versus threshold. ", True), ("An accuracy of 0.6 at threshold 0.5 can mean that the detector no longer separates real from generated images, or that it separates them but its scores have shifted so that 0.5 is no longer the right cut; AUC measures only the first. To separate the two on our own set we report accuracy at four thresholds: (a) the fixed 0.5 used everywhere else; (b) the threshold that maximizes balanced accuracy on the Community Forensics validation split, which a practitioner could choose without any access to the new domain; (c) the threshold that maximizes balanced accuracy on 200 stratified images drawn from our own set (100 real, 100 generated), evaluated on the remaining 1,800 images, repeated 20 times; (d) the oracle threshold that maximizes balanced accuracy on the whole set, which is not achievable in practice and serves as an upper bound. We also report the median score of real and generated images on our own set.", False)],
    [("Leave-one-generator-out. ", True), (f"Three held-out generators are a small sample of generator choices. To see how much the unseen-generator result depends on which generators are held out, we repeat the experiment with every one of the 12 generators held out in turn: the detectors are trained on the cluster-level training split of the other 11 generators (the real images unchanged), the epoch and the regularization parameter are selected on the validation split without that generator, and the 125 images of the excluded generator are evaluated against the {n_real_seen} real images of test-seen. This gives {upto}12 held-out AUC values per detector instead of one aggregate over 3 generators. The networks are trained once per fold (seed 0), so this analysis measures the spread across generator choices, not across seeds." + ("" if (not HAVE_LOGO or n_net == 12) else f" The CLIP-based detector, which needs no network training, was evaluated on all 12 folds; for the two networks {n_net} of the 12 folds were completed, and the remaining {12 - n_net} ({', '.join(miss_net)}) are marked with a dash in Table 4."), False)],
    [("Paired evaluation. ", True), ("For test-own we also report the AUC with the 400 source photographs removed, the AUC of each kind of generated image against the photographs it was made from, and the paired detection rate: the fraction of the 400 pairs in which the edited image scores higher than its own source (0.5 means the detector cannot tell a photograph from its edit).", False)]]

SEC5_OPEN = (f"Table 3 gives the main results and Figure 2 shows them as bar charts. Table 4 gives the detection rate for every generator and the false alarm rate for every real-image source. For the two networks every number is the mean over {NSW} seeds, written as mean ± standard deviation; brackets give the 95% bootstrap confidence interval (Section 4.4). The last column of Table 4 gives the leave-one-generator-out AUC of each generator. Table 5 gives the threshold analysis on our own set (the accuracy-versus-threshold curve is in the repository).")
TAB3_CAP = f"Table 3: Main results. Accuracy at threshold 0.5, AUC and AP of the three detectors on the three test sets, mean ± std over {NSW} seeds (the CLIP-based detector is deterministic), with 95% bootstrap confidence intervals in brackets (resampled by real/generated pair on test-own and by near-duplicate cluster on the Community Forensics sets). test-unseen contains the 375 held-out generated images and the {n_real_seen} real images of test-seen."
TAB3_ROWS = [[NAMES[m] if t == "test_seen" else "", {"test_seen": "test-seen", "test_unseen": "test-unseen", "test_own": "test-own"}[t]] + [f"{ms(m, t, k)} {ci(m, t, k)}" for k in ("acc", "auc", "ap")] for m in MODELS for t in ("test_seen", "test_unseen", "test_own")]

TAB4_CAP = f"Table 4: Detection rate per generator (fraction of generated images classified as generated at threshold 0.5) and false alarm rate per real-image source (fraction of real images classified as generated), mean ± std over seeds. Seen generators have about {per_gen_seen} test images each, so their rates are coarse; held-out generators have 125 each. Last column: AUC of each generator (all 125 images against the test-seen real images) when that generator is held out of training, for SimpleCNN / ResNet-50 / CLIP + linear (Section 4.4, one seed)."
SEEN = ["BigGAN", "GigaGAN", "Projected GAN", "StyleGAN2", "StyleGAN3", "StyleGAN-XL", "StyleSAN-XL", "GLIDE", "Taming Transformers"]
HELD = ["Gansformer", "LFM", "VQ-Diffusion"]
def gname(g):  # match names in per_generator.csv loosely
    cands = PG.generator.unique(); low = {c.lower().replace(" ", "").replace("-", "").replace("_", ""): c for c in cands}
    k = g.lower().replace(" ", "").replace("-", "").replace("_", "")
    if k in low: return low[k]
    for kk, c in low.items():
        if k in kk or kk in k: return c
    raise KeyError(g)
def rate_cell(m, t, g, kind="fake", fa=False):
    if kind == "real":
        v, s, n = real_rate(m, t)
    else:
        v, s, n = gen_rate(m, t, gname(g))
    return f"{v:.2f}" + (f" ± {s:.2f}" if m != "CLIP+LR" else "")
def logo_cell(g): return " / ".join(("–" if lg_gen(m, g)[0] != lg_gen(m, g)[0] else f"{lg_gen(m, g)[0]:.2f}") for m in MODELS) if HAVE_LOGO else "–"
TAB4_ROWS = ([[g, "test-seen"] + [rate_cell(m, "test_seen", g) for m in MODELS] + [logo_cell(g)] for g in SEEN]
           + [[f"{g} (held out)", "test-unseen"] + [rate_cell(m, "test_unseen", g) for m in MODELS] + [logo_cell(g)] for g in HELD]
           + [["SDXL-Turbo t2i (600)", "test-own"] + [f"{tpr_t2i[m]:.2f}" + (f" ± {float(S.loc[(m,'test_own')]['tpr_t2i_std']):.2f}" if m != "CLIP+LR" else "") for m in MODELS] + ["–"]]
           + [["SDXL-Turbo i2i (400)", "test-own"] + [f"{tpr_i2i[m]:.2f}" + (f" ± {float(S.loc[(m,'test_own')]['tpr_i2i_std']):.2f}" if m != "CLIP+LR" else "") for m in MODELS] + ["–"]]
           + [[f"False alarms, CF real ({n_real_seen})", "test-seen"] + [rate_cell(m, "test_seen", None, "real") for m in MODELS] + ["–"]]
           + [["False alarms, COCO real (1,000)", "test-own"] + [rate_cell(m, "test_own", None, "real") for m in MODELS] + ["–"]])

TAB5_CAP = ("Table 5: Accuracy on our own set (2,000 images) at different decision thresholds, mean ± std over seeds: (a) fixed 0.5; (b) threshold chosen on the Community Forensics validation split; (c) threshold chosen on 200 images of our own set, evaluated on the other 1,800 (20 draws); (d) oracle threshold. Last two columns: median score of real and generated images.")
TAB5_HEAD = ["Detector", "AUC", "(a) acc @ 0.5", "(b) acc @ val thr.", "(c) acc @ 200-image thr.", "(d) acc @ oracle thr.", "median score, real", "median score, fake"]
TAB5_ROWS = [[NAMES[m], cal_ms(m, "auc"), cal_ms(m, "acc_thr05"), f"{cal_ms(m, 'acc_thr_val')}\nt = {calv(m, 'thr_val'):.2f}", f"{calv(m, 'acc_thr_cal200_mean'):.3f} ± {calv(m, 'acc_thr_cal200_std'):.3f}", f"{cal_ms(m, 'acc_thr_oracle')}\nt = {calv(m, 'thr_oracle'):.2f}", f"{calv(m, 'median_p_real'):.2f}", f"{calv(m, 'median_p_fake'):.2f}"] for m in MODELS]

def g3(m, t, g): return gen_rate(m, t, gname(g))[0]
weak2 = sorted(SEEN, key=lambda g: g3("SimpleCNN", "test_seen", g))[:2]
weak_txt = " and ".join(f"{g} ({g3('SimpleCNN', 'test_seen', g):.2f})" for g in weak2)
SEEN_PAR = [("Seen generators. ", True), (f"On new images from the nine training generators, the two pre-trained detectors work well: ResNet-50 reaches an accuracy of {ms('ResNet-50','test_seen','acc')} and an AUC of {ms('ResNet-50','test_seen','auc')}, and the CLIP-based detector {ms('CLIP+LR','test_seen','acc')} and {ms('CLIP+LR','test_seen','auc')}. The small CNN trained from scratch is clearly weaker (accuracy {ms('SimpleCNN','test_seen','acc')}, AUC {ms('SimpleCNN','test_seen','auc')}), and its confidence interval does not reach the two pre-trained detectors. Table 4 shows that it is weak in a specific way: it detects {', '.join(g for g in SEEN if g3('SimpleCNN','test_seen',g) >= 0.85)} well but misses most {weak_txt} images, and its per-generator rates vary strongly between seeds. The pre-trained detectors reach {rng([g3(m,'test_seen',g) for m in ('ResNet-50','CLIP+LR') for g in SEEN if g != 'GigaGAN'])} on every seen generator except GigaGAN ({g3('ResNet-50','test_seen','GigaGAN'):.2f} and {g3('CLIP+LR','test_seen','GigaGAN'):.2f}), which is the only text-conditioned GAN in the set and produces more varied images.", False)]

UNSEEN_PAR = [("Unseen generators. ", True), (f"On the three held-out generators the picture splits by family. Gansformer, a GAN with attention, is detected {'perfectly' if min(g3(m,'test_unseen','Gansformer') for m in MODELS) >= 0.995 else 'almost perfectly (' + rng([g3(m,'test_unseen','Gansformer') for m in MODELS]) + ')'} by all three detectors, including the small CNN; it is close to the StyleGAN-type models that dominate the training set. LFM and VQ-Diffusion, which are not GANs, are where the artifact-based detectors fail: ResNet-50 detects only {pct(g3('ResNet-50','test_unseen','LFM'))} and {pct(g3('ResNet-50','test_unseen','VQ-Diffusion'))} of them, and the small CNN {pct(g3('SimpleCNN','test_unseen','LFM'))} and {pct(g3('SimpleCNN','test_unseen','VQ-Diffusion'))}. The CLIP-based detector still detects {pct(g3('CLIP+LR','test_unseen','LFM'))} and {pct(g3('CLIP+LR','test_unseen','VQ-Diffusion'))}. In aggregate, ResNet-50 drops from an AUC of {f3(mean('ResNet-50','test_seen','auc'))} on seen generators to {f3(mean('ResNet-50','test_unseen','auc'))} on unseen ones, while the AUC of the CLIP-based detector {'stays at' if abs(mean('CLIP+LR','test_seen','auc') - mean('CLIP+LR','test_unseen','auc')) < 0.005 else 'moves only from'} {f3(mean('CLIP+LR','test_seen','auc'))}{'–' if abs(mean('CLIP+LR','test_seen','auc') - mean('CLIP+LR','test_unseen','auc')) < 0.005 else ' to '}{f3(mean('CLIP+LR','test_unseen','auc'))}, and its accuracy stays at {f3(mean('CLIP+LR','test_unseen','acc'))}; the confidence intervals of the two detectors on test-unseen {'do not overlap' if not ov_unseen else 'overlap'}. This is the pattern reported by Ojha et al. [4]: a classifier trained end-to-end learns the artifacts of the training generators, while fixed CLIP features separate real from generated images in a way that transfers to a new family.", False)]

_ir, _ic = tpr_i2i["ResNet-50"], tpr_i2i["CLIP+LR"]
resnet_cmp = (f"ResNet-50 detects more text-to-image samples ({pct(tpr_t2i['ResNet-50'])}) and the same share of image-to-image samples ({pct(_ir)})" if abs(_ir - _ic) < 0.015
              else f"ResNet-50 detects more SDXL-Turbo images ({pct(tpr_t2i['ResNet-50'])} and {pct(_ir)})" if _ir > _ic
              else f"ResNet-50 detects more text-to-image samples ({pct(tpr_t2i['ResNet-50'])}) but fewer image-to-image samples ({pct(_ir)})")
LOGO_PAR = [("Leave-one-generator-out. ", True), (f"Holding out the generators one at a time (Table 4, last column; Figure 3) gives the same picture with more generator choices than three. The CLIP-based detector keeps an AUC of at least {lg_min['CLIP+LR']:.2f} on every one of the 12 held-out generators (mean {lg('CLIP+LR','auc_heldout_mean'):.3f} ± {lg('CLIP+LR','auc_heldout_std'):.3f} across generators; lowest on {' and '.join(lg_hard['CLIP+LR'])}). {net_on}, ResNet-50 averages {lg('ResNet-50','auc_heldout_mean'):.3f} ± {lg('ResNet-50','auc_heldout_std'):.3f} and falls below 0.90 on {lg_nlow['ResNet-50']} generators, lowest on {' and '.join(lg_hard['ResNet-50'])} ({lg_min['ResNet-50']:.2f}); the small CNN averages {lg('SimpleCNN','auc_heldout_mean'):.3f} ± {lg('SimpleCNN','auc_heldout_std'):.3f} with a minimum of {lg_min['SimpleCNN']:.2f}. The spread across generator choices is therefore much larger for the two networks than for the CLIP-based detector, and the three generators held out in the main protocol are not an unusually easy or hard choice.", False)] if HAVE_LOGO else None
OWN_PAR = [("Our own set. ", True), (f"On the COCO + SDXL-Turbo set the accuracy at the fixed threshold of 0.5 drops for all three detectors, to {', '.join(f3(v) for v in own_acc[:2])} and {f3(own_acc[2])}. The AUC values tell a different story for each detector: {', '.join(f3(v) for v in own_auc[:2])} and {f3(own_auc[2])}. For the two networks the AUC is low, so they have largely lost the ability to order real and generated images from the new domain; for the CLIP-based detector the ordering is still informative and the problem is the threshold (Table 5). The median score of real COCO photographs is {calv('CLIP+LR','median_p_real'):.2f} and of SDXL-Turbo images {calv('CLIP+LR','median_p_fake'):.2f}, so {pct(calv('CLIP+LR','frac_fake_below05'))} of the generated images fall below 0.5. A threshold chosen on the Community Forensics validation split {help_text('CLIP+LR')} ({f3(calv('CLIP+LR','acc_thr_val'))}), because the shift is specific to the new domain. A threshold chosen on only 200 images of the new domain raises the accuracy of the CLIP-based detector to {calv('CLIP+LR','acc_thr_cal200_mean'):.3f} ± {calv('CLIP+LR','acc_thr_cal200_std'):.3f}, close to the oracle value of {f3(calv('CLIP+LR','acc_thr_oracle'))}; for ResNet-50 and the small CNN the gain is {'smaller' if max(gain_cal['ResNet-50'], gain_cal['SimpleCNN']) < gain_cal['CLIP+LR'] else 'of the same size'} ({'+' if gain_cal['ResNet-50']>=0 else ''}{gain_cal['ResNet-50']:.3f} and {'+' if gain_cal['SimpleCNN']>=0 else ''}{gain_cal['SimpleCNN']:.3f}).{oracle_sentence} At the fixed threshold the CLIP-based detector stays conservative: it raises a false alarm on {pct(fa_coco['CLIP+LR'])} of COCO photographs but detects only {pct(tpr_t2i['CLIP+LR'])} of the text-to-image samples and {pct(tpr_i2i['CLIP+LR'])} of the image-to-image samples. {resnet_cmp} at the price of flagging {pct(fa_coco['ResNet-50'])} of real COCO photographs as generated. The image-to-image samples are the hardest subset for every detector: these images inherit the composition, lighting and coarse texture of a real photograph, although every pixel has been decoded by the VAE, and the detectors treat them as real (Figure 4, top row). The paired evaluation confirms this: with the 400 source photographs removed, the AUC of the CLIP-based detector is {f3(pa('CLIP+LR','auc_noPaired'))} (full set: {f3(pa('CLIP+LR','auc_full'))}), so the pairing does not change the picture. Against their own source photographs, the text-to-image samples reach an AUC of {f3(pa('CLIP+LR','auc_t2i_vs_caption_reals'))} and the image-to-image samples only {f3(pa('CLIP+LR','auc_i2i_vs_source_reals'))}; the edited image receives a higher score than its own source in {pct(pa('CLIP+LR','paired_rate'))} of the 400 pairs for the CLIP-based detector and {pct(pa('ResNet-50','paired_rate'))} and {pct(pa('SimpleCNN','paired_rate'))} for ResNet-50 and the small CNN{', so the two networks are close to guessing between a photograph and its partial re-synthesis' if paired_close else ''}.", False)]

FA_PAR = [("False alarms. ", True), (f"The false alarm rate on real images also changes between sources. On the real images of Community Forensics-Small, the three detectors flag {pct(fa_cf['SimpleCNN'])}, {pct(fa_cf['ResNet-50'])} and {pct(fa_cf['CLIP+LR'])} of photographs; on COCO photographs they flag {pct(fa_coco['SimpleCNN'])}, {pct(fa_coco['ResNet-50'])} and {pct(fa_coco['CLIP+LR'])}. For ResNet-50 the false alarm rate {'quadruples' if fa_coco['ResNet-50'] >= 3.5*fa_cf['ResNet-50'] else 'triples' if fa_coco['ResNet-50'] >= 2.5*fa_cf['ResNet-50'] else 'doubles' if fa_coco['ResNet-50'] >= 1.5*fa_cf['ResNet-50'] else 'rises'}. Figure 4 (bottom row) shows typical false alarms: smooth water surfaces, repetitive textures, dark scenes and low-detail objects. Part of the drop on our own set is therefore not caused by the new generator at all, but by the new source of real photographs, whose compression history and content differ from the real images the detectors were trained on.", False)]

FIG2_CAP = f"Figure 2: AUC (left) and accuracy at threshold 0.5 (right) of the three detectors on the three test sets. Bars are means over {NSW} seeds; error bars are 95% bootstrap confidence intervals. All detectors lose from seen generators to unseen generators to our own set; the CLIP-based detector loses the least on unseen generators, and on our own set its AUC stays well above chance while its accuracy at the fixed threshold does not."
FIG3_CAP = ("Figure 3: Leave-one-generator-out. AUC of each detector on the 125 images of a generator against the test-seen real images when that generator is excluded from training (one seed per fold); generators sorted by the CLIP-based detector" + ("." if n_net == 12 else f"; network bars are missing for the {12 - n_net} folds that were not run.") if HAVE_LOGO else "Figure 3: Detection rate per generator for the CLIP-based detector (threshold 0.5). Blue: generators seen in training; orange: held-out generators; green: SDXL-Turbo from our own set.")
FIG4_CAP = "Figure 4: Errors of the CLIP-based detector on our own set. Top: image-to-image samples with the lowest generated-probability (missed fakes). Bottom: real COCO photographs with the highest generated-probability (false alarms)."
FIG5_CAP = f"Figure 5: Accuracy on our own set as a function of the decision threshold (scores averaged over seeds for the networks). The dashed line marks the fixed threshold of 0.5 used in Tables 3 and 4. For the CLIP-based detector the best threshold lies {clip_thr_dir} 0.5 (about {calv('CLIP+LR','thr_oracle'):.2f}) and gives an accuracy of about {f2(calv('CLIP+LR','acc_thr_oracle'))}."

logo_disc = (f"The leave-one-generator-out analysis adds that the middle level does not depend on which three generators were held out: across all 12 choices the CLIP-based detector never falls below an AUC of {lg_min['CLIP+LR']:.2f}, while ResNet-50 ranges from {lg_min['ResNet-50']:.2f} to {lg_max_r:.2f}{net_tail}. " if HAVE_LOGO else "")
DISC_3LEVELS = [("Three levels of generalization. ", True), (f"Our three test sets measure different things. On new images from the training generators nothing is lost; from seen to unseen generators the loss depends on the family of the new generator (a new GAN is easy for everyone, a new non-GAN model is hard for the artifact-based detectors and easy for the CLIP-based one); from the benchmark to our own data every detector loses most of its margin above chance at the fixed threshold. The ranking of the detectors is the same at every level (CLIP > ResNet-50 > SimpleCNN) and the AUC confidence intervals of the CLIP-based detector and ResNet-50 {ov_text}, so with our test sizes this ranking is {ov_verdict}. The gap between the benchmark and our own set is in any case much larger than the gap between detectors, as Yan et al. [28] warn. {logo_disc}With twelve generators from one benchmark and one external generator we cannot say how far these numbers transfer to other generators; what we can say is that the three-level drop is clearly visible even in a study of this size.", False)]

DISC_CLIP = [("Why the CLIP-based detector generalizes better. ", True), (f"The CLIP encoder was never trained on our data; only a logistic regression with a strong regularization (C = {clip_C:g} was selected on the validation set) was fitted on top of it. The classifier therefore cannot overfit to the fine artifacts of the nine training generators, and the features it uses are the general-purpose features CLIP learned from 400 million web images. The price is visible on our own set: the detector is confident about real photographs but not about SDXL-Turbo images, because nothing in its training set looked like a modern latent-diffusion output.", False)]

DISC_CALIB = [("Ranking versus calibration. ", True), (f"The drop on our own set has two different causes, and Table 5 separates them. For the small CNN and ResNet-50 the AUC itself is low ({f2(own_auc[0])} and {f2(own_auc[1])}), and even the oracle threshold only brings their accuracy to {f2(calv('SimpleCNN','acc_thr_oracle'))} and {f2(calv('ResNet-50','acc_thr_oracle'))}; these detectors have mostly lost the ability to tell real COCO photographs from SDXL-Turbo images, so their problem is ranking. For the CLIP-based detector the AUC of {f2(own_auc[2])} shows that the ordering is largely intact, but its scores for generated images have moved below 0.5 (median {calv('CLIP+LR','median_p_fake'):.2f}), so its problem at the fixed threshold is mainly calibration. A threshold chosen on only 200 images of the new domain recovers an accuracy of {f2(calv('CLIP+LR','acc_thr_cal200_mean'))}, against {f2(calv('CLIP+LR','acc_thr05'))} at 0.5. The practical point is that a detector deployed on a new source of images should have its threshold checked on a small labelled sample from that source, and that accuracy at a fixed 0.5 threshold on a new domain should not be read as the detector's ability to separate the classes, nor described as “chance level” when the AUC says otherwise. We did not fine-tune any detector on our own set, because the question of the study is what happens without adaptation.", False)]

DISC_EDIT = [("Why edited photographs are the hardest case. ", True), (f"Our image-to-image samples were produced from real COCO photographs with a strength of 0.5, that is, only the last 2 of 4 denoising steps were run from the noised latent of the photograph, so the result inherits the layout, colours and coarse texture of the original while every pixel is decoded by the VAE, and is only partially re-synthesized. Every detector found them harder than the text-to-image samples ({rngpct(tpr_i2i.values())} versus {rngpct(tpr_t2i.values())} detected), and in the paired comparison the edited image scores above its own source in only {rngpct([pa(m,'paired_rate') for m in MODELS])} of the pairs. This is the case that matters most in practice, because “AI-enhanced” photographs are far more common on social media than fully generated scenes, and the case our training data covers least: Community Forensics contains generated images, not edited photographs.", False)]

DISC_REAL = [("The role of the real-image source. ", True), (f"A result that we did not expect at the start is how much the real images matter. The false alarm rate of ResNet-50 rose from {pct(fa_cf['ResNet-50'])} on Community Forensics real images to {pct(fa_coco['ResNet-50'])} on COCO photographs, with no change in the detector. A detector that is evaluated only against the real images of its own training distribution will look better than it is. We therefore think that a fair evaluation needs a second, independent source of real photographs, and that the normalization steps described in Section 3.3 (one file format, one resizing pipeline) are necessary but not sufficient: they remove the trivial shortcuts, but the detectors still partly learn what “real” looks like in one particular dataset.", False)]

LIMITS = [("Limitations. ", True), (f"The study is small, which limits the conclusions. First, the leave-one-generator-out analysis covers the 12 generators of our subset{" (all of them for the CLIP-based detector, " + str(n_net) + " for the two networks)" if (HAVE_LOGO and n_net < 12) else ""}, but they come from one benchmark and we have a single external generator, so we cannot separate the effect of the generator family from that of the individual generator; the drop on LFM and VQ-Diffusion may be a property of these two models rather than of non-GAN generators in general. Second, the training set has {cnt(n_train)} images and 125 images per generator, and each seen generator has only about {per_gen_seen} test images, so the per-generator rates in Table 4 are coarse even when averaged over seeds; the bootstrap intervals in Table 3 show how much of a difference can be noise. Third, the Community Forensics subset we could process on free hardware is dominated by GANs and contains no modern latent-diffusion model, because the generators with small parquet files are mostly the hand-run models of the Manual subset; the training set is therefore not representative of the current generator population, which is part of why the transfer to SDXL-Turbo is hard. Fourth, the near-duplicate check combines a perceptual hash with CLIP cosine similarity; it finds near-copies and look-alikes, but it cannot prove that no related pair remains. Finally, we did not test robustness to re-compression or resizing after generation, which is known to reduce detection accuracy [30], and we did not adapt any detector to the new domain beyond the threshold analysis of Table 5.", False)]

logo_concl = (f", and when each of the 12 generators is held out in turn its AUC never falls below {lg_min['CLIP+LR']:.2f}, while ResNet-50 falls to {lg_min['ResNet-50']:.2f}{net_tail}" if HAVE_LOGO else "")
CONCL = (f"We reviewed the main approaches to the detection of deepfakes and AI-generated images and why detectors fail on new generators, and we tested three baseline detectors on a dataset that combines a subset of Community Forensics with 2,000 images collected by ourselves, built so that the three evaluation settings differ only in the property we want to measure: new images, new generators and a new data source. "
    f"The results, averaged over {NSW} seeds and reported with confidence intervals, show a three-level drop that is clearly visible even in a study of this size. Across generators of the benchmark, a linear classifier on frozen CLIP features generalizes with little loss (AUC {f2(mean('CLIP+LR','test_unseen','auc'))} on three held-out generators, where a fine-tuned ResNet-50 drops to {f2(mean('ResNet-50','test_unseen','auc'))}), in line with the universal-detector results of [4, 24]{logo_concl}, with the caveat that twelve generators from one benchmark cannot establish this for non-GAN generators in general. Across data sources, the accuracy of all three detectors at a fixed threshold of 0.5 falls to {rng(own_acc)} on our own COCO + SDXL-Turbo set; for the CLIP-based detector most of this drop is a calibration problem (AUC {f2(own_auc[2])}, accuracy {f2(calv('CLIP+LR','acc_thr_cal200_mean'))} after choosing the threshold on 200 images of the new domain), while for the two networks it is mainly a loss of ranking ability. Partially re-synthesized photographs are detected in only {rngpct(tpr_i2i.values())} of cases by any detector. Beyond the numbers, three simple steps are necessary before a detection result can be trusted: a common file format and resizing pipeline for all images, a check for near-duplicates across the split boundaries, and a second, independent source of real photographs to measure false alarms honestly. We recommend that student projects in this area include all three and always report accuracy on a new domain together with AUC and a threshold analysis.")
CONCL_URL = f"Code, split files, configuration and the predictions of every detector: {GH}"

# ----------------------------------------------------------------------------- docx helpers
d = docx.Document(a.v2)
body = d.element.body
def kids(): return list(body.iterchildren())
def P(i): return Paragraph(kids()[i], d)
def find_par(prefix):
    for el in kids():
        if el.tag == qn("w:p") and Paragraph(el, d).text.strip().startswith(prefix): return el
    raise KeyError(prefix)
def find_tbl_after(el):
    nxt = el.getnext()
    while nxt is not None and nxt.tag != qn("w:tbl"): nxt = nxt.getnext()
    return nxt

def set_text(el, segments, keep_rpr_from=None):
    """Replace all runs of paragraph el with segments [(text, bold)], keeping pPr and the first run's rPr."""
    p = Paragraph(el, d)
    runs = el.findall(qn("w:r"))
    src = keep_rpr_from if keep_rpr_from is not None else (runs[-1] if runs else None)
    base = copy.deepcopy(src.find(qn("w:rPr"))) if src is not None else None
    for r in runs: el.remove(r)
    for h in el.findall(qn("w:hyperlink")): el.remove(h)
    if isinstance(segments, str): segments = [(segments, False)]
    for text, bold in segments:
      for li, line in enumerate(text.split("\n")):
        r = el.makeelement(qn("w:r"), {})
        if base is not None:
            rpr = copy.deepcopy(base)
            for tag in ("w:b", "w:bCs"):
                for x in rpr.findall(qn(tag)): rpr.remove(x)
            if bold:
                for tag in ("w:b", "w:bCs"):
                    b = rpr.makeelement(qn(tag), {}); rpr.insert(1 if rpr.find(qn("w:rFonts")) is not None else 0, b)
            r.append(rpr)
        if li > 0: r.append(r.makeelement(qn("w:br"), {}))
        t = r.makeelement(qn("w:t"), {}); t.text = line; t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); r.append(t)
        el.append(r)
    return el

def clone_par_after(template_el, anchor_el, segments):
    new = copy.deepcopy(template_el)
    for dr in new.iter(qn("w:drawing")): pass
    set_text(new, segments)
    anchor_el.addnext(new); return new

def body_rpr(el):
    runs = el.findall(qn("w:r")); return runs[0].find(qn("w:rPr")) if runs else None

def build_table(template_tbl, header, rows, widths, first_col_left=True, font_half_pts=None, keep_together=False):
    """New table with the look of template_tbl (tblPr, header/body cell formatting)."""
    tbl = copy.deepcopy(template_tbl)
    trs = tbl.findall(qn("w:tr"))
    hdr_cell = copy.deepcopy(trs[0].findall(qn("w:tc"))[1]); hdr_cell0 = copy.deepcopy(trs[0].findall(qn("w:tc"))[0])
    body_cell = copy.deepcopy(trs[1].findall(qn("w:tc"))[1]); body_cell0 = copy.deepcopy(trs[1].findall(qn("w:tc"))[0])
    last_cell = copy.deepcopy(trs[-1].findall(qn("w:tc"))[1]); last_cell0 = copy.deepcopy(trs[-1].findall(qn("w:tc"))[0])
    for tr in trs: tbl.remove(tr)
    grid = tbl.find(qn("w:tblGrid"))
    for g in list(grid): grid.remove(g)
    for w in widths:
        gc = grid.makeelement(qn("w:gridCol"), {qn("w:w"): str(w)}); grid.append(gc)
    tbl.find(qn("w:tblPr")).find(qn("w:tblW")).set(qn("w:w"), str(sum(widths)))
    def mk_row(vals, c0, c, bold):
        tr = tbl.makeelement(qn("w:tr"), {})
        for j, v in enumerate(vals):
            tc = copy.deepcopy(c0 if (j == 0 and first_col_left) else c)
            tc.find(qn("w:tcPr")).find(qn("w:tcW")).set(qn("w:w"), str(widths[j]))
            p = tc.find(qn("w:p")); set_text(p, [(str(v), bold)])
            if not bold and not keep_together:
                for kn in p.find(qn("w:pPr")).findall(qn("w:keepNext")): p.find(qn("w:pPr")).remove(kn)
            if font_half_pts:
                for sz in p.iter(qn("w:sz")): sz.set(qn("w:val"), str(font_half_pts))
                for sz in p.iter(qn("w:szCs")): sz.set(qn("w:val"), str(font_half_pts))
            tr.append(tc)
        tbl.append(tr)
    mk_row(header, hdr_cell0, hdr_cell, True)
    for i, r in enumerate(rows):
        mk_row(r, last_cell0 if i == len(rows) - 1 else body_cell0, last_cell if i == len(rows) - 1 else body_cell, False)
    return tbl

def replace_picture(par_el, png, width_emu):
    p = Paragraph(par_el, d)
    for r in par_el.findall(qn("w:r")):
        if r.find(qn("w:drawing")) is not None: par_el.remove(r)
    run = p.add_run(); run.add_picture(png, width=Emu(width_emu))

# ----------------------------------------------------------------------------- apply
set_text(find_par("Detection of Deepfakes and AI-Generated Images"), TITLE)
set_text(find_par("Images produced by generative models are now easy"), ABSTRACT)
set_text(find_par("The second part is an empirical study."), INTRO2)
set_text(find_par("Community Forensics is distributed as parquet files"), SEC31)
set_text(find_par("The generated images were produced with SDXL-Turbo"), SEC32)
fig1 = find_par("Figure 1: Random examples"); set_text(fig1, Paragraph(fig1, d).text.replace("a real COCO photograph re-drawn by SDXL-Turbo (image-to-image)", "a real COCO photograph partially re-synthesized by SDXL-Turbo (image-to-image, strength 0.5)"))

# ---- Section 2 condensed (reviewer-neutral; every reference [1]-[30] stays cited) and shorter roadmap
RW21 = ("The term deepfake first meant face-swap videos, and the first large benchmarks (FaceForensics++ [1], DFDC [9], Celeb-DF [10]) were built around face manipulation. In parallel, generative adversarial networks (GANs) [11] learned to generate whole images: StyleGAN and its successors [12, 13] produce faces that people cannot tell from photographs, and class-conditional models such as BigGAN [14] did the same for natural images. Since 2021, diffusion models [15, 16] and the text-to-image systems built on them (GLIDE [17], Stable Diffusion [16], SDXL [18]) have become the dominant family; distilled versions such as SDXL-Turbo [8] generate a photorealistic image in a few steps on a consumer GPU. The detection problem has therefore moved from “find the face swap” to “decide whether any image, of any content, was generated by any of thousands of models”.")
RW22 = [("Artifact-based CNN detectors ", True), ("train a binary classifier directly on real and generated images. Rössler et al. [1] detected face manipulations with an XceptionNet, and Wang et al. [3] showed that a ResNet-50 trained on a single GAN with blur and JPEG augmentation transfers to other GANs of the same era, because CNN generators share low-level up-sampling artifacts [19]. The weakness follows from the strength: when the generator architecture changes, the artifacts change too. ", False),
        ("Frequency-based methods ", True), ("exploit the periodic patterns that up-sampling leaves in the Fourier or DCT spectrum [20, 21, 22]; they are cheap and interpretable but sensitive to resizing and compression, and the patterns differ between generator families. ", False),
        ("Detectors built on large pre-trained models ", True), ("follow Ojha et al. [4], who kept the image encoder of CLIP ViT-L/14 [23] frozen and trained only a nearest-neighbour or linear classifier on top; this “universal fake detector” generalizes across GAN and diffusion generators although CLIP was never trained for forensics, and Cozzolino et al. [24] confirmed that it needs few training images and tolerates compression. Frozen CLIP features with a linear classifier are now a standard strong baseline and one of our three detectors. ", False),
        ("Diffusion-specific methods ", True), ("start from the finding of Corvi et al. [5] that GAN artifacts do not transfer to diffusion images: DIRE [25] uses the reconstruction error of a pre-trained diffusion model as the detection signal, and NPR [26] uses neighbouring-pixel relationships as a generator-independent artifact. They generalize better within the diffusion family but are heavier and depend on assumptions about the generator.", False)]
RW23 = ("What “generalization” means in a paper is decided by its evaluation data. ForenSynths [3], with 11 CNN-based generators, was the GAN-era standard; GenImage [27] pairs ImageNet classes with eight mostly diffusion generators; Community Forensics [6], released in 2025 by Park and Owens, is the largest collection to date, about 2.7 million images from 4,803 generators, and its authors show that training on many generators improves generalization more than more images from a few. Yan et al. [28] built the Chameleon set from generated images that human annotators judged to be real and found most published detectors close to chance on it, a warning that benchmark numbers do not guarantee performance on images people meet online. Across these works the same pattern appears: near-perfect accuracy on the training generator, lower accuracy on the same family, and a large drop on a new family or after social-media compression. Chai et al. [29] showed that detectors rely on local low-level patches rather than global semantics, and Gragnaniello et al. [30] found that most published gains disappear under realistic resizing and compression. Two practical problems are discussed less often: real and fake images in a benchmark often come from different pipelines (generated images saved as PNG at a fixed resolution, real photographs as JPEG files of varying size), and their content may differ, so a classifier can learn the pipeline or the topic instead of the artifact. Our experiment controls both and tests generalization at three levels: new images, new generators and a new data source.")
p21 = find_par("The term deepfake originally referred"); set_text(p21, RW21)
p22a = find_par("Artifact-based CNN detectors."); set_text(p22a, RW22, keep_rpr_from=p22a.findall(qn("w:r"))[-1])
for pref in ("Frequency-based methods.", "Detectors built on large pre-trained vision models.", "Diffusion-specific methods."):
    el = find_par(pref); el.getparent().remove(el)
h23 = find_par("2.3  Datasets and benchmarks"); set_text(h23, "2.3  Benchmarks and the generalization problem")
set_text(find_par("The choice of the evaluation data largely decides"), RW23)
for pref in ("2.4  The generalization problem", "Across these works, the same pattern appears."):
    el = find_par(pref); el.getparent().remove(el)
set_text(find_par("The rest of the paper is organized as follows"), "The rest of the paper is organized as follows. Section 2 reviews detection methods, benchmarks and the generalization problem. Section 3 describes the dataset, Section 4 the detectors and the evaluation protocol, Section 5 the results. Section 6 discusses them and Section 7 concludes.")
# ---- length: tighten generic paragraphs (no numbers involved)
set_text(find_par("Generative models have changed how easy"), "Generative models have changed how easy it is to create a realistic image. A few years ago a convincing photograph of a person who does not exist required a research team and a GPU cluster; today anyone can type a sentence into a free web service and receive a photorealistic image in seconds, or “enhance” a real photograph until most of its pixels are no longer the ones the camera recorded. Such images are called deepfakes when they involve faces and AI-generated images in general. They are used for art and entertainment, but also for fraud, misinformation and harassment, which is why automatic detection of generated images has become an active research area [1, 2].")
set_text(find_par("The detection task itself looks simple"), "The detection task looks simple: given an image, decide whether it was recorded by a camera or produced by a generative model. A standard convolutional neural network (CNN) trained on one generator reaches almost 100% accuracy on new images from the same generator [3]. The problem starts when the generator changes: a detector trained on GAN images may fail on a diffusion model, and a detector trained on last year’s model may fail on this year’s [4, 5]. Because new generators appear every month, a detector that only works on known generators is of limited practical value. This generalization problem is the main topic of this paper.")
set_text(find_par("The second part of the dataset was built to test"), "The second part of the dataset tests the detectors on data that was not collected by the Community Forensics authors: 1,000 real photographs and 1,000 generated images. The real photographs were sampled at random from COCO val2017 [7], ordinary Flickr photographs with short human-written captions; we used COCO rather than our own phone photographs because it is public and makes the experiment reproducible.")
set_text(find_par("All 5,000 images were converted to the same format"), "All 5,000 images were converted to the same format before any experiment: RGB, JPEG with quality 95, at most 512 pixels on the longer side. Generators output PNG files and cameras output JPEG files, and a classifier can separate PNG from JPEG with very high accuracy without looking at the content; after the conversion all images share one compression pipeline.")
set_text(find_par("Resolution is a second source of leakage"), "Resolution is a second source of leakage: the generated images in Community Forensics are mostly 256 × 256 pixels, the real images are larger. Every image is therefore resized so that its shorter side is 256 pixels and a 224 × 224 crop is taken (random in training, central at test), so real and generated images pass through the same resizing and the detector cannot use image size as a shortcut.")
DISC_CLIP2 = [("Why the CLIP-based detector generalizes better. ", True), (f"The CLIP encoder was never trained on our data; only a strongly regularized logistic regression (C = {clip_C:g}, chosen on the validation set) was fitted on top of it, so the classifier cannot overfit to the fine artifacts of the nine training generators and relies on the general-purpose features CLIP learned from 400 million web images. The price is visible on our own set: the detector is confident about real photographs but not about SDXL-Turbo images, because nothing in its training set looked like a modern latent-diffusion output.", False)]
set_text(find_par("Why the CLIP-based detector generalizes better."), DISC_CLIP2)
DISC_REAL2 = [("The role of the real-image source. ", True), (f"The real images matter more than we expected: the false alarm rate of ResNet-50 rose from {pct(fa_cf['ResNet-50'])} on Community Forensics real images to {pct(fa_coco['ResNet-50'])} on COCO photographs with no change in the detector. A fair evaluation therefore needs a second, independent source of real photographs: the normalization of Section 3.3 removes the trivial shortcuts, but the detectors still partly learn what “real” looks like in one particular dataset.", False)]
set_text(find_par("The role of the real-image source."), DISC_REAL2)
# ---- length: reference list at 9 pt (journal style), Figure 2 slightly narrower
from docx.shared import Pt as _Pt
in_refs = False
for p in d.paragraphs:
    if p.style.name == "Heading 1": in_refs = p.text.strip().startswith("References")
    elif in_refs and p.text.strip():
        for r in p.runs: r.font.size = _Pt(9)
        p.paragraph_format.space_after = _Pt(1)
# Key words: add the two new themes of the revision
kw = find_par("Key Words:"); kwt = Paragraph(kw, d).text
if "calibration" not in kwt:
    set_text(kw, [("Key Words: ", True), (kwt.replace("Key Words: ", "").rstrip(".") + ", decision threshold calibration, confidence intervals, near-duplicate leakage.", False)], keep_rpr_from=kw.findall(qn("w:r"))[-1])
# Table 1: let the body rows break freely (otherwise LibreOffice/Word push the whole table and leave half a page blank)
t1 = find_tbl_after(find_par("Table 1: Generators"))
for tr in t1.findall(qn("w:tr"))[1:]:
    for kn in tr.iter(qn("w:keepNext")): kn.getparent().remove(kn)
# 3.4
p34 = find_par("The Community Forensics part was divided"); set_text(p34, SEC34_A)
p34b = clone_par_after(p34, p34, SEC34_B)
cap2 = find_par("Table 2: Data splits"); set_text(cap2, TAB2_CAP)
t2 = find_tbl_after(cap2); t2new = build_table(t2, ["Split", "Source", "Real", "Fake", "Total", "Purpose"], TAB2_ROWS, [1300, 2500, 900, 2000, 900, 2039]); t2.addprevious(t2new); body.remove(t2)

# 4
set_text(find_par("We compare three detectors that represent"), SEC41_A)
set_text(find_par("CLIP + linear. The image encoder"), SEC41_CLIP, keep_rpr_from=find_par("CLIP + linear. The image encoder").findall(qn("w:r"))[-1])
set_text(find_par("SimpleCNN was trained for 10 epochs"), SEC42)
p43 = find_par("Each detector is evaluated on three test sets"); set_text(p43, SEC43)
h43 = find_par("4.3  Evaluation protocol")
h44 = clone_par_after(h43, p43, "4.4  Uncertainty and threshold analysis")
anchor = h44
for seg in SEC44:
    anchor = clone_par_after(p43, anchor, seg)

# 5
set_text(find_par("Table 3 gives the main results"), SEC5_OPEN)
cap3 = find_par("Table 3: Main results"); set_text(cap3, TAB3_CAP)
t3 = find_tbl_after(cap3); t3new = build_table(t3, ["Detector", "Test set", "Accuracy @ 0.5", "AUC", "AP"], TAB3_ROWS, [1250, 1050, 2446, 2446, 2447], font_half_pts=18); t3.addprevious(t3new); body.remove(t3)
fig2cap = find_par("Figure 2: AUC (left)"); set_text(fig2cap, FIG2_CAP); replace_picture(fig2cap.getprevious(), f"{R}/figs/fig_main_paper.png", 4200000)
cap4 = find_par("Table 4: Detection rate"); set_text(cap4, TAB4_CAP)
t4 = find_tbl_after(cap4); t4new = build_table(t4, ["Generator / source", "Set", "SimpleCNN", "ResNet-50", "CLIP + linear", "AUC when held out"], TAB4_ROWS, [2600, 1100, 1300, 1300, 1300, 2039], font_half_pts=18); t4.addprevious(t4new); body.remove(t4)
set_text(find_par("Seen generators. On new images"), SEEN_PAR)
set_text(find_par("Unseen generators. On the three held-out"), UNSEEN_PAR)
pown = find_par("Our own set. On the COCO + SDXL-Turbo"); set_text(pown, OWN_PAR)
if LOGO_PAR: clone_par_after(pown, find_par("Unseen generators. On the three held-out"), LOGO_PAR)
fig3cap = find_par("Figure 3: Detection rate per generator"); set_text(fig3cap, FIG3_CAP); replace_picture(fig3cap.getprevious(), f"{R}/figs/fig_logo_paper.png" if HAVE_LOGO else f"{R}/figs/fig_per_generator_v2.png", 4500000 if HAVE_LOGO else 2800000)
fig4cap = find_par("Figure 4: Errors of the CLIP-based detector"); set_text(fig4cap, FIG4_CAP)
import os
if os.path.exists(f"{R}/figs/fig_errors_v2.png"): replace_picture(fig4cap.getprevious(), f"{R}/figs/fig_errors_v2.png", 3200000)
pfa = find_par("False alarms. The false alarm rate"); set_text(pfa, FA_PAR)
# Table 5 + Figure 5 after the "False alarms" paragraph
cap5 = clone_par_after(cap4, pfa, TAB5_CAP)
t5 = build_table(t4new, TAB5_HEAD, TAB5_ROWS, [1350, 1250, 1250, 1250, 1250, 1250, 1020, 1019], font_half_pts=18, keep_together=True); cap5.addnext(t5)
spacer = copy.deepcopy(cap4.getnext().getnext()) if cap4.getnext().getnext().tag == qn("w:p") and not Paragraph(cap4.getnext().getnext(), d).text.strip() else None
after_t5 = t5
if spacer is not None: t5.addnext(spacer); after_t5 = spacer
# Figure 5 (accuracy vs threshold) is kept in the repository only; Table 5 gives the same thresholds and accuracies

# 6
set_text(find_par("Three levels of generalization."), DISC_3LEVELS)
pclip = find_par("Why the CLIP-based detector generalizes better."); set_text(pclip, DISC_CLIP2)
clone_par_after(pclip, pclip, DISC_CALIB)
set_text(find_par("Why edited photographs are the hardest case."), DISC_EDIT)
set_text(find_par("The role of the real-image source."), DISC_REAL2)
set_text(find_par("Limitations."), LIMITS)
# 7
pc = find_par("We reviewed the main approaches"); set_text(pc, CONCL); purl = clone_par_after(pc, pc, CONCL_URL)
from docx.enum.text import WD_ALIGN_PARAGRAPH as _AL
Paragraph(purl, d).paragraph_format.alignment = _AL.LEFT

# ELCVIA checklist: single-spaced text. Draft v2 used 1.15 line spacing; set every paragraph to single.
for p in d.paragraphs:
    if p.paragraph_format.line_spacing is not None and p.paragraph_format.line_spacing != 1.0:
        p.paragraph_format.line_spacing = 1.0
fig1img = find_par("Figure 1: Random examples").getprevious()
for ext in fig1img.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent"):
    cx, cy = int(ext.get("cx")), int(ext.get("cy")); ext.set("cx", str(3500000)); ext.set("cy", str(int(cy * 3500000 / cx)))
for ext in fig1img.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"):
    cx, cy = int(ext.get("cx")), int(ext.get("cy")); ext.set("cx", str(3900000)); ext.set("cy", str(int(cy * 3900000 / cx)))
_n_fix = 0
for _p in d.paragraphs:   # reference [26]: the second author (H. Liu) was missing
    for _r in _p.runs:
        if "C. Tan, Y. Zhao, S. Wei" in _r.text: _r.text = _r.text.replace("C. Tan, Y. Zhao, S. Wei", "C. Tan, H. Liu, Y. Zhao, S. Wei"); _n_fix += 1
assert _n_fix == 1, _n_fix
d.core_properties.title = "Detection of Deepfakes and AI-Generated Images: A Review of Methods and a Small-Scale Empirical Study of Generalization to Unseen Generators"
d.core_properties.author = "Adema Akhmetova, Mansur Murzhanov"
d.save(a.out)
# ----------------------------------------------------------------------------- checks
d2 = docx.Document(a.out)
full = "\n".join(p.text for p in d2.paragraphs) + "\n" + "\n".join(c.text for t in d2.tables for r in t.rows for c in r.cells)
bad = [w for w in ("near-chance", "near chance", "chance accuracy", "[GitHub link]", "entirely re-synthesized", "single run", "re-drawn", "ViT-B/16") if w in full]
left = re.findall(r"\[[^\]]*(?:csv|json|…|\.\.\.|v2 draft)[^\]]*\]", full)
print("saved:", a.out); print("forbidden phrases left:", bad); print("placeholders left:", left[:10])
print("tables:", len(d2.tables), "| paragraphs:", len(d2.paragraphs), "| words:", len(full.split()))
