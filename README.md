# Detection of Deepfakes and AI-Generated Images: A Small-Scale Generalization Study

Code, data splits and results for the paper
*"Detection of Deepfakes and AI-Generated Images: A Review of Methods and a Small-Scale Empirical Study of Generalization to Unseen Generators"*
(A. Akhmetova, M. Murzhanov — Astana IT University, 2026; submitted to ELCVIA).

The study trains three detectors (a small CNN from scratch, a fine-tuned ResNet-50 and a logistic regression on frozen CLIP features)
on a subset of Community Forensics-Small and tests them on (1) new images from the training generators, (2) three held-out generators
and (3) an external set of real COCO photographs and SDXL-Turbo images that we built ourselves.

## Repository layout

```
notebooks/
  01_own_dataset.ipynb          builds our external set: 1,000 COCO val2017 photos + 600 SDXL-Turbo text-to-image + 400 image-to-image
  02_cf_sample_and_merge.ipynb  samples 1,500 real + 1,500 fake (12 generators x 125) from Community Forensics-Small, normalizes, merges
  03_train_eval.ipynb           first experiment (single seed, v1 of the paper)
  04_revision.ipynb             revised experiment: near-duplicate clustering, cluster-level splits, 3 seeds, bootstrap CIs, threshold analysis
data/
  metadata.csv                  one row per image (5,000): id, label, origin (cf/own), generator, prompt, caption, original_id, held_out, split (v1)
  metadata_cf.csv               the Community Forensics part only, before merging
  splits_v1/*.csv               the splits used in v1 of the paper (train / val / test_seen / test_unseen / test_own)
  splits_v2/*.csv               the splits used in the revised paper (cluster-level, no near-duplicates across splits)
results_v1/                     figures of the single-seed run
results_v2/                     metrics per seed, mean +- std and 95% bootstrap CIs, threshold/calibration analysis, per-generator rates,
                                predictions of every detector on every test image, dedup report, config.json, figures
```

Images themselves are not redistributed (355 MB). They can be rebuilt exactly with notebooks 01 and 02 (fixed seeds) or requested
from the authors. All images are converted to JPEG (quality 95), RGB, at most 512 px on the longer side; `metadata.csv` keeps the
provenance of every image (generator / real source, prompt, source URL for COCO, original image id for image-to-image samples).

## How the data were selected

* **Community Forensics part** (notebook 02). From the Hugging Face release `OwensLab/CommunityForensics-Small` we read only parquet
  files below 1.5 GB (they are grouped by folder and interleaved), reading metadata only. Among generators with at least 125 images
  in the files read, 12 were drawn with `random.sample` (seed 42); 3 of these 12 were drawn again with `random.sample` as held-out
  generators. 125 images per generator and 1,500 real images were drawn with `DataFrame.sample(random_state=42)`.
  Real images come from the "Real" part of CF-Small (the release does not record their original source).
* **External set** (notebook 01). 1,000 random COCO val2017 photographs (seed 42) with their first human caption.
  600 text-to-image samples: SDXL-Turbo (`stabilityai/sdxl-turbo`), 512x512, 4 steps, guidance 0, prompt = caption + ", realistic photo".
  400 image-to-image samples: SDXL-Turbo img2img, input = a different COCO photograph (not used for a text prompt) resized to 512x512,
  strength 0.5 (i.e. 2 of 4 denoising steps starting from the noised latent of the input), prompt = its caption + ", realistic photo".
  `original_id` links every image-to-image sample to its source photograph.
* **Splits.** v1: 70/15/15 per generator and for real images; 3 held-out generators -> `test_unseen`; external set -> `test_own`.
  v2 (notebook 04): images of the same generator with dHash Hamming distance <= 8 are clustered (union-find) and a cluster is assigned
  to a single split; see `results_v2/dedup_report.json` for the numbers.

## Configuration (notebook 04)

| item | value |
|---|---|
| input | shorter side resized to 256, 224x224 crop (random in training, center at test), horizontal flip in training |
| SimpleCNN | 4 x [conv3x3, BN, ReLU, maxpool] with 32/64/128/256 channels, GAP, dropout 0.3, linear; ~0.4 M params; AdamW lr 1e-3, wd 1e-4, cosine, 10 epochs |
| ResNet-50 | torchvision `IMAGENET1K_V2` weights, all layers fine-tuned, AdamW lr 1e-4, wd 1e-4, cosine, 5 epochs |
| loss | BCE with positive-class weight = n_real / n_fake of the training split; batch 32; best epoch by validation AUC |
| CLIP + LR | frozen CLIP image encoder (ViT-L/14 on GPU, ViT-B/16 on CPU), L2-normalized embeddings, StandardScaler, logistic regression (class_weight balanced), C in {0.01, 0.1, 1, 10} chosen on validation AUC |
| seeds | 0, 1, 2 for the networks (init + data order); CLIP + LR is deterministic |
| uncertainty | mean +- std over seeds; 95% percentile bootstrap over test images (1,000 resamples, metric averaged over seeds) |
| threshold analysis | accuracy on test_own at 0.5, at the threshold maximizing balanced accuracy on the CF validation split, at a threshold chosen on 200 stratified target-domain images (20 repeats, evaluated on the other 1,800), and at the oracle threshold |

Exact values used in a run are written to `results_v2/config.json`.

## Reproducing

All notebooks run in Google Colab (free tier). Order: 01 -> 02 -> 04 (03 is the first, single-seed version).
Notebook 04 is resumable: predictions of every model/seed are saved to Google Drive as soon as they exist.
Approximate times: 01 ~1 h on a T4; 02 ~30 min (CPU); 04 ~1.5 h on a T4 or ~8 h on CPU.

Dependencies: `torch`, `torchvision`, `transformers`, `diffusers` (01 only), `pyarrow`, `huggingface_hub`, `scikit-learn`, `pandas`, `Pillow`, `matplotlib`.

## Data sources and licenses

* Community Forensics-Small: Park & Owens, CVPR 2025 — https://huggingface.co/datasets/OwensLab/CommunityForensics-Small (CC BY 4.0)
* COCO val2017: Lin et al., ECCV 2014 — https://cocodataset.org (images under their Flickr licenses, annotations CC BY 4.0)
* SDXL-Turbo: Sauer et al., 2023 — https://huggingface.co/stabilityai/sdxl-turbo
* CLIP: Radford et al., ICML 2021 — https://huggingface.co/openai

Code in this repository is released under the MIT License.
## Revised run (paper v3)

The revised experiments reported in v3 of the paper were run **locally** with `scripts/run_v2_local.py`, a line-by-line port of
`notebooks/04_revision.ipynb` (same seeds, same splits, same protocol) that runs on CUDA, Apple MPS or CPU.
Hardware of the reported run: Apple M3 laptop GPU (PyTorch 2.12, MPS backend), CLIP ViT-L/14 in float32.

```
scripts/
  run_v2_local.py      dHash near-duplicate clustering -> cluster-level splits (seed 42) -> SimpleCNN / ResNet-50 x seeds 0,1,2
                       -> CLIP ViT-L/14 + logistic regression -> metrics per seed, mean +- std, 95% bootstrap CIs
                       -> threshold / calibration analysis on test_own -> per-generator rates -> figures -> predictions_all.csv, config.json
  paired_analysis.py   paired evaluation of test_own (400 source photographs vs their image-to-image edits) from predictions_all.csv
  make_fig_errors.py   Figure 4: missed image-to-image fakes and false alarms of CLIP+LR on test_own
  build_v3_docx.py     writes every number, table and figure of the revised paper from results_v2/ into the manuscript (no hand-typed numbers)
```

Reproduce:

```
python scripts/run_v2_local.py --zip dataset.zip --root /tmp/dataset --res results_v2 --workers 2
python scripts/paired_analysis.py results_v2/predictions_all.csv results_v2/metadata_v2.csv results_v2/paired_analysis.csv
python scripts/make_fig_errors.py results_v2/predictions_all.csv /tmp/dataset/images results_v2/figs/fig_errors_v2.png
```

`results_v2/` contains the complete output of the reported run: `metadata_v2.csv` and `splits_v2/*.csv` (cluster-level splits),
`dedup_report.json`, `trainlog_*_seed*.csv`, `pred_*_seed*.npz`, `predictions_all.csv` (score of every detector and seed on every
validation and test image), `metrics_per_seed.csv`, `metrics_summary.csv`, `calibration_test_own.csv`, `calibration_summary.csv`,
`per_generator.csv`, `paired_analysis.csv`, `config.json` and `figs/`.

Note on determinism: the MPS backend is not bit-exact between runs, so a re-run reproduces the numbers up to the seed-to-seed
variation reported in the paper (the standard deviations in Tables 3-5), not to the last digit. CLIP+LR is deterministic given the features.

## Paper

`paper/Akhmetova_Murzhanov_ELCVIA_draft_v3.pdf` is the revised manuscript (v3) built from `results_v2/` with `scripts/build_v3_docx.py`.

## Revision after the second review (paper v4)

`results_v3/` is the run reported in v4 (`paper/Akhmetova_Murzhanov_ELCVIA_draft_v4.pdf`):

* near-duplicate clusters built with two criteria — dHash Hamming distance <= 8 **or** CLIP ViT-L/14 cosine similarity >= 0.95 within the same generator / real source; splits in `data/splits_v3/` (no pair of either kind crosses a split);
* SimpleCNN and ResNet-50 with five seeds (0-4);
* bootstrap confidence intervals resampled by unit: real photograph + the image generated from it on test_own (1,000 units), near-duplicate cluster on the Community Forensics test sets;
* `check_overlap.py`: Community Forensics real images vs COCO photographs (dHash and CLIP) — no overlap (`results_v3/overlap_report.json`, `results_v3/figs/fig_overlap_pairs.png`);
* `run_logo.py`: leave-one-generator-out over all 12 generators (`results_v3/logo_results.csv`, `logo_summary.csv`, `figs/fig_logo.png`).

```
python scripts/run_v2_local.py --zip dataset.zip --root /tmp/dataset --res results_v3 --workers 2 --seeds 0 1 2 3 4
python scripts/run_logo.py     --zip dataset.zip --root /tmp/dataset --res results_v3 --workers 2 --seed 0
python scripts/check_overlap.py results_v3/metadata_v2.csv /tmp/dataset/images results_v3/clip_feats_all.npz results_v3/overlap_report.json results_v3/figs/fig_overlap_pairs.png
python scripts/paired_analysis.py results_v3/predictions_all.csv results_v3/metadata_v2.csv results_v3/paired_analysis.csv
```
