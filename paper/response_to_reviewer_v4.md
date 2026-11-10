# Response to the second review (comments on draft v3)

Paper: *Detection of Deepfakes and AI-Generated Images: A Review of Methods and a Small-Scale Empirical Study of Generalization to Unseen Generators*
Authors: Adema Akhmetova, Mansur Murzhanov (Astana IT University)
Revision: v4, prepared 9–10 November 2026

Dear Kaiyrkhan Nurym,

Thank you for the second round of comments. We addressed all eight points and the five inconsistencies. The experiment was re-run from the splits onward, so every number in v4 is new; the repository (https://github.com/a33558771-prog/deepfake-generalization) contains the new results in `results_v3/` next to the previous run in `results_v2/`.

---

## 1. Only three held-out generators and one external generator

**Done: leave-one-generator-out over all 12 generators for the three detectors.** Each of the 12 generators was held out in turn; the detectors were trained on the cluster-level training split of the other 11 (real images unchanged), the epoch and the regularization parameter were chosen on the validation split without that generator, and the 125 images of the excluded generator were evaluated against the 225 real images of test-seen (one seed per fold). Section 4.4 describes the protocol; Table 4 has a new column with the held-out AUC of every generator for the three detectors; Figure 3 now shows these AUC values (it replaces the old per-generator bar chart, whose information is already in Table 4); Section 5 has a new paragraph and Section 6 and the Conclusion refer to it.

Result: the CLIP-based detector has a mean held-out AUC of 0.987 ± 0.022 and never falls below 0.93 (lowest on GLIDE, 0.93, and GigaGAN, 0.95). ResNet-50 averages 0.934 ± 0.068 and falls below 0.90 on 4 of the 12 generators (GLIDE 0.80, LFM 0.85, GigaGAN 0.87 and VQ-Diffusion 0.89); the small CNN averages 0.759 ± 0.165 with a minimum of 0.54 (BigGAN) and stays below 0.70 on 6 generators (BigGAN 0.54, GigaGAN 0.59, StyleGAN-XL 0.62, VQ-Diffusion 0.68, GLIDE 0.68 and LFM 0.69). The spread across generator choices is therefore much larger for the two networks than for the CLIP-based detector, and the three generators held out in the main protocol are not an unusually easy or hard choice. The networks are trained once per fold (seed 0), so this analysis measures the spread across generators, not across seeds; the paper says so in Section 4.4.

**Not done: a second external generator.** Generating a second external set (SD 1.5 or FLUX-schnell) on the laptop that runs the experiments would not fit in the time before the deadline next to the leave-one-generator-out runs, and the free Colab GPU quota is exhausted. We therefore kept the conclusions as tightly scoped as in v3: the Introduction, Section 6 (Three levels; Limitations) and the Conclusion now say "twelve generators from one benchmark and one external generator" wherever a claim about generalization is made.

**Where:** §4.4 (new paragraph "Leave-one-generator-out"); Table 4 (new column); Figure 3; §5 (new paragraph); §6; §7.

## 2. Five seeds instead of three

**Done.** SimpleCNN and ResNet-50 were trained with seeds 0–4 on the new splits (Section 4.2). All means, standard deviations, intervals, Table 5 and the paired analysis are over five seeds. The standard deviations of the small CNN grew noticeably with five seeds (for example its detection rate on the text-to-image samples is now 0.46 ± 0.12), which is itself a useful observation: the small network is unstable across initializations, and Section 5 says so.

## 3. Possible COCO overlap

**Checked; there is none.** We compared the 1,000 COCO photographs with the 1,500 Community Forensics real images (all of train, val and test-seen) with both dHash and CLIP ViT-L/14 cosine similarity: no pair at Hamming distance ≤ 4, no pair at cosine ≥ 0.95 (maximum 0.944), and the single pair at Hamming distance ≤ 8 has cosine 0.62, i.e. two different photographs. A contact sheet of the eight closest pairs (`results_v3/figs/fig_overlap_pairs.png`) shows different photographs of similar scenes (zebras in savannah, a pizza, an airport). The check is `scripts/check_overlap.py`; the report is `results_v3/overlap_report.json`; Section 3.4 states the result.

## 4. Bootstrap design

**Done.** The bootstrap now resamples units, not images. On test-own a unit is a real photograph together with the generated image that was made from it (the 400 image-to-image edits via their source photograph, the 600 text-to-image samples via the photograph whose caption was the prompt), 1,000 units in all; on the Community Forensics test sets a unit is a near-duplicate cluster (385 units in test-seen, 595 in test-unseen). Section 4.4 and the caption of Table 3 describe this. The intervals became slightly wider on test-own, as expected.

## 5. Wording of the image-to-image description

**Done.** We removed the claim that image-to-image samples keep "the camera-like noise of a real photograph". Section 3.2 now says that the result "inherits the layout, the colours and the coarse texture of the original photograph, but every pixel is decoded by the VAE of the model, so the low-level signal is synthetic even though the scene is kept"; the same correction is in Section 5 ("Our own set") and Section 6 ("Why edited photographs are the hardest case").

## 6. Missing generation details

**Done.** Section 3.2 now states: fp16 weights of `stabilityai/sdxl-turbo`, the diffusers library (≥ 0.27, Google Colab, T4), the default scheduler of the model repository (EulerAncestralDiscreteScheduler), 512 × 512, 4 steps, guidance scale 0, strength 0.5 for image-to-image, and that every image received its own generator seed drawn from Python's `random` module seeded with 42, so the sequence of seeds is reproducible from notebook 01. The exact diffusers version installed by Colab on 2 November 2026 was not recorded; the notebook pins the minimum version.

## 7. Abstract wording

**Done.** "12 randomly selected generators" is now "12 generators drawn at random from those available in the smaller files of the release", and Section 3.1 keeps the full five-step description.

## 8. A second duplicate check on CLIP embeddings

**Done, and it changed the splits.** With CLIP ViT-L/14 cosine similarity ≥ 0.95 within the same generator we found 262 look-alike pairs, 64 of them not in any dHash cluster, and 114 of them crossed a split boundary in the v3 splits (almost all StyleGAN2/StyleGAN3 faces). Clusters are now built with both criteria (dHash Hamming ≤ 8 OR CLIP cosine ≥ 0.95), the splits were rebuilt (2,875 clusters, largest 67 images; train 1,050 / 781, val 225 / 171, test-seen 225 / 173) and no pair of either kind crosses a boundary. Section 3.4 reports both criteria; the Limitations paragraph now says that the check "finds near-copies and look-alikes but cannot prove that no related pair remains".

The main numbers moved little with the new splits and five seeds (test-own: CLIP AUC 0.849 vs 0.852, ResNet-50 0.698 vs 0.683, SimpleCNN 0.633 vs 0.621; test-unseen: CLIP 0.998 vs 0.998, ResNet-50 0.908 vs 0.904), which is reassuring about the earlier dedup.

---

## Small inconsistencies

- *"ResNet-50 detects more SDXL-Turbo images (57% and 43%)"* — the sentence is now generated from the numbers: with five seeds ResNet-50 detects 63% of text-to-image and 47% of image-to-image samples against 51% and 42% for CLIP, so "more" is now correct for both subsets; had the two been tied, the text would say so.
- *Median fake score 0.47 vs "most"* — replaced by the exact share: "52% of the generated images fall below 0.5" (`frac_fake_below05` in `calibration_test_own.csv`).
- *"every detector loses most of its accuracy"* — now "loses most of its margin above chance at the fixed threshold".
- *SimpleCNN gain +0.005 vs +0.006* — the gain is now computed from the rounded values shown in Table 5, so text and table agree (five-seed values: +0.004 and +0.003 for ResNet-50 and SimpleCNN).
- *"From seen images to seen generators nothing is lost"* — now "On new images from the training generators nothing is lost".

## Other changes

- Every number, table and figure is written from `results_v3/` by `scripts/build_v3_docx.py`; the paper is 12 pages. To stay within 12 pages, the accuracy-versus-threshold curve (Figure 5 of v3) was moved to the repository (`results_v3/figs/fig_calibration_v2.png`); Table 5 gives the same thresholds and accuracies.
- `scripts/run_v2_local.py` now takes `--seeds` and builds the two-criterion clusters; `scripts/run_logo.py` runs the leave-one-generator-out analysis; `scripts/check_overlap.py` is the COCO / CLIP duplicate check; `scripts/logo_collect.py` assembles the leave-one-generator-out table from the saved folds.

We would be grateful for any further comments.

Adema Akhmetova, Mansur Murzhanov
