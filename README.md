# Detection of Deepfakes and AI-Generated Images

Code, split metadata, and evaluation outputs for the ELCVIA submission:

> Detection of Deepfakes and AI-Generated Images: A Review of Methods and a Small-Scale Empirical Study of Generalization to Unseen Generators
>
> Adema Akhmetova and Mansur Murzhanov, Astana IT University, 2026

The study compares a small CNN trained from scratch, a fine-tuned ResNet-50, and logistic regression on frozen CLIP image features. The experiment reported in the current manuscript is in `results_v3/`.

## Repository contents

- `notebooks/`: construction of the external COCO + SDXL-Turbo set, sampling from Community Forensics-Small, and earlier and revised training workflows.
- `scripts/`: local training, leave-one-generator-out evaluation, duplicate checks, paired analysis, and figure generation.
- `data/`: image metadata and the v1–v3 split manifests. Images are not included.
- `results_v1/` and `results_v2/`: earlier experiment outputs.
- `results_v3/`: the reported five-seed run, including metrics, predictions, split metadata, duplicate and overlap reports, and figures.

## Data and reproduction

The image files are not redistributed. Rebuild the Community Forensics and COCO subsets with notebooks `01_own_dataset.ipynb` and `02_cf_sample_and_merge.ipynb`, then run the revised analysis. The source dataset is [Community Forensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small); real photographs are sampled from [COCO val2017](https://cocodataset.org/); generated samples use [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo).

The reported run used Apple M3 / PyTorch MPS and seeds 0–4 for the CNN and ResNet-50. Run configuration and software versions are recorded in `results_v3/config.json`.

```bash
python scripts/run_v2_local.py --zip dataset.zip --root /tmp/dataset --res results_v3 --workers 2 --seeds 0 1 2 3 4
python scripts/run_logo.py --zip dataset.zip --root /tmp/dataset --res results_v3 --workers 2 --seed 0
python scripts/check_overlap.py results_v3/metadata_v2.csv /tmp/dataset/images results_v3/clip_feats_all.npz results_v3/overlap_report.json results_v3/figs/fig_overlap_pairs.png
python scripts/paired_analysis.py results_v3/predictions_all.csv results_v3/metadata_v2.csv results_v3/paired_analysis.csv
```

Install the dependencies listed in `requirements.txt`. Re-running on MPS may produce small numerical differences; the paper reports variation across random seeds.

## Licensing

The code is released under the MIT License in `LICENSE`. The source image datasets and pretrained models have their own licenses and terms; see their respective source pages before using or redistributing images or model weights. No image files are included here.
