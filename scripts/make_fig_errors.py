"""Figure 4 of the paper from the revised predictions: the CLIP+LR errors on test_own.

Usage:
    python make_fig_errors.py results_v2/predictions_all.csv <image dir> results_v2/figs/fig_errors_v2.png

Top row: image-to-image samples with the lowest generated-probability (missed fakes).
Bottom row: real COCO photographs with the highest generated-probability (false alarms).
"""
import sys
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

pred_path, img_dir, out = sys.argv[1:4]
P = pd.read_csv(pred_path)
P = P[(P.model == "CLIP+LR") & (P.test == "test_own")]
missed = P[(P.label == "fake") & (P.fake_type == "i2i")].nsmallest(5, "prob_fake")
alarms = P[P.label == "real"].nlargest(5, "prob_fake")
fig, axes = plt.subplots(2, 5, figsize=(11, 4.4))
for ax, (_, r) in zip(axes[0], missed.iterrows()):
    ax.imshow(Image.open(f"{img_dir}/{r.filename}").convert("RGB")); ax.set_title(f"i2i fake, p = {r.prob_fake:.2f}", fontsize=9); ax.axis("off")
for ax, (_, r) in zip(axes[1], alarms.iterrows()):
    ax.imshow(Image.open(f"{img_dir}/{r.filename}").convert("RGB")); ax.set_title(f"COCO real, p = {r.prob_fake:.2f}", fontsize=9); ax.axis("off")
axes[0, 0].text(-0.08, 0.5, "missed fakes", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=10)
axes[1, 0].text(-0.08, 0.5, "false alarms", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=10)
plt.tight_layout(); plt.savefig(out, dpi=200); print("written:", out)
