"""Paired evaluation of the external set (test_own) for the deepfake generalization study.

Usage:
    python paired_analysis.py results_v2/predictions_all.csv results_v2/metadata_v2.csv results_v2/paired_analysis.csv

Inputs
    predictions_all.csv  written by 04_revision.ipynb: one row per (model, seed, test, image) with prob_fake
    metadata_v2.csv      written by 04_revision.ipynb (or data/metadata.csv): must have id, label, fake_type, original_id

Output (one row per model, mean and std over seeds)
    auc_full, acc_full                 the usual test_own numbers (1,000 real + 1,000 fake), threshold 0.5
    auc_noPaired, acc_noPaired         test_own without the 400 real photographs that are sources of i2i samples
    auc_t2i_vs_caption_reals           600 t2i samples vs the 600 real photographs whose captions were their prompts
    auc_i2i_vs_source_reals            400 i2i samples vs their 400 source photographs
    paired_rate                        fraction of the 400 (source, edit) pairs where prob_fake(edit) > prob_fake(source)
    paired_margin_median               median of prob_fake(edit) - prob_fake(source)
Only the standard library is used, so the script runs anywhere Python 3 runs.
"""
import csv
import statistics
import sys
from collections import defaultdict


def auc(pos, neg):
    """ROC-AUC by the rank (Mann-Whitney) formula; ties count 1/2."""
    if not pos or not neg:
        return float("nan")
    scores = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    ranks, i = {}, 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[j + 1][0] == scores[i][0]:
            j += 1
        r = (i + j) / 2 + 1  # average rank, 1-based
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k, (s, y) in enumerate(scores) if y == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def accuracy(pos, neg, thr=0.5):
    correct = sum(1 for s in pos if s >= thr) + sum(1 for s in neg if s < thr)
    return correct / (len(pos) + len(neg))


def main(pred_path, meta_path, out_path):
    meta = {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["id"]] = r
    i2i_ids = [i for i, r in meta.items() if r.get("fake_type") == "i2i"]
    source_of = {i: meta[i]["original_id"] for i in i2i_ids}
    source_ids = set(source_of.values())
    missing = [s for s in source_ids if s not in meta or meta[s]["label"] != "real"]
    if missing:
        sys.exit(f"{len(missing)} original_id values do not point to a real image in the metadata: {missing[:5]}")

    # scores[(model, seed)][id] = prob_fake, test_own rows only
    scores = defaultdict(dict)
    with open(pred_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["test"] != "test_own":
                continue
            scores[(r["model"], r["seed"])][r["id"]] = float(r["prob_fake"])
    if not scores:
        sys.exit("no test_own rows found in the predictions file")

    per_seed = defaultdict(list)  # model -> list of dicts
    for (model, seed), sc in sorted(scores.items()):
        ids = set(sc)
        real = [i for i in ids if meta[i]["label"] == "real"]
        fake = [i for i in ids if meta[i]["label"] == "fake"]
        t2i = [i for i in fake if meta[i]["fake_type"] == "t2i"]
        i2i = [i for i in fake if meta[i]["fake_type"] == "i2i"]
        real_src = [i for i in real if i in source_ids]
        real_cap = [i for i in real if i not in source_ids]
        p = lambda lst: [sc[i] for i in lst]
        pairs = [(sc[e], sc[source_of[e]]) for e in i2i if source_of[e] in sc]
        margins = [e - s for e, s in pairs]
        per_seed[model].append(dict(
            seed=seed, n_real=len(real), n_fake=len(fake), n_pairs=len(pairs),
            auc_full=auc(p(fake), p(real)), acc_full=accuracy(p(fake), p(real)),
            auc_noPaired=auc(p(fake), p(real_cap)), acc_noPaired=accuracy(p(fake), p(real_cap)),
            auc_t2i_vs_caption_reals=auc(p(t2i), p(real_cap)),
            auc_i2i_vs_source_reals=auc(p(i2i), p(real_src)),
            paired_rate=(sum(1 for m in margins if m > 0) + 0.5 * sum(1 for m in margins if m == 0)) / len(margins) if margins else float("nan"),
            paired_margin_median=statistics.median(margins) if margins else float("nan"),
        ))

    keys = ["auc_full", "acc_full", "auc_noPaired", "acc_noPaired", "auc_t2i_vs_caption_reals",
            "auc_i2i_vs_source_reals", "paired_rate", "paired_margin_median"]
    rows = []
    for model, lst in per_seed.items():
        row = dict(model=model, n_seeds=len(lst), n_real=lst[0]["n_real"], n_fake=lst[0]["n_fake"], n_pairs=lst[0]["n_pairs"])
        for k in keys:
            vals = [d[k] for d in lst]
            row[f"{k}_mean"] = statistics.fmean(vals)
            row[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        rows.append(row)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'model':<12} {'n_pairs':>7} " + " ".join(f"{k[:22]:>22}" for k in keys))
    for row in rows:
        print(f"{row['model']:<12} {row['n_pairs']:>7} " + " ".join(f"{row[k + '_mean']:.3f} ± {row[k + '_std']:.3f}".rjust(22) for k in keys))
    print(f"written: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
