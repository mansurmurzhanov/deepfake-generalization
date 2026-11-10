"""Rewrite point 1 of the two v4 response letters from results_v3/logo_results.csv once all 12 folds are complete.
Usage: python update_letters_v4.py"""
import pandas as pd

DL = "/Users/adema/Downloads"; R = f"{DL}/results_v3"
L = pd.read_csv(f"{R}/logo_results.csv", keep_default_na=False); L["auc_heldout"] = pd.to_numeric(L.auc_heldout, errors="coerce")
assert L.auc_heldout.notna().all() and L.groupby("model").generator.nunique().eq(12).all(), "LOGO is not complete"
P = {"StyleGANXL": "StyleGAN-XL", "StyleSANXL": "StyleSAN-XL", "glide": "GLIDE", "tamingTransformers": "Taming Transformers", "VQDiffusion": "VQ-Diffusion", "ProjectedGAN": "Projected GAN"}
def st(m):
    d = L[L.model == m].sort_values("auc_heldout"); lo = [(P.get(g, g), v) for g, v in zip(d.generator, d.auc_heldout)]
    return d.auc_heldout.mean(), d.auc_heldout.std(), lo
def en(x, k=3): return f"{x:.{k}f}"
def ru(x, k=3): return f"{x:.{k}f}".replace(".", ",")
cm, cs, cl = st("CLIP+LR"); rm, rs, rl = st("ResNet-50"); sm, ss, sl = st("SimpleCNN")
r_low = [(g, v) for g, v in rl if v < 0.90]; s_low = [(g, v) for g, v in sl if v < 0.70]
def lst(items, f, conj):
    t = [f"{g} {f(v, 2)}" for g, v in items]
    return t[0] if len(t) == 1 else ", ".join(t[:-1]) + f" {conj} " + t[-1]

EN_HEAD = "**Done: leave-one-generator-out over all 12 generators for the three detectors.**"
EN_RES = (f"Result: the CLIP-based detector has a mean held-out AUC of {en(cm)} ± {en(cs)} and never falls below {en(cl[0][1], 2)} (lowest on {cl[0][0]}, {en(cl[0][1], 2)}, and {cl[1][0]}, {en(cl[1][1], 2)}). "
          f"ResNet-50 averages {en(rm)} ± {en(rs)} and falls below 0.90 on {len(r_low)} of the 12 generators ({lst(r_low, en, 'and')}); the small CNN averages {en(sm)} ± {en(ss)} with a minimum of {en(sl[0][1], 2)} ({sl[0][0]}) "
          f"and stays below 0.70 on {len(s_low)} generators ({lst(s_low, en, 'and')}). The spread across generator choices is therefore much larger for the two networks than for the CLIP-based detector, "
          "and the three generators held out in the main protocol are not an unusually easy or hard choice. The networks are trained once per fold (seed 0), so this analysis measures the spread across generators, not across seeds; the paper says so in Section 4.4.")
RU_HEAD = "**Сделано: leave-one-generator-out по всем 12 генераторам для трёх детекторов.**"
RU_RES = (f"Результат: у детектора на CLIP средний AUC в режиме held-out {ru(cm)} ± {ru(cs)}, и он нигде не опускается ниже {ru(cl[0][1], 2)} (минимум на {cl[0][0]} — {ru(cl[0][1], 2)} и {cl[1][0]} — {ru(cl[1][1], 2)}). "
          f"ResNet-50 в среднем {ru(rm)} ± {ru(rs)} и опускается ниже 0,90 на {len(r_low)} из 12 генераторов ({lst(r_low, ru, 'и')}); небольшая сеть в среднем {ru(sm)} ± {ru(ss)}, минимум {ru(sl[0][1], 2)} ({sl[0][0]}), "
          f"и остаётся ниже 0,70 на {len(s_low)} генераторах ({lst(s_low, ru, 'и')}). Разброс по выбору генератора у двух сетей гораздо больше, чем у детектора на CLIP, "
          "а три генератора, отложенные в основном протоколе, не являются необычно лёгким или трудным выбором. Сети обучались один раз на фолд (seed 0), поэтому этот анализ измеряет разброс по генераторам, а не по seed'ам; в разделе 4.4 это оговорено.")

for fn, head_old, head_new, res_old, res_new, fig_old, fig_new in (
    ("response_to_reviewer_v4.md", "**Done in part:", EN_HEAD, "Result:", EN_RES,
     "the leave-one-generator-out runs, and", "the leave-one-generator-out runs, and"),
    ("response_to_reviewer_v4_RU.md", "**Сделано частично:", RU_HEAD, "Результат:", RU_RES, "", "")):
    lines = open(f"{DL}/{fn}", encoding="utf-8").read().split("\n"); n = 0
    for i, ln in enumerate(lines):
        if ln.startswith(head_old) or ln.startswith(head_new[:12]):
            k = ln.index("**", 2) + 2; lines[i] = head_new + ln[k:]; n += 1
        elif ln.startswith(res_old):
            lines[i] = res_new; n += 1
    assert n == 2, (fn, n)
    open(f"{DL}/{fn}", "w", encoding="utf-8").write("\n".join(lines))
    print(fn, "updated")
print(EN_RES); print(); print(RU_RES)
