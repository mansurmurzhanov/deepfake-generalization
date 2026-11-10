"""Finalize draft v4 after the leave-one-generator-out run: build docx + PDF from results_v3, fill the two letters,
update the repository working copy. Usage: python finalize_v4.py"""
import json, os, re, shutil, subprocess, sys
import pandas as pd

DL = "/Users/adema/Downloads"; R = f"{DL}/results_v3"; URL = "https://github.com/a33558771-prog/deepfake-generalization"
BUILD = f"{DL}/v3_build"; SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
assert os.path.exists(f"{R}/logo_summary.csv") and os.path.exists(f"{R}/logo_results.csv"), "LOGO not finished"

# 1. docx + PDF
out = f"{DL}/Akhmetova_Murzhanov_ELCVIA_draft_v4.docx"
subprocess.run([sys.executable, "-I", f"{DL}/build_v3_docx.py", "--v2", f"{DL}/Akhmetova_Murzhanov_ELCVIA_draft_v2_1.docx", "--res", R, "--out", out, "--github", URL], check=True)
shutil.copy(out, BUILD)
subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", BUILD, f"{BUILD}/Akhmetova_Murzhanov_ELCVIA_draft_v4.docx"], check=True, capture_output=True)
shutil.copy(f"{BUILD}/Akhmetova_Murzhanov_ELCVIA_draft_v4.pdf", DL)
import fitz
doc = fitz.open(f"{DL}/Akhmetova_Murzhanov_ELCVIA_draft_v4.pdf"); pages = len(doc)
text = "".join(p.get_text() for p in doc)
cited = {int(n) for m in re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text.split("References")[0]) for n in m.split(",")}
print("pages:", pages, "| uncited refs:", [n for n in range(1, 40) if n not in cited], "| placeholders:", text.count("[GitHub"))

# 2. letters: response_to_reviewer_v4(.md/_RU.md) are edited by hand (LOGO summary depends on which folds were completed)

# 3. repository working copy
REPO = f"{DL}/deepfake-generalization"
for f in ("run_v2_local.py", "run_logo.py", "check_overlap.py", "paired_analysis.py", "make_fig_errors.py", "build_v3_docx.py", "finalize_v4.py", "logo_collect.py"):
    shutil.copy(f"{DL}/{f}", f"{REPO}/scripts/")
if os.path.exists(f"{REPO}/results_v3"): shutil.rmtree(f"{REPO}/results_v3")
shutil.copytree(R, f"{REPO}/results_v3", ignore=shutil.ignore_patterns("clip_feats_all.npz", "*.log", "logo"))
os.makedirs(f"{REPO}/results_v3/logo", exist_ok=True)
for f in os.listdir(f"{R}/logo"):
    if f.endswith((".npz", ".json")) or f.startswith("trainlog"): shutil.copy(f"{R}/logo/{f}", f"{REPO}/results_v3/logo/")
os.makedirs(f"{REPO}/data/splits_v3", exist_ok=True)
for f in os.listdir(f"{R}/splits_v2"): shutil.copy(f"{R}/splits_v2/{f}", f"{REPO}/data/splits_v3/{f}")
shutil.copy(f"{DL}/Akhmetova_Murzhanov_ELCVIA_draft_v4.pdf", f"{REPO}/paper/")
for f in ("response_to_reviewer_v4.md",):
    shutil.copy(f"{DL}/{f}", f"{REPO}/paper/")
print("repository working copy updated")
