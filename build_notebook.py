"""Generate `notebooks/phishing_detection_colab.ipynb`.

The notebook is built from a Python list of cells so it stays valid JSON and is
easy to regenerate after editing. Run: python tools/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "phishing_detection_colab.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md(
        """
# AI-Based Phishing Email Detection Using NLP - reproducible experiment suite

This notebook runs the **corrected** pipeline and regenerates every number,
table and figure the paper needs. It replaces `code1.py`-`code5.py`, which fitted
TF-IDF on the full corpus before splitting (data leakage) and reported accuracy only.

**What it produces**

| Output | Where |
|---|---|
| Dataset statistics (measured) | `reports/dataset_statistics.md` |
| Complete metrics, ROC-AUC, FPR/FNR | `results/model_metrics.csv` |
| ROC curve figure | `results/roc_curve.png` |
| Confusion matrices (RF, LR) | `results/confusion_matrix_rf.png`, `..._lr.png` |
| Feature importance | `results/feature_importance.png` |
| Stratified 5-fold CV | `reports/cross_validation_report.md` |
| Ablation study | `results/ablation_results.csv` |
| Error analysis | `reports/error_analysis.md` |
| Inference cost | `results/deployment_metrics.csv` |
| Audit + reviewer assessment | `reports/initial_research_audit.md`, `reports/final_ieee_paper_audit.md` |

**Runtime on a free Colab CPU runtime:** roughly 5-15 minutes for CEAS-08,
depending on the number of rows and the Random Forest size.
"""
    ),
    md("## 1. Install dependencies"),
    code(
        """
# Colab already ships most of these; this pins nothing and upgrades nothing unnecessarily.
%pip install -q -r requirements.txt
"""
    ),
    md(
        """
## 2. Get the project code into the runtime

Pick **one** of these:

- **Option A (Drive):** upload the project folder to your Drive and mount it below.
- **Option B (GitHub):** clone your repository, then `%cd` into it.

The project root must contain the `src/` folder.
"""
    ),
    code(
        """
import os, sys, subprocess

# Option A: Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# PROJECT = '/content/drive/MyDrive/phishing-nlp'
# os.chdir(PROJECT)

# Option B: GitHub (edit the URL to your repository)
PROJECT = 'phishing-nlp'
if not os.path.isdir(PROJECT):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/<your-user>/<your-repo>.git', PROJECT], check=False)

if os.path.isdir(PROJECT):
    os.chdir(PROJECT)

print('working directory:', os.getcwd())
print('src/ found:', os.path.isdir('src'))
%pip install -q -r requirements.txt
"""
    ),
    md(
        """
## 3. Provide the CEAS-08 dataset

Either upload `CEAS_08.csv` (or `CEAS_08.csv.zip`) with the cell below, or place
it in `data/` so the runner finds it automatically.

The script verifies the real column names and the label encoding before it uses
anything. If the schema differs from the expectation, it stops instead of guessing.
"""
    ),
    code(
        """
import os
os.makedirs('data', exist_ok=True)

try:
    from google.colab import files
    print('Select CEAS_08.csv or CEAS_08.csv.zip ...')
    uploaded = files.upload()
    for name in uploaded:
        os.replace(name, os.path.join('data', name))
        print('stored data/' + name)
except ImportError:
    print('Not running in Colab; put the CSV/ZIP in data/ manually.')

print(sorted(os.listdir('data')))
"""
    ),
    md("## 4. Verify the schema before running experiments"),
    code(
        """
import pandas as pd
from src import data_io, config

df, csv_path = data_io.load_dataframe(None)      # first CSV/ZIP in data/
print('file:', csv_path.name, '| shape:', df.shape)
print('columns:', list(df.columns))

texts, labels, meta = data_io.prepare_dataset(df, csv_path)
print()
print('body column  :', meta['body_column'])
print('label column :', meta['label_column'])
print('label mapping:', meta['label_meta']['mapping'])
print('rows used    :', len(texts))
print()
print('class distribution:')
print(pd.Series(labels).value_counts().rename(index={0: 'legitimate', 1: 'phishing'}))
"""
    ),
    md(
        """
## 5. Run the full experiment suite

`--tag main` writes to `results/` and `reports/` (the paths the paper references).
Any other tag writes to `results/_<tag>/`, so a test run can never overwrite real results.

Flags: `--skip-ablation`, `--skip-cv`, `--skip-errors`, `--cv-folds N`.
"""
    ),
    code(
        """
!python -m src.run_experiments --data data --tag main
"""
    ),
    md("## 6. Inspect the headline table"),
    code(
        """
import pandas as pd
metrics = pd.read_csv('results/model_metrics.csv')
display(metrics.drop(columns=['support_phishing', 'support_legitimate']))

print('\\nAblation:')
display(pd.read_csv('results/ablation_results.csv'))
print('\\nCross-validation:')
display(pd.read_csv('results/cross_validation_summary.csv'))
print('\\nInference cost:')
display(pd.read_csv('results/deployment_metrics.csv'))
"""
    ),
    md("## 7. View the figures"),
    code(
        """
from IPython.display import Image, display

for name in ['roc_curve.png', 'confusion_matrix_rf.png',
             'confusion_matrix_lr.png', 'feature_importance.png']:
    print(name)
    display(Image(filename=f'results/{name}'))
"""
    ),
    md(
        """
## 8. Read the audit reports

Open these as Markdown. `initial_research_audit.md` documents the code defects and
how the draft's own numbers fail internal consistency checks.
`final_ieee_paper_audit.md` is the reviewer-style assessment.
"""
    ),
    code(
        """
from IPython.display import Markdown, display

for name in ['initial_research_audit.md',
             'experimental_results.md',
             'cross_validation_report.md',
             'error_analysis.md',
             'limitations_and_future_work.md',
             'final_ieee_paper_audit.md']:
    print('=' * 100)
    display(Markdown(f'## {name}'))
    display(Markdown(open(f'reports/{name}').read()))
"""
    ),
    md("## 9. Download everything as a zip"),
    code(
        """
import shutil
shutil.make_archive('phishing_experiment_outputs', 'zip', root_dir='.',
                    base_dir=None) if False else None

import os, zipfile
with zipfile.ZipFile('phishing_experiment_outputs.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder in ('results', 'reports'):
        for root, _, files in os.walk(folder):
            for name in files:
                path = os.path.join(root, name)
                zf.write(path)

print('wrote phishing_experiment_outputs.zip')
try:
    from google.colab import files as colab_files
    colab_files.download('phishing_experiment_outputs.zip')
except ImportError:
    pass
"""
    ),
    md(
        """
## 10. Before you write the paper

1. Use **only** numbers from `results/` and `reports/` produced by this run.
2. Quote the dataset size from `reports/dataset_statistics.md`, not from memory.
3. Do not mix the holdout table with the cross-validation table.
4. Do not claim cross-dataset generalisation or statistical significance.
5. Verify every reference has a resolvable DOI before submission.
"""
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {TARGET} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
