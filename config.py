"""Central configuration for the phishing-email detection experiments.

Every value that the paper reports must be traceable to a constant here or to a
value computed at runtime from the dataset. Nothing in this module encodes a
result.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Evaluation protocol -----------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --- Column resolution -------------------------------------------------------
# CEAS_08 (Kaggle) is expected to expose these headers; the loader verifies
# them against the real file and fails loudly rather than guessing.
BODY_COLUMN_CANDIDATES = ("body", "email_body", "text", "message", "content")
LABEL_COLUMN_CANDIDATES = ("label", "class", "target", "is_phishing", "phishing")
SUBJECT_COLUMN_CANDIDATES = ("subject",)
URL_COLUMN_CANDIDATES = ("urls", "url")

# --- Label convention --------------------------------------------------------
# CEAS_08 convention: 0 = legitimate, 1 = phishing.
# The runner checks this against the data and fails if the class you call
# "phishing" is not present.
POSITIVE_LABEL = 1
LEGITIMATE_LABEL = 0

# --- Model definitions (defaults, not tuned) ---------------------------------
RF_ESTIMATORS = 100
RF_ESTIMATORS_QUICK = 25          # smoke-test only; never reported
LR_MAX_ITER = 1000

# --- Figure settings ---------------------------------------------------------
DPI = 300
FIG_DPI = 150                      # on-screen only
TOP_FEATURES = 20

# --- Historical numbers printed in the current draft -------------------------
# Kept ONLY so the audit can check the draft against itself. These are not
# results of this pipeline and must never be copied into a results table.
DRAFT_CLAIMS = {
    "reported_total_emails": 18000,
    "reported_test_partition": 7831,
    "reported_test_legitimate": 3456,
    "reported_test_phishing": 4375,
    "reported_rf_accuracy": 0.9927,
    "reported_rf_precision": 0.9954,
    "reported_rf_recall": 0.9915,
    "reported_lr_accuracy": 0.9923,
    "reported_rf_false_positives": 20,
    "reported_rf_false_negatives": 15,
    "reported_majority_baseline": 0.56,
}
