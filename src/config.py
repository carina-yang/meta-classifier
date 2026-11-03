"""
config.py
---------
Central configuration for the PII classification pipeline: paths, file locations,
and model/training hyperparameters.

This module provides:
- Base paths and directories (data/, models/, results/)
- Synthetic data generation parameters
- Fine-tuning hyperparameters (learning rate, epochs, batch size, LR schedule)
- XGBoost classifier parameters
- Multi-run support via run_id namespacing

Each training run gets a unique run_id (timestamp or custom --run-name) with
isolated folders under data/{run_id}/, models/{run_id}/, and results/{run_id}/.
Use the get_*() functions to access per-run paths.
"""

import os
from datetime import datetime

# ----------------------------------------------------------------------------
# 1. PATHS
# ----------------------------------------------------------------------------
# Base directory for all artifacts (data/, models/, results/)
# Override via PIPELINE_BASE_DIR environment variable (see COLAB.md for Colab setup)
BASE_DIR = os.environ.get("PIPELINE_BASE_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sentence transformer model (auto-downloaded from Hugging Face on first use)
# Override via MINILM_MODEL_PATH env var or --minilm-model-path flag
BASE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BASE_MODEL_PATH = os.environ.get("MINILM_MODEL_PATH", BASE_MODEL_NAME)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# 2. CLASSIFIER OPTIONS
# ----------------------------------------------------------------------------
# Available classifiers:
# - LinearSVC: Linear SVM with balanced class weights
# - XGBoost: Gradient boosting with multi-class softmax
# - ClassifierHead: Pre-trained head from fine-tuning (fine-tuned variants only)
#
# ClassifierHead is automatically added when fine-tuning MiniLM. It uses the
# classification head trained jointly with the transformer during fine-tuning,
# rather than training a separate classifier on top of the embeddings.

# ----------------------------------------------------------------------------
# 3. XGBOOST HYPERPARAMETERS
# ----------------------------------------------------------------------------
XGB_PARAMS = {
    "max_depth": 6,
    "eta": 0.3,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
}
XGB_NUM_ROUND = 100  # Boosting rounds

# ----------------------------------------------------------------------------
# 4. INPUT DATA COLUMNS
# ----------------------------------------------------------------------------
# Default input CSV from synthetic_data.py
# CSV must have exactly these column names (case-insensitive, after normalization)
RAW_CSV = os.path.join(BASE_DIR, "data", "synthetic_pii_columns.csv")
COL_TABLE_NAME = "table_name"
COL_NAME = "column_name"
COL_DATA_TYPE = "data_type"
COL_LABEL = "pii_class"  # Target label (see taxonomy.py)
FEATURE_COLUMNS = [COL_NAME, COL_TABLE_NAME, COL_DATA_TYPE]
TEXT_COLUMN = "all_text"
LABEL_ID_COLUMN = "label"

# ----------------------------------------------------------------------------
# 4b. SYNTHETIC DATA GENERATION
# ----------------------------------------------------------------------------
SYNTH_TOTAL_ROWS = 10_000          # Target total rows (split 70/15/15 → ~7k/1.5k/1.5k)
SYNTH_NON_PII_FRACTION = 0.30      # 30% non_pii, remaining 70% split across 9 PII classes
SYNTH_BATCH_SIZE = 40              # Items per LLM generation call
SYNTH_OVERSAMPLE = 1.4             # Initial overgeneration factor (adapts per round)
SYNTH_MAX_ROUNDS = 5               # Max generation rounds to reach quotas
SYNTH_MAX_PER_COLUMN_NAME = 6      # Cap repeats of same normalized column name
SYNTH_WORKERS = 6                  # Parallel worker threads
SYNTH_GEN_MODEL = os.environ.get("SYNTH_GEN_MODEL", "claude-haiku-5-5")      # Generator LLM
SYNTH_LABEL_MODEL = os.environ.get("SYNTH_LABEL_MODEL", "claude-sonnet-5-5")  # Labeler LLM

# ----------------------------------------------------------------------------
# 5. MULTI-RUN SUPPORT
# ----------------------------------------------------------------------------
# Each run gets a unique run_id (timestamp or custom --run-name)
# All outputs nested under data/{run_id}/, models/{run_id}/, results/{run_id}/
DEFAULT_RUN_ID = "default"


def make_run_id(run_name: str = None) -> str:
    """Create run_id: custom name if given, else timestamp (e.g., 'run_20260908_1543')."""
    if run_name:
        return run_name
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M")


def _run_has_existing_output(run_id: str) -> bool:
    return (
        os.path.isdir(get_data_dir(run_id))
        or os.path.isdir(get_results_dir(run_id))
        or os.path.isdir(os.path.join(BASE_DIR, "models", run_id))
    )


def resolve_run_dir(run_id: str) -> str:
    """Ensure run_id doesn't collide with existing output folders.

    If collision detected, appends numeric suffix (_2, _3, ...) rather than
    overwriting. Prevents accidental data loss from reused --run-name values.
    """
    if not _run_has_existing_output(run_id):
        return run_id
    suffix = 2
    candidate = f"{run_id}_{suffix}"
    while _run_has_existing_output(candidate):
        suffix += 1
        candidate = f"{run_id}_{suffix}"
    return candidate


def find_latest_run_id():
    """Find most recently created run_id under models/ by modification time.

    Returns None if no runs exist. Used as default when --run-name omitted.
    """
    models_root = os.path.join(BASE_DIR, "models")
    if not os.path.isdir(models_root):
        return None
    candidates = []
    for name in os.listdir(models_root):
        full_path = os.path.join(models_root, name)
        # Exclude stock base model folder
        if os.path.isdir(full_path) and not name.startswith("all-MiniLM"):
            candidates.append((os.path.getmtime(full_path), name))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


# Per-run path functions
def get_data_dir(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(BASE_DIR, "data", run_id)


def get_train_csv(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(get_data_dir(run_id), "train.csv")


def get_test_csv(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(get_data_dir(run_id), "test.csv")


def get_eval_csv(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(get_data_dir(run_id), "eval.csv")


def get_label_map_json(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(get_data_dir(run_id), "label_mapping.json")


def get_results_dir(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(BASE_DIR, "results", run_id)


def get_run_config_json(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(get_results_dir(run_id), "run_config.json")


def get_epoch_accuracy_csv(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(get_results_dir(run_id), "epoch_accuracy_comparison.csv")


def get_ft_checkpoint_dir_template(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(BASE_DIR, "models", run_id, "ft_epoch_{epoch}")


def get_ft_model_final(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(BASE_DIR, "models", run_id, "ft_final")


def get_ft_loss_history_json(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(BASE_DIR, "models", run_id, "ft_loss_history.json")


def get_classifiers_dir(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(BASE_DIR, "models", run_id, "classifiers")


def get_features_cache_dir(run_id: str = DEFAULT_RUN_ID) -> str:
    return os.path.join(BASE_DIR, "models", run_id, "feature_cache")


# ----------------------------------------------------------------------------
# 6. DATA SPLITTING AND EMBEDDING
# ----------------------------------------------------------------------------
# Split ratios: train for training, test for model selection, eval for final validation
TRAIN_RATIO = 0.70
TEST_RATIO = 0.15
EVAL_RATIO = 0.15

# Minimum samples required to split class across all three sets
# Classes below threshold stay in training set only
MIN_SAMPLES_PER_CLASS = 3

# Batch size for embedding text with MiniLM (stock or fine-tuned)
MINILM_BATCH_SIZE = 64

# ----------------------------------------------------------------------------
# 7. FINE-TUNING HYPERPARAMETERS
# ----------------------------------------------------------------------------
FT_NUM_EPOCHS = 5             # Epochs for fine-tuning MiniLM
FT_BATCH_SIZE = 32            # Training batch size (increase on GPU: see COLAB.md)
FT_LEARNING_RATE = 5e-5       # Peak LR (reached after warmup, then decays)
FT_WARMUP_RATIO = 0.1         # Fraction of steps for warmup (0 → peak LR)
FT_LR_SCHEDULER = "linear"    # Decay schedule: "linear" or "cosine"

