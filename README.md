# PII Column Classifier

> **Automated PII detection for database schemas using machine learning**

A production-ready pipeline that classifies database columns into a comprehensive 10-class PII taxonomy with **98.34% accuracy**, using only metadata (`table_name`, `column_name`, `data_type`).

This project demonstrates that classical TF-IDF features outperform fine-tuned transformer models for structured data classification, achieving state-of-the-art results with efficient CPU-only inference.

## Key Results

- **Best Model:** TF-IDF Combined + LinearSVC → **98.34% accuracy**
- **Fine-tuned Transformers:** BGE-Small + ClassifierHead/XGBoost → 96.67% accuracy
- **Inference Speed:** 1.7ms per column (4,468 columns/sec throughput)
- **Deployment:** Lightweight (3.6 MB), CPU-only, calibrated probabilities

See [MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md) for practitioner guidance and [ML_SCIENTIFIC_ANALYSIS.md](ML_ANALYSIS.md) for research-oriented analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Module Reference](#module-reference)
- [Evaluation Methods](#evaluation-methods)
- [Review Scripts](#review-scripts)
- [Colab + GitHub](#colab--github)
- [Caveats](#caveats)

## Quick Start

### Prerequisites

- Python 3.8+
- Anthropic API key ([get one here](https://console.anthropic.com/))
- 8GB+ RAM (16GB recommended for fine-tuning)
- Optional: CUDA-compatible GPU for faster transformer fine-tuning

### Installation

```bash
# Clone the repository
git clone https://github.com/carina-yang/pii-column-classifier.git
cd pii-column-classifier

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_api_key_here
```

### Local Run
```bash
# 1. Generate synthetic training data (10,000 rows)
python src/synthetic_data.py

# 2. Train and compare models
python src/train_models.py --run-name v3_baseline

# 3. Evaluate checkpoint performance
python src/evaluate_checkpoints.py --run-name v3_baseline

# 4. Evaluate with probability calibration
python src/evaluate_probabilities.py --run-name v3_baseline
```

### Offline Test (no API, no Hugging Face)
```bash
python src/synthetic_data.py --total 1500 --mock
python src/train_models.py --run-name test --skip-minilm
```

Models: generator `claude-haiku-5-5`, labeler `claude-sonnet-5-5` (override via `SYNTH_GEN_MODEL` / `SYNTH_LABEL_MODEL`).

## Pipeline Architecture

The pipeline follows a modular design with each step building on the previous:

```
1. Taxonomy Definition (taxonomy.py)
   ↓
2. Data Generation (synthetic_data.py)
   ↓
3. Data Cleaning (data_cleaning.py)
   ↓
4. Text Processing (text_processing.py)
   ↓
5. Group-Aware Splitting (data_splitting.py)
   ↓
6. Label Encoding (label_encoding.py)
   ↓
7. Feature Extraction (feature_extractors.py)
   ├─ TF-IDF (char/word/combined)
   └─ BGE-Small Embeddings (stock + fine-tuned)
   ↓
8. Model Training (model_training.py)
   ├─ LinearSVC
   ├─ XGBoost
   └─ ClassifierHead (fine-tuned variants only)
   ↓
9. Evaluation
   ├─ Checkpoint Evaluation (evaluate_checkpoints.py)
   └─ Probability Evaluation (evaluate_probabilities.py)
```

## Module Reference

### Core Pipeline Modules

#### `src/taxonomy.py`
**Purpose:** Single source of truth for the 10-class PII taxonomy.

**Classes:**
- `person_name`: Individual's name (first, last, full name, etc.)
- `email`: Email addresses
- `phone`: Telephone numbers
- `physical_address`: Street addresses and location components
- `government_id`: Government-issued identifiers (SSN, passport, etc.)
- `date_of_birth`: Birth dates and birth years
- `financial_account`: Financial account identifiers and payment cards
- `online_identifier`: Online handles and device identifiers (IP, username, etc.)
- `demographic_attribute`: Demographic characteristics (gender, ethnicity, age, etc.)
- `non_pii`: Everything else

**Key Functions:**
- `render_taxonomy()`: Formats taxonomy as markdown for prompts
- `nonpii_excludes(label)`: Returns hard-negative examples for a class
- `routed_excludes(label)`: Returns exclude examples with routing info

#### `src/config.py`
**Purpose:** Central configuration for paths, hyperparameters, and model settings.

**Key Settings:**
- `PIPELINE_BASE_DIR`: Base directory (set via env var for Colab)
- `BASE_MODEL_PATH`: Path to BGE-Small model (default: auto-download from HF)
- `SYNTH_TOTAL_ROWS`: Target synthetic data size (default: 10,000)
- `TRAIN_RATIO / TEST_RATIO / EVAL_RATIO`: Split ratios (70/15/15)
- `FT_NUM_EPOCHS`: Fine-tuning epochs (default: 5)
- `FT_LEARNING_RATE`: Peak learning rate (default: 5e-5)
- `XGB_PARAMS`: XGBoost hyperparameters

**Key Functions:**
- `make_run_id(run_name)`: Creates or resolves run identifier
- `get_*_csv(run_id)`: Per-run path accessors for splits
- `get_ft_model_final(run_id)`: Path to final fine-tuned model
- `find_latest_run_id()`: Auto-discovers most recent run

#### `src/synthetic_data.py`
**Purpose:** Generates synthetic training data using dual-LLM approach.

**Process:**
1. Generator LLM (claude-haiku-5-5) creates column metadata in various modes:
   - Typical columns
   - Abbreviated/legacy naming
   - Ambiguous table contexts
   - Hard negatives
   - Generic non-PII
2. Labeler LLM (claude-sonnet-5-5) labels them **blind** using taxonomy
3. Only rows where both LLMs agree are kept
4. Disagreements saved to `data/synthetic_pii_columns_disagreements.csv`

**Usage:**
```bash
python src/synthetic_data.py                    # 10k rows
python src/synthetic_data.py --total 5000       # custom size
python src/synthetic_data.py --mock             # offline test mode
```

#### `src/data_cleaning.py`
**Purpose:** Loads and cleans the labeled CSV (synthetic or real).

**Required CSV Columns:**
- `column_name` (required)
- `pii_class` (required) - Target label from taxonomy
- `table_name` (optional) - Defaults to empty if missing
- `data_type` (optional) - Defaults to empty if missing

Column names are case-insensitive and spaces are converted to underscores.

**Cleaning Steps:**
1. Normalize CSV headers (lowercase, spaces to underscores)
2. Strip whitespace
3. Drop exact duplicates
4. Drop rows with empty column_name or label
5. Validate labels against taxonomy
6. Drop near-duplicates (same normalized form)
7. Drop conflicting rows (same table/column/type with different labels)

**Key Functions:**
- `load_raw_data(path)`: Loads CSV and normalizes headers
- `clean_dataframe(df)`: Applies all cleaning steps with logging

#### `src/text_processing.py`
**Purpose:** Converts (table_name, column_name, data_type) into normalized text.

**Transformations:**
- Lowercase conversion
- Letter/digit boundary splitting (`addr1` → `addr 1`)
- Schema prefix removal (`dbo.Customer` → `customer`)
- Data type normalization (`VARCHAR(255)` → `varchar 255`)
- Labeled field format: `"column: ... | table: ... | type: ..."`
- Missing values become `"unknown"`

**Key Functions:**
- `normalize_identifier(text)`: Converts identifiers to lowercase tokens
- `normalize_table_name(text)`: Removes schema prefixes
- `normalize_data_type(text)`: Normalizes data type strings
- `build_all_text(row)`: Builds the final labeled text string

#### `src/data_splitting.py`
**Purpose:** Group-aware train/test/eval split (70/15/15).

**Why Group-Aware:**
Column names repeat heavily (`email` in 40 tables). Random row split would put the same column name in train AND test, inflating accuracy. Group-aware splitting ensures each normalized column name appears in exactly ONE split.

**Algorithm:**
1. Normalize column names to create groups
2. Assign groups greedily (largest first) to maintain class balance
3. Keeps per-class proportions close to target ratios

**Key Functions:**
- `group_key(column_name)`: Creates normalized group key
- `split_data(df, train_ratio, test_ratio, eval_ratio, min_samples_per_class, seed)`: Returns train/test/eval DataFrames

#### `src/label_encoding.py`
**Purpose:** Converts string labels to stable integer IDs.

**Key Functions:**
- `encode_labels(df, labels_list)`: Adds `label` column with integer IDs
- Creates label mapping JSON for decoding predictions

#### `src/feature_extractors.py`
**Purpose:** Feature representations for model training.

**TF-IDF Variants:**
- `tfidf_char`: Character n-grams (2-5), robust to typos/abbreviations
- `tfidf_word`: Word n-grams (1-2), captures semantic tokens
- `tfidf_combined`: Both feature sets concatenated

**BGE-Small Embeddings:**
- `MiniLMEmbedder`: Wrapper for sentence-transformers (works with BGE and other models)
  - Works with stock OR fine-tuned models (same code, different weights)
  - Batch encoding for efficiency
  - GPU auto-detection

**Key Functions:**
- `make_tfidf_variants()`: Returns dict of TF-IDF extractors
- `MiniLMEmbedder.fit_transform(texts)`: Fits and transforms training data
- `MiniLMEmbedder.transform(texts)`: Transforms new data

#### `src/sentence_transformer_finetuning.py`
**Purpose:** Fine-tunes BGE-Small with classification head.

**Process:**
1. Loads base model (`BAAI/bge-small-en-v1.5`)
2. Attaches linear classification head
3. Fine-tunes with cross-entropy loss + AdamW
4. Per-epoch validation, keeps best epoch
5. Saves fine-tuned model AND classification head

**Features:**
- GPU auto-detection
- Warmup + decay LR schedule (linear or cosine)
- Per-epoch validation accuracy tracking
- Best model selection based on validation

**Key Functions:**
- `finetune_sentence_transformer(train_texts, train_labels, val_texts, val_labels, ...)`
- `select_best_checkpoint(checkpoint_template, num_epochs, ...)`: Picks best epoch

#### `src/model_training.py`
**Purpose:** Generic train/predict/save logic for all feature × classifier combinations.

**Classifiers:**
- **LinearSVC**: Fast linear classifier with class weighting
- **XGBoost**: Gradient boosting with multi-class softmax
- **ClassifierHead**: Pre-trained head from fine-tuning (fine-tuned variants only)

**Key Functions:**
- `train_predict_svm(x_train, y_train, x_test, x_eval, seed)`: Trains LinearSVC
- `train_predict_xgboost(x_train, y_train, x_test, x_eval, num_classes, seed)`: Trains XGBoost
- `train_predict_classifier_head(embedder, head, x_train, y_train, x_test, x_eval)`: Uses pre-trained head
- `run_feature_variant(feature_name, extractor, train_df, test_df, eval_df, ...)`: Complete training loop

#### `src/train_models.py`
**Purpose:** Main training pipeline that compares all model variants.

**What It Does:**
1. Loads and cleans data
2. Splits into train/test/eval
3. Processes text
4. Encodes labels
5. Optionally fine-tunes BGE-Small
6. Trains all variants:
   - TF-IDF char/word/combined × LinearSVC/XGBoost
   - BGE-Small stock × LinearSVC/XGBoost
   - BGE-Small fine-tuned × LinearSVC/XGBoost/ClassifierHead
7. Saves results and models

**Usage:**
```bash
python src/train_models.py --run-name my_run
python src/train_models.py --run-name test --skip-minilm  # TF-IDF only (note: flag name kept for compatibility)
python src/train_models.py --raw-csv path/to/custom.csv
```

## Evaluation Methods

### Checkpoint Evaluation (`src/evaluate_checkpoints.py`)

**Purpose:** Empirically checks if more fine-tuning epochs would help by re-embedding existing splits with each saved epoch checkpoint.

**Process:**
1. Loads existing train/test/eval splits from a run
2. For each saved epoch checkpoint:
   - Re-embeds data with that checkpoint
   - Trains LinearSVC/XGBoost on top
   - Records accuracy
3. No additional fine-tuning required
4. ClassifierHead is skipped (tied to final model only)

**Usage:**
```bash
# Evaluate explicitly named run
python src/evaluate_checkpoints.py --run-name v3_baseline

# Use latest run automatically
python src/evaluate_checkpoints.py

# TF-IDF only
python src/evaluate_checkpoints.py --skip-minilm
```

**Outputs:**
- Results written to `results/{run_id}/epoch_accuracy_comparison.csv`

### Probability Evaluation (`src/evaluate_probabilities.py`)

**Purpose:** Re-evaluates an existing run with probability predictions and temperature calibration.

**Process:**
1. Loads existing run's splits and label mapping
2. Loads stock BGE-Small and fine-tuned models (no network access needed)
3. Loads all classifiers (LinearSVC, XGBoost, ClassifierHead)
4. Fits temperature calibration on **TEST** split
5. Reports calibrated vs uncalibrated metrics on **EVAL** split

**Why Calibrate on Test, Report on Eval:**
Methodology intentionally separates calibration fitting (test) from final evaluation (eval) to avoid overfitting the calibration temperature.

**Usage:**
```bash
# Evaluate explicitly named run
python src/evaluate_probabilities.py --run-name v3_baseline

# Use latest run automatically
python src/evaluate_probabilities.py

# TF-IDF only
python src/evaluate_probabilities.py --run-name baseline --skip-minilm

# Skip XGBoost, no temperature calibration
python src/evaluate_probabilities.py --run-name baseline --skip-gbt --calibration none

# Add infrastructure cost estimate
python src/evaluate_probabilities.py --run-name baseline --hourly-rate-usd 2.50
```

**Outputs:**
- Probability matrices: `results/{run_id}/probabilities/`
- Fitted artifacts: `models/{run_id}/prob_eval/`
- Summary report: `results/{run_id}/probabilities/slide_ready_summary.md`
  - Sorted by EVAL accuracy
  - Includes macro-F1 across all classes and across classes with support > 0

## Review Scripts

### Overview

The `scripts/review/` directory contains tools for reviewing and resolving disagreements between the labeling model and intended classes during synthetic data generation. When the dual-LLM approach produces disagreements, these scripts help systematically review and correct them.

**Use Case:** During synthetic data generation, the labeler LLM may disagree with the generator's intended class. These scripts provide:
- Automated rule-based resolution for clear cases
- Interactive review interface for ambiguous cases
- Merged dataset with human-verified labels

### Quick Start

```bash
# 1. List available review categories
python scripts/review/review_disagreements.py --list-categories

# 2. Review clear-cut cases in batch mode (fastest)
python scripts/review/review_disagreements.py --category clear_cut_cases --batch

# 3. Review ambiguous cases interactively
python scripts/review/review_disagreements.py --category ambiguous_needs_context

# 4. Merge all reviews into final dataset
python scripts/review/merge_reviews.py
```

### Available Scripts

#### `review_disagreements.py` - Interactive Review Tool

Review model-human disagreements through an interactive CLI.

**Features:**
- 4 pre-categorized review groups (clear-cut, high-confidence right/wrong, ambiguous)
- Interactive mode with taxonomy reference
- Batch mode for bulk decisions
- Progress saving (quit anytime with `q`)

**Usage:**
```bash
# Interactive review
python scripts/review/review_disagreements.py --category high_confidence_model_likely_right

# Batch review (accept all intended or all model predictions)
python scripts/review/review_disagreements.py --category clear_cut_cases --batch
```

**Interactive Commands:**
- `1-10`: Select class by number
- `i`: Accept intended class
- `m`: Accept model's predicted class
- `?`: Show taxonomy definitions
- `s`: Skip record
- `q`: Quit and save progress

**Output:** Creates `data/reviewed_<category>.csv` with human labels and review decisions.

#### `auto_review.py` - Automated Review Engine

Automatically resolves disagreements using taxonomy rules and pattern matching.

**Features:**
- Birth date detection (`dob`, `birth`, `birthday`)
- Session identifier detection
- Account number patterns
- Segment vs. individual demographic distinction
- Confidence scoring for flagging uncertain cases

**Usage:**
```bash
python scripts/review/auto_review.py
```

**Output:** Auto-resolved labels with confidence annotations.

#### `manual_final_review.py` - Manual Decision Repository

Contains codified manual decisions for 65 edge cases with explanations.

**Format:**
```python
(table_name, column_name): (decision, confidence, reason)
```

**Example:**
```python
('CustomerPrefs', 'DobFlg'):
    ('non_pii', 'high', 'Flag/indicator about DOB, not the DOB value itself')
```

**Categories Covered:**
- Metadata flags vs. actual PII values
- Segment-level demographics vs. individual data
- Ambiguous name fields
- Context-dependent identifiers

#### `merge_reviews.py` - Review Merger

Combines all reviewed labels into a final training dataset.

**Process:**
1. Loads all `reviewed_*.csv` files from `data/`
2. Merges with original disagreements
3. Tracks resolution decisions (agreed_with_model, agreed_with_intended, manual_override)
4. Outputs statistics

**Usage:**
```bash
python scripts/review/merge_reviews.py
```

**Output:** `data/synthetic_pii_columns_disagreements_resolved.csv`

**Example Output:**
```
Found 4 review file(s)
Total reviewed records: 197

Resolution breakdown:
  agreed_with_intended          :   130 ( 66.0%)
  agreed_with_model             :    52 ( 26.4%)
  manual_override               :    15 (  7.6%)

Total disagreements: 197
Resolved by human review: 197 (100.0%)
```

### Review Workflow

Complete workflow for reviewing synthetic data disagreements:

```bash
# Step 1: Generate synthetic data (produces disagreements file)
python src/synthetic_data.py

# Step 2: Review disagreements by category (start with easiest)
python scripts/review/review_disagreements.py --category clear_cut_cases --batch
python scripts/review/review_disagreements.py --category high_confidence_model_likely_right
python scripts/review/review_disagreements.py --category high_confidence_model_likely_wrong
python scripts/review/review_disagreements.py --category ambiguous_needs_context

# Step 3: Merge all reviews
python scripts/review/merge_reviews.py

# Step 4: (Optional) Retrain with corrected labels
# Update synthetic_data.py to incorporate human-reviewed labels
python src/train_models.py --run-name human_verified_v1
```

### Review Guidelines

**Edge Case Decision Tree:**

1. **Metadata vs. Actual PII:**
   - `dob_flag` → **non_pii** (indicator about DOB)
   - `date_of_birth` → **date_of_birth** (actual value)

2. **Individual vs. Aggregate:**
   - `customer_gender` → **demographic_attribute** (individual)
   - `segment_target_gender` → **non_pii** (aggregate metadata)

3. **System vs. Personal Identifier:**
   - `user_id` (internal key) → **non_pii**
   - `username` (user-chosen) → **online_identifier**
   - `session_id` → **online_identifier** (tracks behavior)

4. **Public vs. Personal:**
   - `company_name` → **non_pii** (business entity)
   - `owner_name` → **person_name** (individual)

### Files Generated

**Input:**
- `data/synthetic_pii_columns_disagreements.csv` - Original disagreements from synthetic_data.py
- `data/review_categories.json` - Categorized disagreements

**Output:**
- `data/reviewed_<category>.csv` - Human-reviewed labels per category
- `data/synthetic_pii_columns_disagreements_resolved.csv` - Final merged dataset

### Documentation

For detailed usage instructions, decision guidelines, and troubleshooting, see:
- **`scripts/review/README.md`** - Comprehensive guide with examples, best practices, and API reference

## Running on Google Colab

See **[COLAB.md](COLAB.md)** for instructions on running the pipeline on Google Colab with GPU acceleration. The `colab_pipeline.ipynb` notebook clones the repo, stores artifacts on Google Drive, and optionally uploads models to Hugging Face.

## Caveats

- **Synthetic data limitation:** Accuracy on synthetic data (98.34%) is optimistic. For production deployment, validate on real database schemas.
- **Health information excluded:** Medical/health PII is out of scope per the current taxonomy.
- **Column names only:** This classifier uses metadata only. For complete PII detection, combine with content analysis.
- **English language:** Optimized for English database naming conventions.

## Citation

If you use this work in your research or project, please cite:

```bibtex
@software{pii_column_classifier,
  author = {Yang, Carina},
  title = {PII Column Classifier: Automated PII Detection for Database Schemas},
  year = {2026},
  url = {https://github.com/carina-yang/pii-column-classifier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [sentence-transformers](https://www.sbert.net/)
- Synthetic data generated using [Anthropic's Claude API](https://www.anthropic.com/)
- Inspired by privacy engineering and data governance best practices
