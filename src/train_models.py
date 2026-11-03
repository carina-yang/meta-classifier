"""
train_models.py
---------------------------
Main pipeline entry point. Orchestrates the complete training and comparison
workflow for PII column classification.

Produces a three-way feature comparison:
(a) TF-IDF (char/word/combined n-grams) × LinearSVC/XGBoost
(b) Stock BGE-Small-EN-V1.5 embeddings × LinearSVC/XGBoost
(c) Fine-tuned BGE-Small-EN-V1.5 embeddings × LinearSVC/XGBoost/ClassifierHead

Pipeline steps:
1. Data cleaning (data_cleaning.py) - Load CSV, dedupe, validate
2. Text processing (text_processing.py) - Normalize to all_text field
3. Label encoding (label_encoding.py) - Map labels to stable integer IDs
4. Group-aware splitting (data_splitting.py) - 70/15/15 train/test/eval
5. Feature extraction (feature_extractors.py) - TF-IDF variants + embeddings
6. Fine-tuning (sentence_transformer_finetuning.py) - Optional BGE-Small tuning
7. Model training (model_training.py) - Train classifiers and evaluate

Multi-run support:
Each run gets a unique run_id (timestamp or custom --run-name) with isolated
folders: data/{run_id}/, models/{run_id}/, results/{run_id}/. Run collisions
are automatically resolved with numeric suffixes.

Usage:
    python src/train_models.py --raw-csv data/synthetic_pii_columns.csv --run-name baseline
Optional:
    python src/train_models.py --skip-gbt
    python src/train_models.py --skip-minilm          # skip BOTH stock and fine-tuned (flag name kept for compatibility)
    python src/train_models.py --skip-finetune        # stock only, no fine-tuning
    python src/train_models.py --finetune-epochs 3 --finetune-batch-size 32
    python src/train_models.py --finetune-warmup-ratio 0.1 --finetune-lr-scheduler cosine
    python src/train_models.py --train-ratio 0.7 --test-ratio 0.15 --eval-ratio 0.15
    python src/train_models.py --min-samples-per-class 5

Outputs (all nested under the run's run_id -- see above):
    - results/{run_id}/classification_report_<variant>_<classifier>.txt
    - results/{run_id}/confusion_matrix_<variant>_<classifier>.csv       (test + eval, in one file)
    - results/{run_id}/accuracy_comparison_all_configurations.csv
    - results/{run_id}/run_config.json                                   (exact hyperparameters used)
    - data/{run_id}/train.csv / test.csv / eval.csv
    - data/{run_id}/label_mapping.json
    - models/{run_id}/ft_epoch_<N>/    (checkpoint after each fine-tuning epoch)
    - models/{run_id}/ft_final/        (final fine-tuned sentence transformer)
    - models/{run_id}/ft_head/         (classification head from fine-tuning)
    - models/{run_id}/ft_loss_history.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from config import (  # noqa: E402
    BASE_MODEL_PATH,
    COL_DATA_TYPE,
    COL_LABEL,
    COL_NAME,
    COL_TABLE_NAME,
    EVAL_RATIO,
    FT_BATCH_SIZE,
    FT_LEARNING_RATE,
    FT_LR_SCHEDULER,
    FT_NUM_EPOCHS,
    FT_WARMUP_RATIO,
    MIN_SAMPLES_PER_CLASS,
    MINILM_BATCH_SIZE,
    RAW_CSV,
    TEST_RATIO,
    TRAIN_RATIO,
    LABEL_ID_COLUMN,
    RANDOM_SEED,
    TEXT_COLUMN,
)
from data_cleaning import clean_dataframe, load_raw_data  # noqa: E402
from data_splitting import split_data  # noqa: E402
from feature_extractors import MiniLMEmbedder, make_tfidf_variants  # noqa: E402
from label_encoding import encode_labels  # noqa: E402
from model_training import run_feature_variant  # noqa: E402
from text_processing import build_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare PII column classifiers: "
                    "TF-IDF vs stock BGE-Small vs fine-tuned BGE-Small, "
                    "with LinearSVC/XGBoost/ClassifierHead classifiers."
    )
    parser.add_argument("--raw-csv", type=str, default=RAW_CSV,
                        help=f"Path to raw historical CSV. Default: {RAW_CSV}")
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Custom name for this run's output folder, e.g. 'lr_schedule_v1'. "
             "If omitted, an auto-generated timestamp id (run_YYYYMMDD_HHMM) is used. "
             "If the resulting run folder already has output on disk, a numeric "
             "suffix (_2, _3, ...) is appended automatically -- previous runs are "
             "never overwritten.",
    )
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument("--min-samples-per-class", type=int, default=MIN_SAMPLES_PER_CLASS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-gbt", action="store_true", help="Skip XGBoost, run only LinearSVC.")
    parser.add_argument("--skip-minilm", action="store_true",
                        help="Skip BOTH the stock and fine-tuned embeddings (TF-IDF only). Flag name kept for compatibility.")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Still run stock embeddings, but skip fine-tuning.")
    parser.add_argument("--minilm-model-path", type=str, default=BASE_MODEL_PATH,
                        help="Hugging Face model id or local path (default: BAAI/bge-small-en-v1.5). Flag name kept for compatibility.")
    parser.add_argument("--minilm-batch-size", type=int, default=MINILM_BATCH_SIZE,
                        help="Batch size for encoding text with embeddings. Flag name kept for compatibility.")
    parser.add_argument("--finetune-epochs", type=int, default=FT_NUM_EPOCHS,
                        help="Number of fine-tuning epochs. Lower for larger datasets -- see config.py notes.")
    parser.add_argument("--finetune-batch-size", type=int, default=FT_BATCH_SIZE,
                        help="Batch size used during fine-TUNING (separate from --minilm-batch-size, which is for encoding).")
    parser.add_argument("--finetune-learning-rate", type=float, default=FT_LEARNING_RATE,
                        help="PEAK learning rate, reached at the end of warmup (not a constant rate -- see README).")
    parser.add_argument("--finetune-warmup-ratio", type=float, default=FT_WARMUP_RATIO,
                        help="Fraction of total fine-tuning steps spent ramping LR 0 -> peak (default 0.1 = 10%%).")
    parser.add_argument("--finetune-lr-scheduler", type=str, default=FT_LR_SCHEDULER,
                        choices=["linear", "cosine"],
                        help="Decay shape applied to the LR after warmup (default: linear, decays to ~0).")
    parser.add_argument(
        "--xgb-dense-max-mb", type=int, default=512,
        help="Safety guard: skip XGBoost for a feature variant if its dense-equivalent "
             "size would exceed this many MB. Default: 512 MB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve run ID and output paths
    requested_run_id = config.make_run_id(args.run_name)
    run_id = config.resolve_run_dir(requested_run_id)
    if run_id != requested_run_id:
        print(
            f"NOTE: run '{requested_run_id}' already has output on disk; using "
            f"'{run_id}' instead so the previous run's results aren't overwritten."
        )
    print(f"\n{'=' * 70}\nRUN ID: {run_id}\n{'=' * 70}")

    data_dir = config.get_data_dir(run_id)
    results_dir = config.get_results_dir(run_id)
    train_csv = config.get_train_csv(run_id)
    test_csv = config.get_test_csv(run_id)
    eval_csv = config.get_eval_csv(run_id)
    label_map_json = config.get_label_map_json(run_id)
    ft_checkpoint_dir_template = config.get_ft_checkpoint_dir_template(run_id)
    ft_model_final = config.get_ft_model_final(run_id)
    ft_loss_history_json = config.get_ft_loss_history_json(run_id)
    run_config_json = config.get_run_config_json(run_id)
    classifiers_dir = config.get_classifiers_dir(run_id)
    cache_dir = config.get_features_cache_dir(run_id)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(classifiers_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Save run configuration for reproducibility
    run_config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_csv": args.raw_csv,
        "train_ratio": args.train_ratio,
        "test_ratio": args.test_ratio,
        "eval_ratio": args.eval_ratio,
        "min_samples_per_class": args.min_samples_per_class,
        "random_seed": args.random_seed,
        "skip_gbt": args.skip_gbt,
        "skip_minilm": args.skip_minilm,
        "skip_finetune": args.skip_finetune,
        "minilm_batch_size": args.minilm_batch_size,
        "finetune_epochs": args.finetune_epochs,
        "finetune_batch_size": args.finetune_batch_size,
        "finetune_learning_rate": args.finetune_learning_rate,
        "finetune_warmup_ratio": args.finetune_warmup_ratio,
        "finetune_lr_scheduler": args.finetune_lr_scheduler,
        "xgb_dense_max_mb": args.xgb_dense_max_mb,
    }
    with open(run_config_json, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved run config: {run_config_json}")

    print("\n=== Step 1: Load raw data ===")
    raw_df = load_raw_data(args.raw_csv)
    print(f"Raw rows loaded: {len(raw_df)}")

    print("\n=== Step 2: Clean (dedupe, drop unusable rows) ===")
    clean_df = clean_dataframe(raw_df)
    if clean_df.empty:
        raise ValueError("No rows remain after cleaning -- check your raw CSV.")

    print("\n=== Step 3: Build all_text + encode labels ===")
    clean_df[TEXT_COLUMN] = build_text(clean_df)
    clean_df = clean_df[clean_df[TEXT_COLUMN].str.len() > 0].reset_index(drop=True)
    clean_df, label_mapping, idx_to_tag = encode_labels(clean_df)
    with open(label_map_json, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Saved: {label_map_json}")

    print("\n=== Step 4: Split into train / test / eval ===")
    train_df, test_df, eval_df = split_data(
        clean_df, args.train_ratio, args.test_ratio, args.eval_ratio,
        args.min_samples_per_class, args.random_seed,
    )
    keep_cols = [COL_TABLE_NAME, COL_NAME, COL_DATA_TYPE, COL_LABEL,
                 TEXT_COLUMN, LABEL_ID_COLUMN]
    train_df[keep_cols].to_csv(train_csv, index=False)
    test_df[keep_cols].to_csv(test_csv, index=False)
    eval_df[keep_cols].to_csv(eval_csv, index=False)
    print(f"Saved: {train_csv}\nSaved: {test_csv}\nSaved: {eval_csv}")

    y_train = train_df[LABEL_ID_COLUMN].to_numpy()
    y_test = test_df[LABEL_ID_COLUMN].to_numpy()
    y_eval = eval_df[LABEL_ID_COLUMN].to_numpy()
    train_text, test_text, eval_text = train_df[TEXT_COLUMN], test_df[TEXT_COLUMN], eval_df[TEXT_COLUMN]

    print("\n=== Step 5: Train + evaluate every feature/classifier configuration ===")
    all_rows: List[dict] = []
    all_manifest_entries: List[dict] = []

    # TF-IDF variants
    for variant_name, extractor in make_tfidf_variants().items():
        rows = run_feature_variant(
            variant_name, extractor, train_text, test_text, eval_text,
            y_train, y_test, y_eval, idx_to_tag, results_dir,
            run_xgboost=not args.skip_gbt, xgb_dense_max_mb=args.xgb_dense_max_mb,
            seed=args.random_seed,
            classifiers_dir=classifiers_dir,
            cache_dir=cache_dir,
        )
        all_rows.extend(rows)
        # Extract manifest entries
        for row in rows:
            if "_manifest_entries" in row:
                all_manifest_entries.extend(row["_manifest_entries"])
                del row["_manifest_entries"]

    # Stock MiniLM embeddings (not fine-tuned)
    if args.skip_minilm:
        print("\nSkipping MiniLM pipelines entirely (--skip-minilm).")
    else:
        try:
            stock_embedder = MiniLMEmbedder(args.minilm_model_path, args.minilm_batch_size)
        except ImportError as exc:
            print(f"\nNOTE: Skipping MiniLM pipelines entirely -- missing dependency ({exc}). "
                  "Run `pip install sentence-transformers torch transformers` to enable them.")
            stock_embedder = None
        except Exception as exc:  # e.g. Hugging Face download blocked / offline
            print(f"\nNOTE: Skipping MiniLM pipelines -- could not load '{args.minilm_model_path}' ({exc}). "
                  "Check network access to huggingface.co or pass --minilm-model-path to a local copy.")
            stock_embedder = None

        if stock_embedder is not None:
            rows = run_feature_variant(
                "minilm_stock", stock_embedder, train_text, test_text, eval_text,
                y_train, y_test, y_eval, idx_to_tag, results_dir,
                run_xgboost=not args.skip_gbt, xgb_dense_max_mb=args.xgb_dense_max_mb,
                seed=args.random_seed,
                classifiers_dir=classifiers_dir,
                cache_dir=cache_dir,
            )
            all_rows.extend(rows)
            for row in rows:
                if "_manifest_entries" in row:
                    all_manifest_entries.extend(row["_manifest_entries"])
                    del row["_manifest_entries"]

        # Fine-tuned MiniLM embeddings
        if stock_embedder is None:
            pass  # already printed the missing-dependency note above; nothing further to do
        elif args.skip_finetune:
            print("\nSkipping MiniLM fine-tuning (--skip-finetune). Stock MiniLM results only.")
        else:
            print("\n=== Step 5c: Fine-tune all-MiniLM-L6-v2 on your historical training data ===")
            try:
                from sentence_transformer_finetuning import fine_tune_sentence_transformer
                fine_tuned_model_dir = fine_tune_sentence_transformer(
                    base_model_path=args.minilm_model_path,
                    train_df=train_df,
                    text_column=TEXT_COLUMN,
                    label_column=LABEL_ID_COLUMN,
                    num_classes=len(idx_to_tag),
                    num_epochs=args.finetune_epochs,
                    batch_size=args.finetune_batch_size,
                    learning_rate=args.finetune_learning_rate,
                    checkpoint_dir_template=ft_checkpoint_dir_template,
                    final_model_dir=ft_model_final,
                    seed=args.random_seed,
                    loss_history_json=ft_loss_history_json,
                    warmup_ratio=args.finetune_warmup_ratio,
                    lr_scheduler_type=args.finetune_lr_scheduler,
                    val_df=test_df,  # per-epoch validation + best-epoch selection (EVAL stays untouched)
                )
            except ImportError as exc:
                print(f"\nNOTE: Skipping fine-tuning -- missing dependency ({exc}). "
                      "Run `pip install torch transformers sentence-transformers` to enable it.")
                fine_tuned_model_dir = None

            if fine_tuned_model_dir is not None:
                finetuned_embedder = MiniLMEmbedder(fine_tuned_model_dir, args.minilm_batch_size)
                rows = run_feature_variant(
                    "minilm_finetuned", finetuned_embedder, train_text, test_text, eval_text,
                    y_train, y_test, y_eval, idx_to_tag, results_dir,
                    run_xgboost=not args.skip_gbt, xgb_dense_max_mb=args.xgb_dense_max_mb,
                    seed=args.random_seed,
                    classifiers_dir=classifiers_dir,
                    cache_dir=cache_dir,
                )
                all_rows.extend(rows)
                for row in rows:
                    if "_manifest_entries" in row:
                        all_manifest_entries.extend(row["_manifest_entries"])
                        del row["_manifest_entries"]

    print("\n=== Step 6: Final side-by-side comparison ===")
    comparison = pd.DataFrame(all_rows).sort_values(
        ["test_accuracy", "eval_accuracy", "model"], ascending=[False, False, True],
    )
    comparison_path = os.path.join(results_dir, "accuracy_comparison_all_configurations.csv")
    comparison.to_csv(comparison_path, index=False)

    # Save manifest for evaluate_probabilities.py
    manifest_path = os.path.join(classifiers_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(all_manifest_entries, f, indent=2)
    print(f"Saved classifier manifest: {manifest_path}")
    printable = comparison.copy()
    for col in ["test_accuracy", "eval_accuracy"]:
        printable[col] = printable[col].map(lambda v: "N/A" if pd.isna(v) else f"{float(v):.2%}")
    print(printable.rename(columns={
        "model": "Model", "test_accuracy": "Held-out test", "eval_accuracy": "Final eval",
    })[["Model", "Held-out test", "Final eval"]].to_string(index=False))

    svm_rows = comparison[comparison["classifier"] == "LinearSVC"]
    tfidf_best = svm_rows[svm_rows["feature_variant"].str.startswith("tfidf")].sort_values("test_accuracy", ascending=False)
    minilm_stock_best = svm_rows[svm_rows["feature_variant"] == "minilm_stock"]
    minilm_ft_best = svm_rows[svm_rows["feature_variant"] == "minilm_finetuned"]
    if not tfidf_best.empty:
        print("\n=== Headline three-way comparison (best-per-family, LinearSVC) ===")
        print(tfidf_best.iloc[0][["model", "test_accuracy", "eval_accuracy"]].to_string())
        if not minilm_stock_best.empty:
            print(minilm_stock_best.iloc[0][["model", "test_accuracy", "eval_accuracy"]].to_string())
        if not minilm_ft_best.empty:
            print(minilm_ft_best.iloc[0][["model", "test_accuracy", "eval_accuracy"]].to_string())
            if not minilm_stock_best.empty:
                improvement = (
                    minilm_ft_best.iloc[0]["test_accuracy"] - minilm_stock_best.iloc[0]["test_accuracy"]
                ) * 100
                print(f"\nFine-tuning improvement over stock MiniLM (held-out test): {improvement:+.1f} percentage points")

    print("\n=== Outputs saved ===")
    print(f"- {comparison_path}")
    print(f"- {run_config_json}")
    print(f"- {manifest_path}")
    print(f"- Saved classifiers and extractors: {classifiers_dir}")
    if cache_dir and os.listdir(cache_dir):
        print(f"- Cached MiniLM features: {cache_dir}")
    print(f"- classification_report_*.txt and confusion_matrix_*.csv under {results_dir}")
    print(
        "\nTIP: To check whether MORE fine-tuning epochs would likely have helped "
        "(without fine-tuning again), run:\n"
        f"  python src/evaluate_checkpoints.py --run-name {run_id}"
    )
    print(
        "\nTIP: To evaluate with probability predictions and calibration "
        "(loads saved models, no retraining), run:\n"
        f"  python src/evaluate_probabilities.py --run-name {run_id}"
    )


if __name__ == "__main__":
    main()
