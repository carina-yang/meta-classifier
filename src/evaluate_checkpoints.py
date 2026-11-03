"""
evaluate_checkpoints.py
-----------------------
Evaluates fine-tuning epoch checkpoints to determine if more epochs would
improve downstream classification accuracy.

Purpose:
Training loss alone doesn't indicate if additional epochs would improve
classification accuracy. This script empirically tests each saved epoch
checkpoint by:
1. Re-embedding existing train/test/eval splits with each checkpoint
2. Training a quick classifier on each set of embeddings
3. Comparing accuracy across epochs

Checkpoint structure:
  models/{run_id}/ft_epoch_0   (after epoch 1)
  models/{run_id}/ft_epoch_1   (after epoch 2)
  ...
  models/{run_id}/ft_epoch_{N-1}  (after epoch N)

No additional fine-tuning required - uses existing checkpoints and splits
from train_models.py.

If --run-name is omitted, automatically uses the most recently created run.

Usage:
    python src/evaluate_checkpoints.py --run-name lr_schedule_v1
    python src/evaluate_checkpoints.py                                  # uses the most recent run
Optional:
    python src/evaluate_checkpoints.py --run-name lr_schedule_v1 --skip-gbt   # LinearSVC only (faster)
    python src/evaluate_checkpoints.py --run-name lr_schedule_v1 --minilm-batch-size 128
    python src/evaluate_checkpoints.py --run-name lr_schedule_v1 --checkpoint-dir-template "models/lr_schedule_v1/ft_epoch_{epoch}"

Output:
    - Printed table: epoch | training loss | LR (if available) | test accuracy | eval accuracy (per classifier)
    - results/{run_id}/epoch_accuracy_comparison.csv
    - A printed verdict on whether the trend suggests more epochs would help

NOTE: ClassifierHead is skipped in checkpoint evaluation since it's tied to the
final fine-tuned model. Use LinearSVC/XGBoost for epoch comparison. The
classification head is only available for the final model (ft_final), not for
intermediate checkpoints.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from config import MINILM_BATCH_SIZE, LABEL_ID_COLUMN, RANDOM_SEED, TEXT_COLUMN  # noqa: E402
from feature_extractors import MiniLMEmbedder  # noqa: E402
from model_training import safe_transform, train_predict_svm, train_predict_xgboost  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every saved fine-tuning epoch checkpoint (for a given run) against "
                    "that run's existing train/test/eval splits, to check whether more epochs would likely help."
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="run_id to evaluate, as printed by train_models.py (e.g. 'lr_schedule_v1' "
             "or an auto-generated 'run_20260908_1543'). If omitted, the most recently created run "
             "under models/ is used automatically.",
    )
    parser.add_argument("--checkpoint-dir-template", type=str, default=None,
                        help="Override: template with {epoch} placeholder for checkpoint folders. "
                             "Default: models/{run-name}/ft_epoch_{epoch}.")
    parser.add_argument("--final-model-dir", type=str, default=None,
                        help="Override: path to the final fine-tuned model (included in the table as 'final'). "
                             "Default: models/{run-name}/ft_final.")
    parser.add_argument("--loss-history-json", type=str, default=None,
                        help="Override path. Default: models/{run-name}/ft_loss_history.json.")
    parser.add_argument("--minilm-batch-size", type=int, default=MINILM_BATCH_SIZE,
                        help="Batch size for embedding inference. Flag name kept for compatibility.")
    parser.add_argument("--skip-gbt", action="store_true", help="Only evaluate LinearSVC (faster).")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--xgb-dense-max-mb", type=int, default=512)
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Override path. Default: results/{run-name}/epoch_accuracy_comparison.csv.")
    return parser.parse_args()


def discover_checkpoints(checkpoint_dir_template: str, final_model_dir: str):
    """Find every epoch checkpoint that actually exists on disk, in order.
    Returns a list of (label, path) tuples, e.g. [("epoch_1", ".../epoch_0"),
    ("epoch_2", ".../epoch_1"), ..., ("final", final_model_dir)].
    Labels use 1-based epoch numbers for readability (checkpoint folders
    themselves are 0-indexed, matching sentence_transformer_finetuning.py).
    """
    checkpoints = []
    epoch = 0
    while True:
        path = checkpoint_dir_template.format(epoch=epoch)
        if not os.path.isdir(path):
            break
        checkpoints.append((f"epoch_{epoch + 1}", path))
        epoch += 1
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found matching '{checkpoint_dir_template}'. "
            "Run train_models.py (without --skip-finetune) first, "
            "or double-check --run-name."
        )
    if os.path.isdir(final_model_dir):
        # Include final model (usually last epoch, but may differ if fine-tuned separately)
        checkpoints.append(("final", final_model_dir))
    return checkpoints


def load_loss_history(path: str):
    """Load the per-epoch loss/LR JSON saved by sentence_transformer_finetuning.py,
    if present. Returns {0-based epoch index: {"loss": ..., "learning_rate": ...}},
    or {} if missing. Also handles the OLD format (a bare loss float per epoch,
    from before the LR-schedule change) for backward compatibility."""
    if not path or not os.path.exists(path):
        print(f"NOTE: No loss history found at {path} -- table will omit training loss/LR.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    history = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            history[int(k)] = {"loss": v.get("loss"), "learning_rate": v.get("learning_rate")}
        else:
            # Legacy format: bare float (loss only)
            history[int(k)] = {"loss": v, "learning_rate": None}
    return history


def main() -> None:
    args = parse_args()

    # Resolve run to evaluate
    run_id = args.run_name or config.find_latest_run_id()
    if run_id is None:
        raise FileNotFoundError(
            "No runs found under models/. Run train_models.py first, "
            "or pass --run-name explicitly."
        )
    if not args.run_name:
        print(f"NOTE: --run-name not provided; auto-selected the most recent run: '{run_id}'")
    print(f"\n{'=' * 70}\nEVALUATING RUN: {run_id}\n{'=' * 70}")

    checkpoint_dir_template = args.checkpoint_dir_template or config.get_ft_checkpoint_dir_template(run_id)
    final_model_dir = args.final_model_dir or config.get_ft_model_final(run_id)
    loss_history_json = args.loss_history_json or config.get_ft_loss_history_json(run_id)
    output_csv = args.output_csv or config.get_epoch_accuracy_csv(run_id)
    train_csv = config.get_train_csv(run_id)
    test_csv = config.get_test_csv(run_id)
    eval_csv = config.get_eval_csv(run_id)
    label_map_json = config.get_label_map_json(run_id)
    results_dir = config.get_results_dir(run_id)

    os.makedirs(results_dir, exist_ok=True)
    for required_path in (train_csv, test_csv, eval_csv, label_map_json):
        if not os.path.exists(required_path):
            raise FileNotFoundError(
                f"{required_path} not found. Run train_models.py --run-name {run_id} "
                "first (it saves these as part of Step 4)."
            )

    print("=== Loading existing train/test/eval splits + label mapping ===")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    eval_df = pd.read_csv(eval_csv)
    with open(label_map_json, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    num_classes = len(label_mapping)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)} | Eval rows: {len(eval_df)} | Classes: {num_classes}")

    train_text, test_text, eval_text = train_df[TEXT_COLUMN], test_df[TEXT_COLUMN], eval_df[TEXT_COLUMN]
    y_train = train_df[LABEL_ID_COLUMN].to_numpy()
    y_test = test_df[LABEL_ID_COLUMN].to_numpy()
    y_eval = eval_df[LABEL_ID_COLUMN].to_numpy()

    loss_history = load_loss_history(loss_history_json)
    checkpoints = discover_checkpoints(checkpoint_dir_template, final_model_dir)
    print(f"\nFound {len(checkpoints)} checkpoint(s) to evaluate: {[label for label, _ in checkpoints]}")

    rows = []
    for label, path in checkpoints:
        print(f"\n=== Evaluating checkpoint: {label} ({path}) ===")
        embedder = MiniLMEmbedder(path, args.minilm_batch_size)
        x_train = embedder.fit_transform(train_text)
        is_sparse = sparse.issparse(x_train)
        n_features = x_train.shape[1]
        x_test = safe_transform(embedder, test_text, n_features, is_sparse)
        x_eval = safe_transform(embedder, eval_text, n_features, is_sparse)

        # Map epoch checkpoint to 0-based index for loss/LR lookup
        loss_value, lr_value = None, None
        if label.startswith("epoch_"):
            epoch_num = int(label.split("_")[1])
            entry = loss_history.get(epoch_num - 1, {})
            loss_value = entry.get("loss")
            lr_value = entry.get("learning_rate")

        svm_test_pred, svm_eval_pred = train_predict_svm(x_train, y_train, x_test, x_eval, args.random_seed)
        svm_test_acc = accuracy_score(y_test, svm_test_pred) if len(y_test) else float("nan")
        svm_eval_acc = accuracy_score(y_eval, svm_eval_pred) if len(y_eval) else float("nan")
        print(f"  LinearSVC -> test: {svm_test_acc:.2%} | eval: {svm_eval_acc:.2%}")
        row = {
            "checkpoint": label, "classifier": "LinearSVC",
            "training_loss": loss_value, "learning_rate": lr_value,
            "test_accuracy": svm_test_acc, "eval_accuracy": svm_eval_acc,
        }
        rows.append(row)

        if not args.skip_gbt:
            itemsize = np.dtype(np.float32).itemsize
            estimated_dense_mb = (x_train.shape[0] * x_train.shape[1] * itemsize) / (1024 ** 2)
            if estimated_dense_mb <= args.xgb_dense_max_mb:
                xgb_test_pred, xgb_eval_pred = train_predict_xgboost(
                    x_train, y_train, x_test, x_eval, num_classes, args.random_seed,
                )
                xgb_test_acc = accuracy_score(y_test, xgb_test_pred) if len(y_test) else float("nan")
                xgb_eval_acc = accuracy_score(y_eval, xgb_eval_pred) if len(y_eval) else float("nan")
                print(f"  XGBoost   -> test: {xgb_test_acc:.2%} | eval: {xgb_eval_acc:.2%}")
                rows.append({
                    "checkpoint": label, "classifier": "XGBoost",
                    "training_loss": loss_value, "learning_rate": lr_value,
                    "test_accuracy": xgb_test_acc, "eval_accuracy": xgb_eval_acc,
                })

    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)

    print("\n=== Accuracy per checkpoint (in training order) ===")
    printable = results.copy()
    printable["training_loss"] = printable["training_loss"].map(
        lambda v: "N/A" if pd.isna(v) else f"{float(v):.4f}"
    )
    printable["learning_rate"] = printable["learning_rate"].map(
        lambda v: "N/A" if pd.isna(v) else f"{float(v):.2e}"
    )
    for col in ["test_accuracy", "eval_accuracy"]:
        printable[col] = printable[col].map(lambda v: "N/A" if pd.isna(v) else f"{float(v):.2%}")
    print(printable.rename(columns={
        "checkpoint": "Checkpoint", "classifier": "Classifier", "training_loss": "Train loss",
        "learning_rate": "LR (epoch end)", "test_accuracy": "Held-out test", "eval_accuracy": "Final eval",
    }).to_string(index=False))

    # Verdict: was accuracy still improving?
    print("\n=== Verdict: were more epochs likely to help? ===")
    for classifier_name in results["classifier"].unique():
        subset = results[results["classifier"] == classifier_name]
        epoch_rows = subset[subset["checkpoint"].str.startswith("epoch_")].copy()
        if len(epoch_rows) < 2:
            print(f"[{classifier_name}] Not enough epoch checkpoints to compare (need >= 2).")
            continue
        epoch_rows["epoch_num"] = epoch_rows["checkpoint"].str.split("_").str[1].astype(int)
        epoch_rows = epoch_rows.sort_values("epoch_num")
        last_two = epoch_rows.tail(2)["test_accuracy"].to_numpy()
        delta = last_two[-1] - last_two[-2]
        last_epoch = int(epoch_rows["epoch_num"].iloc[-1])
        if delta > 0.01:
            verdict = (
                f"Test accuracy was STILL RISING by +{delta:.2%} from epoch {last_epoch - 1} to "
                f"{last_epoch} -- additional epochs likely would have helped further."
            )
        elif delta < -0.01:
            verdict = (
                f"Test accuracy DROPPED by {delta:.2%} from epoch {last_epoch - 1} to {last_epoch} -- "
                "a sign of overfitting; the earlier checkpoint may generalize better than the final one."
            )
        else:
            verdict = (
                f"Test accuracy changed by only {delta:+.2%} from epoch {last_epoch - 1} to {last_epoch} -- "
                "accuracy had already PLATEAUED; your epoch count was likely already sufficient."
            )
        print(f"[{classifier_name}] {verdict}")

    print(f"\n=== Output saved ===\n- {output_csv}")


if __name__ == "__main__":
    main()
