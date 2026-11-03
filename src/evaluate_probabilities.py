"""
evaluate_probabilities.py
-------------------------
Re-evaluates an existing classification run with probability predictions and
optional temperature calibration.

Reuses saved train/test/eval splits and label mapping. Fits only the
downstream classifier heads (LinearSVC/XGBoost) - no fine-tuning required.

Methodology:
- Temperature calibration fitted on TEST split
- Final metrics reported on EVAL split
- Saves both uncalibrated and calibrated probability matrices
- Includes calibration quality metrics (ECE, MCE)

Outputs:
- Probability matrices and predictions
- Confusion matrices and metrics
- Threshold curves
- Performance summary markdown
- Model disagreement analysis

Usage:
    python src/evaluate_probabilities.py --run-name baseline
    python src/evaluate_probabilities.py --skip-minilm
    python src/evaluate_probabilities.py --calibration none
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
)
from sklearn.svm import LinearSVC
import sklearn

try:
    import xgboost as xgb
    HAS_XGB = True
except (ImportError, Exception):
    xgb = None
    HAS_XGB = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from config import (  # noqa: E402
    COL_DATA_TYPE, COL_NAME, COL_TABLE_NAME, MINILM_BATCH_SIZE, LABEL_ID_COLUMN,
    RANDOM_SEED, TEXT_COLUMN, XGB_NUM_ROUND, XGB_PARAMS,
)

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and evaluate saved classifiers with probability predictions. "
                    "Does NOT retrain - loads models saved by train_models.py."
    )
    parser.add_argument("--run-name", default=None,
                        help="Run ID to evaluate. If omitted, uses the most recent run.")
    parser.add_argument("--skip-gbt", action="store_true",
                        help="Skip XGBoost configurations (LinearSVC/ClassifierHead only).")
    parser.add_argument("--skip-minilm", action="store_true",
                        help="Skip embedding configurations (TF-IDF only). Flag name kept for compatibility.")
    parser.add_argument("--minilm-batch-size", type=int, default=MINILM_BATCH_SIZE,
                        help="Batch size for embedding inference. Flag name kept for compatibility.")
    parser.add_argument("--latency-samples", type=int, default=500,
                        help="Number of samples for latency measurement.")
    parser.add_argument("--hourly-rate-usd", type=float, default=None,
                        help="Hourly compute rate (USD) for cost estimation. Omit for 'N/A'.")
    parser.add_argument("--calibration", choices=["none", "temperature"], default="temperature",
                        help="Calibration method: 'temperature' fits on TEST, 'none' uses raw probabilities.")
    parser.add_argument("--threshold-step", type=float, default=0.01,
                        help="Step size for threshold curve (0.0-1.0).")
    # Deprecated but accepted for backward compatibility
    parser.add_argument("--xgb-dense-max-mb", type=int, default=512,
                        help=argparse.SUPPRESS)  # No longer used (training time only)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help=argparse.SUPPRESS)  # No longer used (training time only)
    return parser.parse_args()


def empty_probabilities(n_classes: int) -> np.ndarray:
    return np.empty((0, n_classes), dtype=np.float64)


def ensure_probability_shape(values: np.ndarray, n_classes: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return empty_probabilities(n_classes)
    if values.ndim == 1:
        if n_classes != 2:
            raise ValueError(f"A 1-D score vector is only valid for binary classification, got {n_classes} classes.")
        values = np.column_stack([-values, values])
    if values.shape[1] != n_classes:
        raise ValueError(f"Expected {n_classes} score columns, got {values.shape[1]}.")
    return values


def decision_scores(classifier: LinearSVC, x: Any, n_classes: int) -> np.ndarray:
    if x.shape[0] == 0:
        return empty_probabilities(n_classes)
    return ensure_probability_shape(classifier.decision_function(x), n_classes)


def probabilities_from_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if scores.shape[0] == 0:
        return scores.copy()
    return softmax(scores / max(float(temperature), EPS), axis=1)


def xgb_probabilities(booster: Any, x: Any, n_classes: int) -> np.ndarray:
    if not HAS_XGB:
        raise ImportError("XGBoost is not available")
    if x.shape[0] == 0:
        return empty_probabilities(n_classes)
    return ensure_probability_shape(np.asarray(booster.predict(xgb.DMatrix(x))), n_classes)


def fit_temperature(logits: np.ndarray, y: np.ndarray, n_classes: int, config_name: str) -> float:
    if len(y) == 0:
        print(f"WARNING [{config_name}]: TEST is empty; using temperature=1.0 (uncalibrated).")
        return 1.0
    try:
        def objective(log_t: float) -> float:
            p = probabilities_from_scores(logits, math.exp(log_t))
            return float(log_loss(y, p, labels=list(range(n_classes))))
        result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(result.message)
        return float(math.exp(result.x))
    except Exception as exc:
        print(f"WARNING [{config_name}]: temperature fitting failed ({exc}); using uncalibrated probabilities.")
        return 1.0


def multiclass_brier(y: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    if len(y) == 0:
        return float("nan")
    targets = np.eye(n_classes, dtype=np.float64)[y.astype(int)]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def expected_calibration_error(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        mask = (confidence >= edges[idx]) & (confidence < edges[idx + 1] if idx < bins - 1 else confidence <= edges[idx + 1])
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def split_metrics(y: np.ndarray, pred: np.ndarray, prob_uncal: np.ndarray, prob_cal: np.ndarray,
                  labels: List[int]) -> Dict[str, float]:
    if len(y) == 0:
        return {key: float("nan") for key in [
            "accuracy", "macro_f1_all_classes", "macro_f1_supported_classes", "weighted_f1",
            "log_loss_uncalibrated", "log_loss_calibrated", "brier_uncalibrated",
            "brier_calibrated", "ece_uncalibrated", "ece_calibrated"]}
    supported = sorted(set(int(v) for v in y))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1_all_classes": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_supported_classes": float(f1_score(y, pred, labels=supported, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "log_loss_uncalibrated": float(log_loss(y, prob_uncal, labels=labels)),
        "log_loss_calibrated": float(log_loss(y, prob_cal, labels=labels)),
        "brier_uncalibrated": multiclass_brier(y, prob_uncal, len(labels)),
        "brier_calibrated": multiclass_brier(y, prob_cal, len(labels)),
        "ece_uncalibrated": expected_calibration_error(y, prob_uncal),
        "ece_calibrated": expected_calibration_error(y, prob_cal),
    }


def directory_size(path: Optional[str]) -> int:
    if not path or not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(root, name)) for root, _, files in os.walk(path) for name in files)


def rss_mb() -> Tuple[float, str]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2), "psutil RSS"
    except ImportError:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return value / (1024 ** 2), "resource ru_maxrss"
        return value / 1024, "resource ru_maxrss"


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": platform.platform(), "cpu_model": platform.processor() or platform.machine(),
        "logical_core_count": os.cpu_count(), "python": platform.python_version(),
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                      "scikit_learn": sklearn.__version__, "xgboost": xgb.__version__ if HAS_XGB else "not available",
                      "matplotlib": importlib.metadata.version("matplotlib"),
                      "joblib": importlib.metadata.version("joblib")},
        "cuda_available": False, "mps_available": False, "minilm_devices": {},
    }
    try:
        import psutil
        info["total_ram_bytes"] = int(psutil.virtual_memory().total)
    except ImportError:
        try:
            info["total_ram_bytes"] = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            info["total_ram_bytes"] = None
    try:
        import torch
        info["libraries"]["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except ImportError:
        info["libraries"]["torch"] = None
    return info


def batched(series: pd.Series, size: int) -> Iterable[pd.Series]:
    for start in range(0, len(series), max(1, size)):
        yield series.iloc[start:start + max(1, size)]


def classifier_probabilities(classifier_name: str, classifier: Any, x: Any,
                             n_classes: int, temperature: float) -> np.ndarray:
    if classifier_name == "LinearSVC":
        return probabilities_from_scores(decision_scores(classifier, x, n_classes), temperature)
    elif classifier_name == "ClassifierHead":
        # Classification head outputs logits directly
        import torch
        if x.shape[0] == 0:
            return empty_probabilities(n_classes)
        device = next(classifier.parameters()).device
        with torch.no_grad():
            x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)
            features = {"sentence_embedding": x_tensor}
            logits = classifier(features)
            logits_np = logits.cpu().numpy()
        return probabilities_from_scores(logits_np, temperature)
    else:  # XGBoost
        raw = xgb_probabilities(classifier, x, n_classes)
        if raw.shape[0] == 0:
            return raw
        return probabilities_from_scores(np.log(np.clip(raw, EPS, 1.0)), temperature)


def full_inference(extractor: Any, classifier_name: str, classifier: Any, text: pd.Series,
                   n_classes: int, temperature: float) -> np.ndarray:
    x = extractor.transform(text)
    return classifier_probabilities(classifier_name, classifier, x, n_classes, temperature)


def measure_performance(extractor: Any, classifier_name: str, classifier: Any, eval_text: pd.Series,
                        n_classes: int, temperature: float, latency_samples: int,
                        configured_batch: int, seed: int) -> Dict[str, float]:
    if len(eval_text) == 0:
        return {"latency_median_ms": np.nan, "latency_p95_ms": np.nan,
                "throughput_batch_1_rps": np.nan, "throughput_configured_batch_rps": np.nan}
    warm = eval_text.iloc[:min(3, len(eval_text))]
    for i in range(len(warm)):
        full_inference(extractor, classifier_name, classifier, warm.iloc[i:i + 1], n_classes, temperature)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(eval_text), size=min(latency_samples, len(eval_text)), replace=False)
    timings = []
    for idx in chosen:
        start = time.perf_counter()
        full_inference(extractor, classifier_name, classifier, eval_text.iloc[idx:idx + 1], n_classes, temperature)
        timings.append(time.perf_counter() - start)
    throughputs = {}
    for label, batch_size in [("batch_1", 1), ("configured_batch", configured_batch)]:
        start = time.perf_counter()
        for chunk in batched(eval_text, batch_size):
            full_inference(extractor, classifier_name, classifier, chunk, n_classes, temperature)
        elapsed = time.perf_counter() - start
        throughputs[label] = len(eval_text) / elapsed if elapsed > 0 else np.nan
    return {"latency_median_ms": float(np.median(timings) * 1000),
            "latency_p95_ms": float(np.percentile(timings, 95) * 1000),
            "throughput_batch_1_rps": float(throughputs["batch_1"]),
            "throughput_configured_batch_rps": float(throughputs["configured_batch"])}


def save_confusions(output_dir: str, slug: str, split: str, y: np.ndarray, pred: np.ndarray,
                    labels: List[int], names: List[str]) -> None:
    counts = confusion_matrix(y, pred, labels=labels) if len(y) else np.zeros((len(labels), len(labels)), dtype=int)
    sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, sums, out=np.zeros_like(counts, dtype=float), where=sums != 0)
    index = [f"true_{name}" for name in names]
    columns = [f"pred_{name}" for name in names]
    pd.DataFrame(counts, index=index, columns=columns).to_csv(os.path.join(output_dir, f"confusion_matrix_{slug}_{split}_counts.csv"))
    pd.DataFrame(normalized, index=index, columns=columns).to_csv(os.path.join(output_dir, f"confusion_matrix_{slug}_{split}_normalized.csv"))


def threshold_curve(y: np.ndarray, probabilities: np.ndarray, step: float, labels: List[int]) -> pd.DataFrame:
    rows = []
    thresholds = np.unique(np.append(np.arange(0.0, 1.0 + step / 2, step), 1.0))
    pred = probabilities.argmax(axis=1) if len(y) else np.array([], dtype=int)
    conf = probabilities.max(axis=1) if len(y) else np.array([])
    for threshold in thresholds:
        accepted = conf >= min(float(threshold), 1.0)
        n = int(accepted.sum())
        rows.append({"threshold": min(float(threshold), 1.0), "coverage": n / len(y) if len(y) else np.nan,
                     "accuracy": accuracy_score(y[accepted], pred[accepted]) if n else np.nan,
                     "macro_f1": f1_score(y[accepted], pred[accepted], labels=labels, average="macro", zero_division=0) if n else np.nan,
                     "precision": precision_score(y[accepted], pred[accepted], labels=labels, average="macro", zero_division=0) if n else np.nan,
                     "abstained": len(y) - n})
    return pd.DataFrame(rows)


def save_threshold_plot(curve: pd.DataFrame, path: str, title: str) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(curve["coverage"], curve["accuracy"], label="Accuracy")
    plt.plot(curve["coverage"], curve["macro_f1"], label="Macro-F1")
    plt.xlabel("Coverage")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def prediction_frame(eval_df: pd.DataFrame, y: np.ndarray, probabilities: np.ndarray,
                     names: List[str]) -> pd.DataFrame:
    pred = probabilities.argmax(axis=1) if len(y) else np.array([], dtype=int)
    top = np.argsort(-probabilities, axis=1)[:, :min(3, len(names))] if len(y) else np.empty((0, 0), dtype=int)
    result = pd.DataFrame()
    for col in [COL_TABLE_NAME, COL_NAME, COL_DATA_TYPE]:
        result[col] = eval_df[col] if col in eval_df else ""
    result["true_label"] = [names[int(v)] for v in y]
    result["predicted_label"] = [names[int(v)] for v in pred]
    result["confidence"] = probabilities.max(axis=1) if len(y) else []
    for rank in range(3):
        result[f"top_{rank + 1}_label"] = [names[row[rank]] if rank < len(row) else "" for row in top]
        result[f"top_{rank + 1}_probability"] = [probabilities[i, row[rank]] if rank < len(row) else np.nan for i, row in enumerate(top)]
    result["correct"] = np.where(pred == y, "yes", "no")
    return result


def check_argmax(config_name: str, split: str, original: np.ndarray, probabilities: np.ndarray) -> None:
    if len(original) and not np.array_equal(original, probabilities.argmax(axis=1)):
        print(f"WARNING [{config_name}]: probability argmax differs from original predict() path on {split}.")


def _load_extractor(entry: Dict[str, Any], classifiers_dir: str, args: argparse.Namespace) -> Any:
    """Load an extractor from saved artifacts."""
    extractor_path = os.path.join(classifiers_dir, entry["extractor_path"])
    extractor_type = entry["extractor_type"]

    if extractor_type == "tfidf":
        return joblib.load(extractor_path)
    elif extractor_type == "minilm":
        with open(extractor_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        from feature_extractors import MiniLMEmbedder
        return MiniLMEmbedder(
            metadata["model_path"],
            metadata.get("batch_size", args.minilm_batch_size),
            metadata.get("max_seq_length", 64)
        )
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")


def _load_classifier_head(classifier_path: str) -> Any:
    """Load the pre-trained classification head."""
    import json
    import torch
    import os

    # Read reference to head path
    with open(classifier_path, "r", encoding="utf-8") as f:
        ref = json.load(f)

    head_path = ref["head_path"]

    # Load metadata
    metadata_path = os.path.join(head_path, "head_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Reconstruct head
    from sentence_transformer_finetuning import _build_model_classes
    ClassificationHead, _ = _build_model_classes()
    head = ClassificationHead(metadata["embedding_dim"], metadata["num_classes"])

    # Load weights
    weights_path = os.path.join(head_path, "head_weights.pt")
    head.load_state_dict(torch.load(weights_path, map_location="cpu"))
    head.eval()

    return head


def _load_cached_features(entry: Dict[str, Any], split: str, classifiers_dir: str) -> Optional[np.ndarray]:
    """Load cached features if available."""
    if not entry.get("feature_cache"):
        return None
    cache_path = entry["feature_cache"].get(split)
    if not cache_path:
        return None
    full_path = os.path.join(classifiers_dir, cache_path)
    if not os.path.exists(full_path):
        return None
    return np.load(full_path)


def run_feature_config(entry: Dict[str, Any], frames: Dict[str, pd.DataFrame],
                       labels_by_split: Dict[str, np.ndarray], names: List[str], args: argparse.Namespace,
                       output_dir: str, classifiers_dir: str, hardware: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process one classifier configuration by loading saved artifacts."""
    variant_name = entry["variant"]
    classifier_name = entry["classifier"]
    slug = entry["slug"]
    config_name = f"{variant_name} + {classifier_name}"
    print(f"\n=== Loading {config_name} ===")

    # Load extractor
    extractor = _load_extractor(entry, classifiers_dir, args)

    # Load or compute features
    matrices = {}
    cache_used = {"train": False, "test": False, "eval": False}
    for split in ("train", "test", "eval"):
        cached = _load_cached_features(entry, split, classifiers_dir)
        if cached is not None:
            matrices[split] = cached
            cache_used[split] = True
        else:
            matrices[split] = extractor.transform(frames[split][TEXT_COLUMN])

    if any(cache_used.values()):
        print(f"  Using cached features: {[s for s, used in cache_used.items() if used]}")

    is_sparse = sparse.issparse(matrices["train"])
    n_features = matrices["train"].shape[1]

    # Load classifier
    classifier_path = os.path.join(classifiers_dir, entry["classifier_path"])
    if classifier_name == "LinearSVC":
        classifier = joblib.load(classifier_path)
        logits = {split: decision_scores(classifier, matrices[split], len(names)) for split in frames}
        uncal = {split: probabilities_from_scores(logits[split]) for split in frames}
        original_pred = {split: classifier.predict(matrices[split]) if len(frames[split]) else np.array([], dtype=int) for split in frames}
        calibration_scores = logits
    elif classifier_name == "ClassifierHead":
        import torch
        classifier = _load_classifier_head(classifier_path)
        device = next(classifier.parameters()).device

        def _get_head_logits(x, n_classes):
            if x.shape[0] == 0:
                return np.empty((0, n_classes))
            with torch.no_grad():
                x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)
                features = {"sentence_embedding": x_tensor}
                logits = classifier(features)
                return logits.cpu().numpy()

        logits = {split: _get_head_logits(matrices[split], len(names)) for split in frames}
        uncal = {split: probabilities_from_scores(logits[split]) for split in frames}
        original_pred = {split: uncal[split].argmax(axis=1) if len(frames[split]) else np.array([], dtype=int) for split in frames}
        calibration_scores = logits
    else:  # XGBoost
        if not HAS_XGB:
            raise ImportError("XGBoost is not available")
        booster = xgb.Booster()
        booster.load_model(classifier_path)
        uncal = {split: xgb_probabilities(booster, matrices[split], len(names)) for split in frames}
        original_pred = {split: uncal[split].argmax(axis=1) if len(frames[split]) else np.array([], dtype=int) for split in frames}
        calibration_scores = {split: np.log(np.clip(uncal[split], EPS, 1.0)) for split in frames}
        classifier = booster

    # Load metadata
    metadata_path = os.path.join(classifiers_dir, entry["metadata_path"])
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    fit_time = metadata["fit_time_seconds"]
    memory_delta = metadata["memory_rss_delta_mb"]

    # Temperature calibration
    temperature = 1.0
    if args.calibration == "temperature":
        temperature = fit_temperature(calibration_scores["test"], labels_by_split["test"], len(names), config_name)
    calibrated = {split: probabilities_from_scores(calibration_scores[split], temperature) for split in frames}
    pred = {split: calibrated[split].argmax(axis=1) if len(frames[split]) else np.array([], dtype=int) for split in frames}

    # Sanity check
    for split in frames:
        check_argmax(config_name, split, original_pred[split], calibrated[split])
    # Save probabilities
    output_rows: List[Dict[str, Any]] = []
    np.savez_compressed(os.path.join(output_dir, f"probabilities_{slug}.npz"),
                        test_uncalibrated=uncal["test"], test_calibrated=calibrated["test"],
                        eval_uncalibrated=uncal["eval"], eval_calibrated=calibrated["eval"],
                        temperature=np.array([temperature]))

    train_accuracy = accuracy_score(labels_by_split["train"], pred["train"]) if len(frames["train"]) else np.nan
    train_macro = f1_score(labels_by_split["train"], pred["train"], labels=list(range(len(names))), average="macro", zero_division=0) if len(frames["train"]) else np.nan

    metric_rows = []
    for split in ("test", "eval"):
        values = split_metrics(labels_by_split[split], pred[split], uncal[split], calibrated[split], list(range(len(names))))
        values.update({"config": config_name, "variant": variant_name, "classifier": classifier_name,
                       "split": split, "temperature": temperature,
                       "calibration_method": args.calibration,
                       "training_accuracy": train_accuracy, "training_macro_f1_all_classes": train_macro})
        metric_rows.append(values)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_by_split[split], pred[split], labels=list(range(len(names))), zero_division=0)
        output_rows.append({"kind": "metrics", **values})
        for idx, name in enumerate(names):
            output_rows.append({"kind": "per_class", "config": config_name, "split": split, "class": name,
                                "precision": precision[idx], "recall": recall[idx], "f1": f1[idx], "support": int(support[idx])})
        save_confusions(output_dir, slug, split, labels_by_split[split], pred[split], list(range(len(names))), names)

    curve = threshold_curve(labels_by_split["eval"], calibrated["eval"], args.threshold_step, list(range(len(names))))
    curve.to_csv(os.path.join(output_dir, f"threshold_curve_{slug}.csv"), index=False)
    save_threshold_plot(curve, os.path.join(output_dir, f"threshold_curve_{slug}.png"), config_name)
    predictions = prediction_frame(frames["eval"], labels_by_split["eval"], calibrated["eval"], names)
    predictions.to_csv(os.path.join(output_dir, f"predictions_{slug}.csv"), index=False)
    predictions[predictions["correct"] == "no"].sort_values("confidence", ascending=False).head(50).to_csv(
        os.path.join(output_dir, f"high_confidence_errors_{slug}.csv"), index=False)

    # Performance measurement
    batch_size = args.minilm_batch_size
    perf = measure_performance(extractor, classifier_name, classifier, frames["eval"][TEXT_COLUMN],
                               len(names), temperature, args.latency_samples, batch_size, args.random_seed)

    # Disk size calculation
    extractor_path = os.path.join(classifiers_dir, entry["extractor_path"])
    classifier_path_full = os.path.join(classifiers_dir, entry["classifier_path"])
    disk_bytes = directory_size(extractor_path) + directory_size(classifier_path_full)

    # For MiniLM, count model directory only once per variant
    if variant_name.startswith("minilm"):
        with open(extractor_path, 'r', encoding='utf-8') as f:
            model_meta = json.load(f)
        model_path = model_meta["model_path"]
        if os.path.isdir(model_path) and variant_name not in hardware.get("_model_sizes_counted", set()):
            disk_bytes += directory_size(model_path)
            hardware.setdefault("_model_sizes_counted", set()).add(variant_name)

    throughput = perf["throughput_configured_batch_rps"]
    if args.hourly_rate_usd is None:
        cost_million: Any = "N/A (no rate provided)"
        training_cost: Any = "N/A (no rate provided)"
    else:
        cost_million = args.hourly_rate_usd * (1_000_000 / throughput) / 3600 if throughput and np.isfinite(throughput) else np.nan
        training_cost = args.hourly_rate_usd * fit_time / 3600

    device = "N/A"
    if variant_name.startswith("minilm"):
        device = str(extractor.model.device)
        hardware["minilm_devices"][variant_name] = device

    output_rows.append({"kind": "performance", "config": config_name, "variant": variant_name,
                        "classifier": classifier_name, **perf, "fit_time_seconds": fit_time,
                        "disk_size_bytes": disk_bytes, "disk_size_mb": disk_bytes / (1024 ** 2),
                        "memory_rss_delta_mb": memory_delta, "memory_measurement": "from_saved_metadata",
                        "device": device, "estimated_cost_per_1m_records_usd": cost_million,
                        "measured_training_cost_usd": training_cost})
    output_rows.append({"kind": "prediction_cache", "config": config_name, "variant": variant_name,
                        "classifier": classifier_name, "predictions": predictions,
                        "eval_accuracy": metric_rows[-1]["accuracy"], "curve": curve})
    gc.collect()
    return output_rows


def sanity_check(run_id: str, metrics: pd.DataFrame) -> None:
    """Verify loaded models produce the same accuracies as the original training run."""
    path = os.path.join(config.get_results_dir(run_id), "accuracy_comparison_all_configurations.csv")
    if not os.path.exists(path):
        print("NOTE: Original accuracy comparison not found; skipping save/load integrity check.")
        return
    prior = pd.read_csv(path)
    current = metrics.pivot_table(index=["variant", "classifier"], columns="split", values="accuracy").reset_index()
    issues = []
    for _, row in current.iterrows():
        match = prior[(prior["feature_variant"] == row["variant"]) & (prior["classifier"] == row["classifier"])]
        if match.empty:
            continue
        for split in ("test", "eval"):
            old_col = f"{split}_accuracy"
            if split in row and old_col in match.columns and pd.notna(row[split]) and pd.notna(match.iloc[0][old_col]):
                delta = abs(float(row[split]) - float(match.iloc[0][old_col]))
                if delta > 0.005:
                    issues.append(f"{row['variant']} + {row['classifier']} {split}: {delta:.2%} difference")
    if issues:
        print("WARNING: Save/load integrity check found accuracy differences:")
        for issue in issues:
            print(f"  - {issue}")
        print("This may indicate a problem with model saving/loading or data splits.")
    else:
        print("Save/load integrity check: All accuracies match original training run (within 0.5%).")


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold_step <= 1:
        raise ValueError("--threshold-step must be in (0, 1].")
    run_id = args.run_name or config.find_latest_run_id()
    if run_id is None:
        raise FileNotFoundError("No prior run found. Pass --run-name or create a run first.")
    if args.run_name is None:
        print(f"NOTE: --run-name not provided; auto-selected the most recent run: '{run_id}'")
    print(f"\n{'=' * 70}\nPROBABILITY EVALUATION RUN: {run_id}\n{'=' * 70}")
    print("Methodology: fit temperature on TEST; report headline model comparison on EVAL.")
    run_config_path = config.get_run_config_json(run_id)
    if os.path.exists(run_config_path):
        with open(run_config_path, "r", encoding="utf-8") as handle:
            original_seed = json.load(handle).get("random_seed")
        if original_seed is not None and args.random_seed == RANDOM_SEED and int(original_seed) != args.random_seed:
            print(f"NOTE: using original run seed {original_seed} instead of default {args.random_seed}.")
            args.random_seed = int(original_seed)
        elif original_seed is not None and int(original_seed) != args.random_seed:
            print(f"WARNING: --random-seed {args.random_seed} differs from original run seed {original_seed}; accuracies may not reproduce.")
    required = {"train": config.get_train_csv(run_id), "test": config.get_test_csv(run_id),
                "eval": config.get_eval_csv(run_id)}
    required["labels"] = config.get_label_map_json(run_id)
    for path in required.values():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    frames = {split: pd.read_csv(required[split]) for split in ("train", "test", "eval")}
    if frames["train"].empty:
        raise ValueError("Saved training split is empty; classifiers cannot be fitted.")
    with open(required["labels"], "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    labels_by_split = {split: frames[split][LABEL_ID_COLUMN].to_numpy(dtype=int) for split in frames}
    output_dir = os.path.join(config.get_results_dir(run_id), "probabilities")
    classifiers_dir = config.get_classifiers_dir(run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Load manifest
    manifest_path = os.path.join(classifiers_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Classifier manifest not found: {manifest_path}\n"
            f"Please run train_models.py --run-name {run_id} first to train and save models."
        )

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    print(f"Loaded manifest with {len(manifest)} classifier configurations.")

    # Validate all referenced files exist
    missing = []
    for entry in manifest:
        for key in ["extractor_path", "classifier_path", "metadata_path"]:
            path = os.path.join(classifiers_dir, entry[key])
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} required file(s) from manifest:\n" +
            "\n".join(f"  - {p}" for p in missing[:5]) +
            ("\n  ..." if len(missing) > 5 else "") +
            f"\n\nPlease run train_models.py --run-name {run_id} first."
        )

    # Filter manifest based on CLI flags
    filtered_manifest = []
    for entry in manifest:
        if args.skip_gbt and entry["classifier"] == "XGBoost":
            continue
        if args.skip_minilm and entry["variant"].startswith("minilm"):
            continue
        filtered_manifest.append(entry)

    print(f"Processing {len(filtered_manifest)} configurations (after filters).")

    hardware = hardware_info()
    all_rows: List[Dict[str, Any]] = []
    for entry in filtered_manifest:
        all_rows.extend(run_feature_config(entry, frames, labels_by_split, names,
                                           args, output_dir, classifiers_dir, hardware))
    metrics = pd.DataFrame([row for row in all_rows if row["kind"] == "metrics"]).drop(columns="kind")
    per_class = pd.DataFrame([row for row in all_rows if row["kind"] == "per_class"]).drop(columns="kind")
    performance = pd.DataFrame([row for row in all_rows if row["kind"] == "performance"]).drop(columns="kind")
    caches = [row for row in all_rows if row["kind"] == "prediction_cache"]
    metrics.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    per_class.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)
    performance.to_csv(os.path.join(output_dir, "performance_summary.csv"), index=False)
    sanity_check(run_id, metrics)
    plt.figure(figsize=(8, 5))
    for item in caches:
        curve = item["curve"]
        plt.plot(curve["coverage"], curve["accuracy"], label=item["config"])
    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs coverage")
    if caches:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_curves_all.png"), dpi=160)
    plt.close()
    tfidf = [row for row in caches if row["variant"].startswith("tfidf")]
    finetuned = [row for row in caches if row["variant"] == "minilm_finetuned"]
    disagreements_path = os.path.join(output_dir, "disagreements_best_tfidf_vs_best_finetuned.csv")
    if tfidf and finetuned:
        best_tfidf = max(tfidf, key=lambda row: (-np.inf if pd.isna(row["eval_accuracy"]) else row["eval_accuracy"]))
        best_ft = max(finetuned, key=lambda row: (-np.inf if pd.isna(row["eval_accuracy"]) else row["eval_accuracy"]))
        left = best_tfidf["predictions"].copy()
        right = best_ft["predictions"][["predicted_label", "confidence", "correct"]].copy()
        right.columns = ["finetuned_prediction", "finetuned_confidence", "finetuned_correct"]
        left = left.rename(columns={"predicted_label": "tfidf_prediction", "confidence": "tfidf_confidence",
                                    "correct": "tfidf_correct"})
        merged = pd.concat([left, right], axis=1)
        merged[merged["tfidf_prediction"] != merged["finetuned_prediction"]].to_csv(disagreements_path, index=False)
    else:
        pd.DataFrame(columns=["note"]).assign(note=["Requires both a TF-IDF and fine-tuned MiniLM result."]).to_csv(disagreements_path, index=False)
    hardware["methodology"] = "Temperature fitted on TEST; headline comparison reported on EVAL."
    hardware.pop("_model_sizes_counted", None)
    with open(os.path.join(output_dir, "hardware.json"), "w", encoding="utf-8") as handle:
        json.dump(hardware, handle, indent=2)
    headline = metrics[metrics["split"] == "eval"].merge(performance, on=["config", "variant", "classifier"], how="left")
    headline = headline.sort_values("accuracy", ascending=False)
    columns = ["config", "accuracy", "macro_f1_all_classes", "macro_f1_supported_classes", "weighted_f1",
               "latency_p95_ms", "throughput_configured_batch_rps", "disk_size_mb", "estimated_cost_per_1m_records_usd"]
    with open(os.path.join(output_dir, "slide_ready_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("# Probability evaluation summary\n\n")
        handle.write("Temperature calibration is fitted on TEST; headline numbers below are reported on EVAL. ")
        handle.write("Macro-F1 is shown both across all mapped classes and only classes with support > 0.\n\n")
        handle.write(headline[columns].to_markdown(index=False, floatfmt=".4f"))
        handle.write("\n")
    print(f"\nCompleted {len(caches)} configurations. Outputs: {output_dir}")
    if not headline.empty:
        print("\nHeadline EVAL results:")
        print(headline[["config", "accuracy", "macro_f1_all_classes", "weighted_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
