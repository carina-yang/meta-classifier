"""
model_training.py
-----------------
Generic training and evaluation logic for all feature × classifier combinations.

Trains and evaluates:
- TF-IDF variants (char/word/combined) × LinearSVC/XGBoost
- MiniLM embeddings (stock/fine-tuned) × LinearSVC/XGBoost

Single shared training loop (run_feature_variant) eliminates code duplication.
Adding new features or classifiers requires no copy-pasting.

Classifiers:
- LinearSVC: Linear SVM with balanced class weights
- XGBoost: Gradient boosting with multi-class softmax
"""

import hashlib
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from config import XGB_NUM_ROUND, XGB_PARAMS

try:
    import xgboost as xgb
    HAS_XGB = True
except (ImportError, Exception):
    xgb = None
    HAS_XGB = False


def safe_transform(extractor, texts: pd.Series, n_features: int, is_sparse: bool):
    """Transform `texts` with `extractor`, WITHOUT calling the underlying
    vectorizer/embedder on an empty input. TfidfVectorizer.transform([])
    (and some embedding models) raise on 0 rows -- this can happen for very
    small/rare datasets where an entire split ends up empty (see
    data_splitting.py's per-class rare-label handling). Returns a
    correctly-shaped 0-row matrix in that case instead of crashing.
    """
    if len(texts) == 0:
        return sparse.csr_matrix((0, n_features)) if is_sparse else np.empty((0, n_features))
    return extractor.transform(texts)


def fit_svm(x_train, y_train, seed: int) -> LinearSVC:
    """Fit and return a LinearSVC classifier."""
    classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
    classifier.fit(x_train, y_train)
    return classifier


def fit_xgboost(x_train, y_train, num_classes: int, seed: int):
    """Fit and return an XGBoost booster."""
    if not HAS_XGB:
        raise ImportError("XGBoost is not available")
    params = dict(XGB_PARAMS)
    params["num_class"] = num_classes
    params["seed"] = seed
    params.setdefault("nthread", -1)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=XGB_NUM_ROUND)
    return booster


def train_predict_svm(x_train, y_train, x_test, x_eval, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit LinearSVC and return predictions for both the held-out test and
    final eval sets. Empty test/eval matrices (0 rows) short-circuit to an
    empty prediction array rather than calling predict() on nothing."""
    classifier = fit_svm(x_train, y_train, seed)
    test_pred = classifier.predict(x_test) if x_test.shape[0] else np.array([], dtype=int)
    eval_pred = classifier.predict(x_eval) if x_eval.shape[0] else np.array([], dtype=int)
    return test_pred, eval_pred


def train_predict_xgboost(x_train, y_train, x_test, x_eval, num_classes: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit native XGBoost with config.py parameters and return predictions
    for both the held-out test and final eval sets. Empty test/eval
    matrices (0 rows) short-circuit to an empty prediction array rather than
    building a 0-row DMatrix."""
    booster = fit_xgboost(x_train, y_train, num_classes, seed)

    def _predict(x):
        if x.shape[0] == 0:
            return np.array([], dtype=int)
        raw = np.asarray(booster.predict(xgb.DMatrix(x)))
        return np.argmax(raw, axis=1) if raw.ndim == 2 else raw.astype(int)

    return _predict(x_test), _predict(x_eval)


def fit_classifier_head(embedder, classification_head_path: str):
    """Load the pre-trained classification head from fine-tuning.

    Unlike LinearSVC and XGBoost, the classification head is NOT fitted here.
    It was already trained during fine-tuning. This just loads it.

    Args:
        embedder: The MiniLMEmbedder instance (needed for inference)
        classification_head_path: Path to saved classification head

    Returns:
        Tuple of (embedder, classification_head)
    """
    import torch
    import json
    import os

    # Load the classification head definition
    import sys
    sys.path.append(os.path.dirname(__file__))
    from sentence_transformer_finetuning import _build_model_classes
    ClassificationHead, _ = _build_model_classes()

    # Load metadata
    metadata_path = os.path.join(classification_head_path, "head_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Reconstruct head
    head = ClassificationHead(metadata["embedding_dim"], metadata["num_classes"])

    # Load weights
    weights_path = os.path.join(classification_head_path, "head_weights.pt")
    head.load_state_dict(torch.load(weights_path, map_location="cpu"))
    head.eval()

    return embedder, head


def train_predict_classifier_head(embedder, head, x_train, y_train, x_test, x_eval):
    """Use pre-trained classification head for predictions.

    The head was already trained during fine-tuning, so no training happens here.
    This is different from LinearSVC/XGBoost which are trained on the embeddings.
    """
    import torch
    import numpy as np

    device = next(head.parameters()).device if hasattr(head, 'parameters') else torch.device('cpu')
    head = head.to(device)

    def _predict(x):
        if x.shape[0] == 0:
            return np.array([], dtype=int)

        # Convert embeddings to predictions
        with torch.no_grad():
            x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)
            # Wrap in expected format for classification head
            features = {"sentence_embedding": x_tensor}
            logits = head(features)
            preds = torch.argmax(logits, dim=1)
            return preds.cpu().numpy()

    test_pred = _predict(x_test)
    eval_pred = _predict(x_eval)
    return test_pred, eval_pred


def safe_slug(value: str) -> str:
    """Turn a display value into a stable output-file component."""
    return value.lower().replace("+", "_").replace("-", "_").replace(" ", "_")


def _compute_model_fingerprint(model_path: str) -> str:
    """Compute a fingerprint for a model directory based on path and file metadata.

    For HuggingFace model IDs, uses the ID itself.
    For local directories, hashes the path and file metadata (names, sizes, mtimes).
    """
    if not os.path.isdir(model_path):
        # Hugging Face model ID or non-existent path
        return hashlib.sha1(model_path.encode('utf-8')).hexdigest()[:12]

    # Local directory: hash path + file metadata
    hasher = hashlib.sha1()
    hasher.update(model_path.encode('utf-8'))

    try:
        for root, _, files in sorted(os.walk(model_path)):
            for name in sorted(files):
                fpath = os.path.join(root, name)
                try:
                    stat = os.stat(fpath)
                    hasher.update(name.encode('utf-8'))
                    hasher.update(str(stat.st_size).encode('utf-8'))
                    hasher.update(str(int(stat.st_mtime)).encode('utf-8'))
                except OSError:
                    pass
    except OSError:
        pass

    return hasher.hexdigest()[:12]


def _save_extractor(extractor: Any, variant_name: str, classifiers_dir: str) -> Tuple[str, str, Optional[str]]:
    """Save an extractor and return (extractor_path, extractor_type, model_path_for_json).

    For TF-IDF: saves as joblib, returns ("path.joblib", "tfidf", None)
    For MiniLM: saves JSON metadata, returns ("path.json", "minilm", model_path)
    """
    if variant_name.startswith("tfidf"):
        extractor_path = os.path.join(classifiers_dir, f"{variant_name}_vectorizer.joblib")
        joblib.dump(extractor, extractor_path)
        return extractor_path, "tfidf", None
    elif variant_name.startswith("minilm"):
        # MiniLM: save JSON metadata instead of pickling the model
        extractor_path = os.path.join(classifiers_dir, f"{variant_name}_extractor.json")
        model_path = extractor.model.model_card_data.model_id if hasattr(extractor.model, 'model_card_data') else str(extractor.model.model_name_or_path) if hasattr(extractor.model, 'model_name_or_path') else "unknown"

        # For local models, use the actual path
        if hasattr(extractor.model, '_model_card_data'):
            model_path = getattr(extractor.model, '_model_card_data', {}).get('model_id', model_path)

        # Try to get the actual model path used
        try:
            if hasattr(extractor.model, 'model_name_or_path'):
                model_path = str(extractor.model.model_name_or_path)
            elif hasattr(extractor.model, '_modules') and hasattr(extractor.model._modules.get('0', None), 'auto_model'):
                auto_model = extractor.model._modules['0'].auto_model
                if hasattr(auto_model, 'name_or_path'):
                    model_path = str(auto_model.name_or_path)
        except:
            pass

        metadata = {
            "type": "minilm",
            "model_path": model_path,
            "batch_size": extractor.batch_size,
            "max_seq_length": extractor.model.max_seq_length,
        }
        with open(extractor_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        return extractor_path, "minilm", model_path
    else:
        raise ValueError(f"Unknown variant type: {variant_name}")


def _cache_features(features: np.ndarray, variant_name: str, split: str,
                    model_path: Optional[str], cache_dir: str) -> Optional[str]:
    """Cache MiniLM features to .npy file with model fingerprint.

    Returns cache file path or None if not MiniLM variant.
    """
    if not variant_name.startswith("minilm") or model_path is None:
        return None

    fingerprint = _compute_model_fingerprint(model_path)
    cache_file = os.path.join(cache_dir, f"{variant_name}_{split}_{fingerprint}.npy")
    np.save(cache_file, features.astype(np.float32))
    return cache_file


def _rss_mb() -> float:
    """Get current RSS memory in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except ImportError:
        import resource
        import sys
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return value / (1024 ** 2)
        return value / 1024


def save_evaluation(
    results_dir: str, variant: str, classifier: str,
    y_test: np.ndarray, test_pred: np.ndarray,
    y_eval: np.ndarray, eval_pred: np.ndarray,
    idx_to_tag: List[str],
) -> str:
    """Print + save one classification report and one confusion-matrix CSV
    (covering BOTH the held-out test split and the final eval split) for one
    feature-variant/classifier configuration."""
    import os

    labels = list(range(len(idx_to_tag)))
    slug = f"{safe_slug(variant)}_{safe_slug(classifier)}"

    header = f"Feature variant: {variant}\nClassifier: {classifier}\n\n"
    if len(y_test):
        test_report = classification_report(
            y_test, test_pred, labels=labels, target_names=idx_to_tag, digits=4, zero_division=0,
        )
        test_section = (
            "=== HELD-OUT TEST SPLIT ===\n"
            f"Accuracy: {accuracy_score(y_test, test_pred):.4f}\n\n{test_report}\n"
        )
    else:
        test_section = "=== HELD-OUT TEST SPLIT ===\n(test split was empty)\n"
    if len(y_eval):
        eval_report = classification_report(
            y_eval, eval_pred, labels=labels, target_names=idx_to_tag, digits=4, zero_division=0,
        )
        eval_section = (
            "=== FINAL EVAL SPLIT ===\n"
            f"Accuracy: {accuracy_score(y_eval, eval_pred):.4f}\n\n{eval_report}"
        )
    else:
        eval_section = "=== FINAL EVAL SPLIT ===\n(eval split was empty)"
    report_text = header + test_section + eval_section

    report_path = os.path.join(results_dir, f"classification_report_{slug}.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    matrix_path = os.path.join(results_dir, f"confusion_matrix_{slug}.csv")
    with open(matrix_path, "w", encoding="utf-8", newline="") as file:
        if len(y_test):
            file.write("evaluation_set,held_out_test\n")
            pd.DataFrame(
                confusion_matrix(y_test, test_pred, labels=labels),
                index=[f"true_{t}" for t in idx_to_tag], columns=[f"pred_{t}" for t in idx_to_tag],
            ).to_csv(file)
        else:
            file.write("evaluation_set,held_out_test\n(test split was empty)\n")
        if len(y_eval):
            file.write("\nevaluation_set,final_eval\n")
            pd.DataFrame(
                confusion_matrix(y_eval, eval_pred, labels=labels),
                index=[f"true_{t}" for t in idx_to_tag], columns=[f"pred_{t}" for t in idx_to_tag],
            ).to_csv(file)
    return report_path


def run_feature_variant(
    variant_name: str,
    extractor,
    train_text: pd.Series, test_text: pd.Series, eval_text: pd.Series,
    y_train: np.ndarray, y_test: np.ndarray, y_eval: np.ndarray,
    idx_to_tag: List[str], results_dir: str,
    run_xgboost: bool, xgb_dense_max_mb: int, seed: int,
    classifiers_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[dict]:
    """Fit ONE feature extractor once, then train/evaluate every requested
    classifier on top of it. This is the single reusable loop shared by
    every TF-IDF variant AND both MiniLM embedding pipelines (stock and
    fine-tuned) -- no per-feature duplicated training code.

    If classifiers_dir is provided, saves all fitted artifacts:
    - Extractors (TF-IDF as joblib, MiniLM as JSON)
    - Classifiers (LinearSVC as joblib, XGBoost as JSON)
    - Training metadata (fit time, memory, etc.)
    - Feature cache for MiniLM variants
    """
    print(f"\n=== Feature variant: {variant_name} ===")
    feature_start = time.time()
    before_rss = _rss_mb() if classifiers_dir else 0.0

    x_train = extractor.fit_transform(train_text)
    is_sparse = sparse.issparse(x_train)
    n_features = x_train.shape[1]
    x_test = safe_transform(extractor, test_text, n_features, is_sparse)
    x_eval = safe_transform(extractor, eval_text, n_features, is_sparse)
    feature_time = time.time() - feature_start
    print(f"Feature matrix: {x_train.shape[0]} rows x {x_train.shape[1]} features "
          f"(built in {feature_time:.1f}s)")

    # Save extractor once per variant
    extractor_path, extractor_type, model_path = None, None, None
    feature_cache_paths = {}
    if classifiers_dir:
        os.makedirs(classifiers_dir, exist_ok=True)
        extractor_path, extractor_type, model_path = _save_extractor(extractor, variant_name, classifiers_dir)

        # Cache MiniLM features
        if cache_dir and variant_name.startswith("minilm"):
            os.makedirs(cache_dir, exist_ok=True)
            for split, features in [("train", x_train), ("test", x_test), ("eval", x_eval)]:
                cache_path = _cache_features(features, variant_name, split, model_path, cache_dir)
                if cache_path:
                    feature_cache_paths[split] = cache_path

    classifiers: List[Tuple[str, Callable]] = [("LinearSVC", None)]
    if run_xgboost:
        n_rows, n_features_check = x_train.shape
        itemsize = np.dtype(np.float32).itemsize
        estimated_dense_mb = (n_rows * n_features_check * itemsize) / (1024 ** 2)
        if estimated_dense_mb > xgb_dense_max_mb:
            print(f"Skipping XGBoost for '{variant_name}': dense-equivalent size "
                  f"{estimated_dense_mb:.1f} MB exceeds {xgb_dense_max_mb} MB threshold.")
        else:
            classifiers.append(("XGBoost", None))

    # Check if this is a fine-tuned MiniLM variant with saved classification head
    if variant_name == "minilm_finetuned" and classifiers_dir:
        # Look for saved classification head
        head_path = os.path.join(os.path.dirname(classifiers_dir), "ft_head")
        if os.path.exists(head_path):
            classifiers.append(("ClassifierHead", head_path))

    rows = []
    manifest_entries = []
    for classifier_name, classifier_arg in classifiers:
        slug = f"{variant_name}_{classifier_name.lower()}"
        print(f"--- Training {variant_name} + {classifier_name} ---")

        classifier_start = time.time()
        if classifier_name == "ClassifierHead":
            # Special case: Load pre-trained head instead of training
            embedder, head = fit_classifier_head(extractor, classifier_arg)
            test_pred, eval_pred = train_predict_classifier_head(
                embedder, head, x_train, y_train, x_test, x_eval
            )
            classifier = head  # For saving
        elif classifier_name == "LinearSVC":
            classifier = fit_svm(x_train, y_train, seed)
            test_pred = classifier.predict(x_test) if x_test.shape[0] else np.array([], dtype=int)
            eval_pred = classifier.predict(x_eval) if x_eval.shape[0] else np.array([], dtype=int)
        else:  # XGBoost
            classifier = fit_xgboost(x_train, y_train, len(idx_to_tag), seed)
            def _predict(x):
                if x.shape[0] == 0:
                    return np.array([], dtype=int)
                raw = np.asarray(classifier.predict(xgb.DMatrix(x)))
                return np.argmax(raw, axis=1) if raw.ndim == 2 else raw.astype(int)
            test_pred = _predict(x_test)
            eval_pred = _predict(x_eval)

        classifier_fit_time = time.time() - classifier_start
        total_fit_time = feature_time + classifier_fit_time

        test_acc = accuracy_score(y_test, test_pred) if len(y_test) else float("nan")
        eval_acc = accuracy_score(y_eval, eval_pred) if len(y_eval) else float("nan")
        test_acc_str = f"{test_acc:.2%}" if len(y_test) else "N/A (empty)"
        eval_acc_str = f"{eval_acc:.2%}" if len(y_eval) else "N/A (empty)"
        print(f"Held-out test accuracy: {test_acc_str} | Final eval accuracy: {eval_acc_str}")

        save_evaluation(results_dir, variant_name, classifier_name, y_test, test_pred, y_eval, eval_pred, idx_to_tag)

        # Save classifier and metadata
        classifier_path = None
        metadata_path = None
        if classifiers_dir:
            # Save classifier
            if classifier_name == "ClassifierHead":
                # Classification head already saved during fine-tuning
                # Just save a reference to it
                classifier_path = os.path.join(classifiers_dir, f"{slug}_classifier_ref.json")
                with open(classifier_path, "w", encoding="utf-8") as f:
                    json.dump({"head_path": classifier_arg}, f, indent=2)
            elif classifier_name == "LinearSVC":
                classifier_path = os.path.join(classifiers_dir, f"{slug}_classifier.joblib")
                joblib.dump(classifier, classifier_path)
            else:  # XGBoost
                classifier_path = os.path.join(classifiers_dir, f"{slug}_classifier.json")
                classifier.save_model(classifier_path)

            # Save metadata
            after_rss = _rss_mb()
            memory_delta = max(0.0, after_rss - before_rss)
            metadata = {
                "fit_time_seconds": total_fit_time,
                "feature_time_seconds": feature_time,
                "classifier_fit_time_seconds": classifier_fit_time,
                "memory_rss_delta_mb": memory_delta,
                "n_features": int(n_features),
                "random_seed": seed,
            }
            metadata_path = os.path.join(classifiers_dir, f"{slug}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            # Add to manifest
            manifest_entry = {
                "variant": variant_name,
                "classifier": classifier_name,
                "slug": slug,
                "extractor_path": os.path.relpath(extractor_path, classifiers_dir) if extractor_path else None,
                "extractor_type": extractor_type,
                "classifier_path": os.path.relpath(classifier_path, classifiers_dir) if classifier_path else None,
                "metadata_path": os.path.relpath(metadata_path, classifiers_dir) if metadata_path else None,
                "feature_cache": {k: os.path.relpath(v, classifiers_dir) for k, v in feature_cache_paths.items()} if feature_cache_paths else None,
            }
            manifest_entries.append(manifest_entry)

        rows.append({
            "feature_variant": variant_name,
            "classifier": classifier_name,
            "model": f"{variant_name} + {classifier_name}",
            "test_accuracy": test_acc,
            "eval_accuracy": eval_acc,
            "n_features": x_train.shape[1] if hasattr(x_train, "shape") else None,
            "_manifest_entries": manifest_entries if classifiers_dir else [],
        })

    return rows
