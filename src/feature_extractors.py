"""
feature_extractors.py
---------------------
Feature extraction implementations for the classification pipeline.

Provides two feature representation families:
1. TF-IDF variants (char/word/combined n-grams)
2. MiniLM sentence embeddings (stock or fine-tuned)

All extractors expose a consistent fit_transform()/transform() interface,
enabling a single shared training loop in model_training.py.

TF-IDF variants:
- char: Character n-grams (2-5), robust to typos and naming conventions
- word: Word n-grams (1-2), captures semantic tokens
- combined: Both feature sets concatenated

MiniLMEmbedder is model-path-agnostic: accepts either stock base model or
fine-tuned model path. Same encoding code, different weights.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def make_tfidf_variants() -> Dict[str, object]:
    """Independent TF-IDF extractors for a fair variant comparison.

    - char: char_wb n-grams (2,5) -- robust to typos/abbreviation styles,
      largely insensitive to snake_case vs camelCase differences.
    - word: word n-grams (1,2) -- captures whole-token semantics
      (e.g. "email", "birth", "postal").
    - combined: both feature sets concatenated via FeatureUnion.
    """
    char_vectorizer = lambda: TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), lowercase=True,
        sublinear_tf=True, min_df=1, dtype=np.float32,
    )
    word_vectorizer = lambda: TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), lowercase=True,
        sublinear_tf=True, min_df=1, token_pattern=r"(?u)\b\w+\b", dtype=np.float32,
    )
    return {
        "tfidf_char": char_vectorizer(),
        "tfidf_word": word_vectorizer(),
        "tfidf_combined": FeatureUnion([("char", char_vectorizer()), ("word", word_vectorizer())]),
    }


class MiniLMEmbedder:
    """Thin wrapper giving an all-MiniLM-L6-v2 model (stock OR fine-tuned --
    just pass a different model_path) the same fit_transform/transform
    interface as the sklearn TF-IDF vectorizers above, so it can flow
    through the exact same training loop.

    Encoding is BATCHED (not one row at a time) so this scales to 10,000+
    rows -- see config.MINILM_BATCH_SIZE.
    """

    def __init__(self, model_path: str, batch_size: int, max_seq_length: int = 64, device=None):
        """`model_path` may be a Hugging Face model id (downloaded/cached on first use)
        or a local folder. Inputs are three short fields, so 64 tokens is plenty."""
        from sentence_transformers import SentenceTransformer  # local import: optional dependency
        print(f"Loading sentence-transformer from: {model_path}")
        self.model = SentenceTransformer(model_path, device=device)  # device=None -> cuda/mps/cpu auto
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _encode(self, texts: pd.Series) -> np.ndarray:
        text_list = texts.fillna("").astype(str).tolist()
        return np.asarray(
            self.model.encode(
                text_list, batch_size=self.batch_size, show_progress_bar=True,
                convert_to_numpy=True,
            )
        )

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        return self._encode(texts)

    def transform(self, texts: pd.Series) -> np.ndarray:
        return self._encode(texts)
