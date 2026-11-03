"""
sentence_transformer_finetuning.py
----------------------------------
Fine-tunes the all-MiniLM-L6-v2 sentence transformer on training data for
improved column classification embeddings.

Approach:
1. Load base sentence transformer model
2. Attach temporary linear classification head for training signal
3. Train with cross-entropy loss and AdamW optimizer
4. Save checkpoint after every epoch (for evaluation via evaluate_checkpoints.py)
5. Discard classification head, keep fine-tuned embeddings

The classification head is used ONLY during fine-tuning. Downstream models
(LinearSVC/XGBoost) train on the embeddings, not this head.

Learning Rate Schedule:
- Warmup: Linear ramp from 0 to peak LR over first 10% of steps (configurable)
- Decay: Linear or cosine decay from peak to ~0 over remaining steps
- `learning_rate` parameter specifies the PEAK rate, not a constant rate

Per-epoch checkpoints enable empirical evaluation of whether additional
epochs would improve downstream classification accuracy.
    - `scheduler.step()` is called every step, right after
      `optimizer.step()` -- unchanged from before.
    - The current LR is printed at every epoch boundary (alongside the
      existing avg-loss print) and saved into the same loss-history JSON
      structure, so you can visually/programmatically confirm the LR is
      actually decaying during training.

Model Loading:
The base model can be loaded from either:
- A local folder path (via BASE_MODEL_PATH env var or --minilm-model-path flag)
- HuggingFace Hub (default: "sentence-transformers/all-MiniLM-L6-v2")

Local loading is useful for offline usage, reproducibility, or version control.

Requirements (optional -- only needed if you don't pass --skip-finetune):
    pip install torch sentence-transformers transformers
"""

import json
import math
import os
import shutil
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _build_model_classes():
    """Lazily import torch + build the two small nn.Module classes used only
    during fine-tuning. Kept inside a function so torch stays an optional
    dependency, matching the pattern already used by MiniLMEmbedder in
    feature_extractors.py."""
    import torch.nn as nn

    class _ClassificationHead(nn.Module):
        def __init__(self, embedding_dim, num_classes):
            super().__init__()
            self.linear = nn.Linear(embedding_dim, num_classes)

        def forward(self, features):
            x = features["sentence_embedding"]
            x = self.linear(x)
            return x

    class _SentenceTransformerWithHead(nn.Module):
        """Wraps the base sentence transformer + classification head
        together so we can fine-tune both in one training loop."""

        def __init__(self, transformer, head):
            super().__init__()
            self.transformer = transformer
            self.head = head

        def forward(self, input_ids_and_mask):
            features = self.transformer(input_ids_and_mask)
            logits = self.head(features)
            return logits

    return _ClassificationHead, _SentenceTransformerWithHead


def _collate_fn(batch):
    import torch
    texts = [example.texts[0] for example in batch]
    labels = torch.tensor([example.label for example in batch])
    return texts, labels


def _save_classification_head(head, num_classes, embedding_dim, save_path):
    """Save classification head as a standalone artifact.

    Saves both the model state dict and metadata needed for reconstruction.
    """
    import torch
    import json
    import os

    os.makedirs(save_path, exist_ok=True)

    # Save model weights
    torch.save(head.state_dict(), os.path.join(save_path, "head_weights.pt"))

    # Save metadata for reconstruction
    metadata = {
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
    }
    with open(os.path.join(save_path, "head_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _build_warmup_decay_scheduler(optimizer, scheduler_type: str, num_warmup_steps: int, num_training_steps: int):
    """Return a step-based LR scheduler that linearly warms up from 0 -> the
    optimizer's initial ("peak") LR over `num_warmup_steps`, then decays
    (linearly or via a half-cosine curve) down to ~0 by `num_training_steps`.

    Prefers transformers' own helpers (already a project dependency); falls
    back to an equivalent hand-built torch.optim.lr_scheduler.LambdaLR if
    transformers' scheduler helpers can't be imported for any reason.
    """
    scheduler_type = (scheduler_type or "linear").lower()
    if scheduler_type not in ("linear", "cosine"):
        print(f"[fine-tune] WARNING: unknown lr_scheduler_type '{scheduler_type}', defaulting to 'linear'.")
        scheduler_type = "linear"

    try:
        from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
        if scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
            )
        print(f"[fine-tune] Using transformers.get_{scheduler_type}_schedule_with_warmup "
              f"({num_warmup_steps}/{num_training_steps} warmup steps).")
        return scheduler
    except ImportError as exc:
        print(f"[fine-tune] NOTE: could not import transformers scheduler helpers ({exc}); "
              "building an equivalent schedule manually with torch.optim.lr_scheduler.LambdaLR.")
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            progress = min(1.0, progress)
            if scheduler_type == "cosine":
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return max(0.0, 1.0 - progress)  # linear decay to 0

        return LambdaLR(optimizer, lr_lambda)


def _evaluate_val(model, model_with_head, criterion, val_df, text_column, label_column, device, batch_size=128):
    import torch
    model_with_head.eval()
    texts = val_df[text_column].astype(str).tolist()
    labels = torch.tensor(val_df[label_column].astype(int).tolist())
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            inputs = model.tokenize(texts[i:i + batch_size])
            feats = {"input_ids": inputs["input_ids"].to(device),
                     "attention_mask": inputs["attention_mask"].to(device)}
            y = labels[i:i + batch_size].to(device)
            logits = model_with_head(feats)
            total_loss += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
    return total_loss / len(texts), correct / len(texts)


def fine_tune_sentence_transformer(
    base_model_path: str,
    train_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    num_classes: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_dir_template: str,
    final_model_dir: str,
    seed: int = 42,
    loss_history_json: Optional[str] = None,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "linear",
    device: Optional[str] = None,
    val_df: Optional[pd.DataFrame] = None,
    select_best: bool = True,
    max_seq_length: int = 64,
) -> str:
    """Fine-tune `base_model_path` on `train_df[text_column]` ->
    `train_df[label_column]`, following the original pilot's
    train_sentence_transformer.py training loop, with a warmup+decay LR
    schedule layered on top (see module docstring). Saves a checkpoint after
    every epoch to `checkpoint_dir_template.format(epoch=N)`, plus the final
    model to `final_model_dir`.

    `learning_rate` is now the PEAK learning rate, reached at the end of the
    warmup phase (the first `warmup_ratio` fraction of total steps), after
    which the LR decays (per `lr_scheduler_type`) to ~0 by the final step.

    Returns `final_model_dir` (so callers can immediately load it, e.g. into
    feature_extractors.MiniLMEmbedder, without hard-coding the path twice).
    """
    import torch
    from sentence_transformers import SentenceTransformer, InputExample
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    # base_model_path can be Hugging Face ID or local folder path
    ClassificationHead, SentenceTransformerWithHead = _build_model_classes()

    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                  else "cpu")
    device = torch.device(device)
    print(f"[fine-tune] Using device: {device}")
    print(f"[fine-tune] Loaded {len(train_df)} training rows.")

    # Load base model
    model = SentenceTransformer(base_model_path)
    model.max_seq_length = max_seq_length
    model = model.to(device)

    # Attach classification head
    embedding_dim = (getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension)()
    print(f"[fine-tune] Embedding dimension: {embedding_dim}, num_classes: {num_classes}")
    classification_head = ClassificationHead(embedding_dim, num_classes)
    model_with_head = SentenceTransformerWithHead(model, classification_head).to(device)

    # Build training dataloader
    torch.manual_seed(seed)
    train_examples = [
        InputExample(texts=[row[text_column]], label=int(row[label_column]))
        for _, row in train_df.iterrows()
    ]
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size, collate_fn=_collate_fn,
    )

    # Loss, optimizer, and LR scheduler
    criterion = torch.nn.CrossEntropyLoss()
    # learning_rate is the peak LR reached after warmup
    optimizer = AdamW(model_with_head.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = max(1, int(round(total_steps * warmup_ratio))) if warmup_ratio > 0 else 0
    print(f"[fine-tune] LR schedule: peak_lr={learning_rate:.2e}, scheduler='{lr_scheduler_type}', "
          f"warmup_ratio={warmup_ratio:.2f} ({num_warmup_steps}/{total_steps} steps).")
    scheduler = _build_warmup_decay_scheduler(optimizer, lr_scheduler_type, num_warmup_steps, total_steps)

    # Training loop - track loss and LR per epoch
    loss_history: List[dict] = []
    best_val_acc, best_epoch = -1.0, None
    for epoch in range(num_epochs):
        model_with_head.train()
        epoch_losses = []
        for step, (texts, labels) in enumerate(train_dataloader):
            labels = labels.to(device)
            optimizer.zero_grad()
            inputs = model.tokenize(texts)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            inputs_final = {"input_ids": input_ids, "attention_mask": attention_mask}
            logits = model_with_head(inputs_final)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
            if step % 50 == 0:
                current_step_lr = scheduler.get_last_lr()[0]
                print(f"[fine-tune] Epoch {epoch + 1}/{num_epochs}, Step {step}/{len(train_dataloader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_step_lr:.3e}")
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_end_lr = scheduler.get_last_lr()[0]
        entry = {"loss": avg_epoch_loss, "learning_rate": epoch_end_lr}
        if val_df is not None and len(val_df):
            val_loss, val_acc = _evaluate_val(model, model_with_head, criterion, val_df,
                                              text_column, label_column, device)
            entry.update(val_loss=val_loss, val_accuracy=val_acc)
            print(f"[fine-tune]     Validation: loss={val_loss:.4f}, accuracy={val_acc:.2%}")
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
        loss_history.append(entry)
        print(f"[fine-tune] >>> Epoch {epoch + 1}/{num_epochs} complete. "
              f"Avg loss: {avg_epoch_loss:.4f} | LR at epoch end: {epoch_end_lr:.3e}")
        # Save checkpoint after each epoch
        checkpoint_path = checkpoint_dir_template.format(epoch=epoch)
        model.save(checkpoint_path)
        print(f"[fine-tune]     Saved checkpoint: {checkpoint_path}")
        # Early-stopping hint: warn if loss plateaus for 3+ epochs
        recent_losses = [entry["loss"] for entry in loss_history]
        if len(recent_losses) >= 4 and min(recent_losses[-3:]) >= recent_losses[-4]:
            print(
                "[fine-tune]     NOTE: Loss hasn't improved over the last 3 epochs. "
                "You may want to stop here (Ctrl+C) and reuse this checkpoint via "
                "--minilm-model-path, or let it finish."
            )

    # Save final model (best validation epoch if available)
    if select_best and best_epoch is not None:
        if os.path.isdir(final_model_dir):
            shutil.rmtree(final_model_dir)
        shutil.copytree(checkpoint_dir_template.format(epoch=best_epoch), final_model_dir)
        print(f"\n[fine-tune] Selected best epoch {best_epoch + 1} (val acc {best_val_acc:.2%}) -> {final_model_dir}")
    else:
        model.save(final_model_dir)
        print(f"\n[fine-tune] Final (last-epoch) model saved to: {final_model_dir}")

    # Save classification head
    head_save_path = final_model_dir.replace("ft_final", "ft_head")
    _save_classification_head(classification_head, num_classes, embedding_dim, head_save_path)
    print(f"[fine-tune] Classification head saved to: {head_save_path}")

    print("[fine-tune] Loss / LR per epoch:")
    for i, entry in enumerate(loss_history):
        print(f"[fine-tune]   Epoch {i + 1}: loss={entry['loss']:.4f}, lr={entry['learning_rate']:.3e}")

    if loss_history_json:
        # Save as {epoch_index: {"loss": ..., "learning_rate": ...}}
        os.makedirs(os.path.dirname(loss_history_json), exist_ok=True)
        with open(loss_history_json, "w", encoding="utf-8") as f:
            json.dump({str(i): entry for i, entry in enumerate(loss_history)}, f, indent=2)
        print(f"[fine-tune] Saved loss/LR history: {loss_history_json}")

    return final_model_dir
