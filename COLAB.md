# Running on Google Colab

This guide explains how to run the PII column classification pipeline on Google Colab with GPU acceleration.

> **Note:** This repository is read-only and not accepting contributions. Fork the repo to experiment with your own modifications.

## Setup (One-Time)

### 1. Enable GPU Runtime
Runtime → Change runtime type → **T4 GPU**

### 2. Configure Secrets
In Colab Secrets (key icon in left sidebar), add the following with "Notebook access" enabled:

- **`ANTHROPIC_API_KEY`** - Required for synthetic data generation
  - Get from: https://console.anthropic.com/

- **`HF_TOKEN`** (Optional) - Only needed if you want to upload models to Hugging Face Hub
  - Get from: https://huggingface.co/settings/tokens
  - Needs write access

### 3. Configure Notebook Variables
Edit the configuration cell in `colab_pipeline.ipynb`:

```python
GITHUB_REPO = "username/pii-column-classifier"      # Use the original repo or your fork
HF_MODEL_REPO = "your-username/pii-column-bge-small"  # Your Hugging Face repo (optional)
RUN = "colab_v1"                                       # Experiment name
```

## Storage Architecture

**Code**: Cloned from GitHub to `/content/pipeline/` (ephemeral, lost on disconnect)

**Artifacts**: Stored on Google Drive at `/content/drive/MyDrive/policy_tag_pipeline_artifacts/`
- `data/` - Synthetic datasets and splits
- `models/` - Fine-tuned models and checkpoints
- `results/` - Evaluation metrics and reports

This ensures data/models persist across Colab sessions.

## Workflow

### Initial Setup Cell
Mounts Google Drive, clones repo, and installs dependencies:
```python
from google.colab import drive, userdata
drive.mount('/content/drive')

os.environ['PIPELINE_BASE_DIR'] = '/content/drive/MyDrive/policy_tag_pipeline_artifacts'
# ... clone and install
```

### 1. Generate Synthetic Data (Optional)
Skip if CSV already exists on Drive:
```python
!python src/synthetic_data.py --total 10000
```

For testing without API calls:
```python
!python src/synthetic_data.py --total 1500 --mock
```

### 2. Train and Compare Models
Trains all variants (TF-IDF, stock BGE-Small, fine-tuned BGE-Small) × (LinearSVC, XGBoost, ClassifierHead*):
```python
!python src/train_models.py \
    --run-name {RUN} \
    --finetune-epochs 8 \
    --finetune-batch-size 64 \
    --minilm-batch-size 256
```
*ClassifierHead only for fine-tuned variants

### 3. Evaluate Checkpoints and Probabilities
```python
!python src/evaluate_checkpoints.py --run-name {RUN} --minilm-batch-size 256
!python src/evaluate_probabilities.py --run-name {RUN} --minilm-batch-size 256
```

### 4. Upload Model to Hugging Face (Optional)
```python
from huggingface_hub import HfApi
api = HfApi(token=userdata.get('HF_TOKEN'))
api.create_repo(HF_MODEL_REPO, private=True, exist_ok=True)
api.upload_folder(
    folder_path=f"{os.environ['PIPELINE_BASE_DIR']}/models/{RUN}/ft_final",
    repo_id=HF_MODEL_REPO
)
```

## Colab-Specific Optimizations

### Batch Sizes
Colab T4 GPU has 16GB VRAM. Recommended settings:
- Fine-tuning: `--finetune-batch-size 64` (vs 32 locally)
- Embedding: `--minilm-batch-size 256` (vs 64 locally, flag name kept for compatibility)

### Memory Management
If you encounter OOM errors:
```python
import gc
gc.collect()
torch.cuda.empty_cache()
```

Or reduce batch sizes:
```bash
--finetune-batch-size 32 --minilm-batch-size 128
```

## Working with the Code

This is a read-only reference implementation. To experiment with modifications:

1. **Fork the repository** on GitHub
2. Update `GITHUB_REPO` in the notebook to point to your fork
3. Make changes in your fork and re-run the clone cell to test them

## Troubleshooting

### "No space left on device"
Colab ephemeral disk fills up. Clear it:
```bash
!rm -rf /content/pipeline/.venv
!pip cache purge
```

### "Runtime disconnected"
Artifacts on Drive are safe. Re-run setup cells to reconnect and resume.

### "API rate limit"
Synthetic data generation may hit rate limits. Use `--resume` to continue:
```python
!python src/synthetic_data.py --resume
```

### Slow fine-tuning
Verify GPU is enabled:
```python
import torch
print('GPU available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
```

## Best Practices

1. **Name experiments**: Use unique `RUN` names for each experiment
2. **Track commits**: Note the git commit for reproducibility
   ```bash
   !git rev-parse --short HEAD
   ```
3. **Save results**: Download key results from Drive before session ends
4. **Incremental runs**: Re-use existing synthetic data across experiments
5. **Monitor costs**: Colab Pro offers more compute; free tier has session limits

## Cost Comparison

**Synthetic data generation** (10,000 rows):
- ~40,000 API calls to Anthropic (20k generate + 20k label)
- Cost: ~$3-5 depending on model choices

**Fine-tuning** (5-8 epochs):
- Free on Colab with T4 GPU
- ~20-30 minutes

**Evaluation**:
- Free on Colab
- ~5-10 minutes
