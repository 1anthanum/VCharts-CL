# VTBench — Project Instructions for Claude Code

## What This Project Is

VTBench is a Python research framework for **time-series classification (TSC) using visual chart representations**. It converts raw time series into chart images (line, bar, area, scatter) or mathematical encoding images (GASF, MTF, RP, CWT, etc.), then classifies them using CNNs — optionally fusing multiple chart types and/or raw numerical features in multimodal architectures.

**Primary use case:** Benchmarking and ablation studies comparing chart-based vs. numerical TSC on UCR/UEA archive datasets.

## How to Run

```bash
# Install (editable mode)
pip install -e .

# Run a single experiment
python scripts/experiment_6a_encodings.py --config vtbench/config/experiment_6a_encodings.yaml

# Run ALL experiments automatically (single command)
python scripts/orchestrator.py              # full suite (~18-22h)
python scripts/orchestrator.py --resume     # skip completed
python scripts/orchestrator.py --only 6b 8a # specific experiments
python scripts/orchestrator.py --dry-run    # show plan

# Pre-generate images
python scripts/pregenerate.py --all-datasets
python scripts/preflight_check.py --dry-run   # audit existing images
```

**Input:** YAML config + time-series `.ts`/`.tsv` files in `UCRArchive_2018/` or `data/`
**Output:** Trained model + metrics saved to `results/` + W&B dashboard

## Project Structure

```
vtbench/                          # Python package root
├── main.py                       # CLI entry point: parse config → train → evaluate → save
├── config/                       # YAML experiment configs (~49 files)
├── data/
│   ├── loader.py                 # read_ucr(), create_dataloaders(), train/val/test splits
│   ├── chart_generator.py        # TimeSeriesImageDataset — generates & caches chart PNGs
│   ├── ts_image_encodings.py     # GASF, GADF, MTF, RP, CWT, STFT encoding functions
│   └── image_dataset.py          # ★ EncodingImageDataset, ChartImageDataset (unified)
├── models/
│   ├── chart_models/             # SimpleCNN, DeepCNN, ResNet18, EfficientNetB0, ViTTiny
│   ├── numerical/                # NumericalFCN, NumericalTransformer, NumericalOSCNN
│   └── multimodal/               # TwoBranchModel, MultiChartModel, FusionModule
├── train/
│   ├── trainer.py                # Full pipeline training (single/two-branch/multi-chart)
│   ├── simple_trainer.py         # ★ Standalone train loop for encoding experiments
│   ├── evaluate.py               # Evaluation + metrics calculation
│   └── factory.py                # get_chart_model() factory
└── utils/
    ├── experiment_helpers.py     # ★ Shared helpers (dataset_entries, build_run_config, etc.)
    ├── wandb_logger.py           # W&B logging wrapper (no-op when wandb unavailable)
    ├── gradcam.py                # Grad-CAM visualization
    ├── ablation.py               # Chart ablation transformations
    ├── augmentations.py          # Image-level augmentations
    └── ts_augmentations.py       # Time-series-level augmentations

scripts/                          # Experiment scripts (26 total)
├── orchestrator.py               # ★ Single-command automation for full suite
├── pregenerate.py                # ★ Unified image/encoding pre-generation
├── preflight_check.py            # Audit & generate missing images
├── experiment_3a_gradcam.py      # Grad-CAM analysis
├── experiment_3b_ablation.py     # Chart ablation
├── experiment_3c_temporal_occlusion.py
├── experiment_4_augmentation_robustness.py  # Augmentation robustness
├── experiment_5a-5m_*.py         # Ablation studies (13 scripts)
├── experiment_6a_encodings.py    # Encoding method comparison (core)
├── experiment_6b_numerical_baseline.py     # Pure numerical baselines
├── experiment_7a-7c_*.py         # Advanced experiments (3 scripts)
├── experiment_8a_broad_evaluation.py       # Headline: 20 datasets × 9 encodings
└── experiment_8b_compute_profile.py        # Compute profiling

_deprecated/                      # Superseded files (safe to delete)
UCRArchive_2018/                  # Dataset directory (TSV files, not committed)
chart_images/                     # Auto-generated images (cached, not committed)
results/                          # Experiment outputs
```

★ = New files from the cleanup refactoring

## Architecture & Key Patterns

### Two Training Pipelines

1. **Chart pipeline** (`trainer.py`): Config-driven, uses `create_dataloaders()` which generates chart images on-the-fly. Used by experiment_4, experiment_5* series.

2. **Encoding pipeline** (`simple_trainer.py`): Loads pre-generated encoding images via `EncodingImageDataset`. Used by experiment_6a, 7a, 8a, 8b.

### Model Types (dispatched via `config['model']['type']`)
| Type | Config Value | What It Does |
|---|---|---|
| Single chart | `single_modal_chart` | One chart type → one CNN encoder → classify |
| Two-branch | `two_branch` | One chart + one numerical branch → fuse → classify |
| Multi-chart | `multi_modal_chart` | N chart branches ± numerical → fuse → classify |

### Available Models
- **Chart encoders:** `simplecnn` (64-dim), `deepcnn` (256-dim), `resnet18` (512-dim), `efficientnet_b0`, `vit_tiny`
- **Numerical encoders:** `fcn`, `transformer`, `oscnn`
- **Fusion modes:** `concat` (concatenate features), `weighted_sum` (learnable softmax weights)

### Shared Experiment Helpers (`vtbench/utils/experiment_helpers.py`)

All experiment_5* scripts import from this module instead of experiment_4:
- `dataset_entries(cfg)` — Parse dataset list from experiment config
- `build_run_config(cfg, dataset_entry, chart_type)` — Build single-run config
- `ensure_base_images(config)` — Generate chart images if missing
- `set_seeds(seed)` — Set all random seeds
- `evaluate_accuracy(model, loader, device)` — Compute test accuracy

### Important Implementation Details
- When `num_classes=None`, chart encoders return features only (classifier becomes `nn.Identity`) — used as branches in multimodal models
- `feature_dim` attribute on model class determines fusion dimensions. ResNet18 = 512, DeepCNN = 256, SimpleCNN falls back to 256
- DeepCNN uses `nn.LazyLinear` to avoid hardcoding flattened conv output size
- NumericalOSCNN accepts `input_channels` (channel count=1 for univariate), NOT `input_dim` (time series length)

### Data Pipeline
- `read_ucr()` reads `.ts`/`.tsv` files, normalizes to 0-indexed integers, resamples ragged series to median length
- Train/val split is **stratified from TRAIN set** (`StratifiedShuffleSplit`, `val_size=0.2`); test set is untouched
- Chart images use **global sample indices** for filenames (not split-local)
- Images are cached under `chart_images/<dataset>_images/` (or `$CHART_IMAGE_ROOT`)
- Encoding images follow: `{root}/{dataset}_images/{encoding_name}/{split}/sample_{idx}.png`

### Training
- Optimizer: Adam with `weight_decay=0.01`
- Scheduler: `ReduceLROnPlateau` (patience=3, factor=0.5) on val loss
- Early stopping: patience=10 on val accuracy
- `drop_last=True` on training loaders (avoids BatchNorm issues)

## Experiment Suite (~4,400 runs)

### Execution Plan (orchestrator.py)
| Phase | Description | Experiments | Parallel |
|-------|-------------|-------------|----------|
| A | Image pre-generation (CPU) | preflight_check.py | No |
| B1 | Numerical baselines (warmup) | 6B | No |
| B2 | Encoding comparison + ablation chain | 6A \|\| 5I→5E→5B | Yes |
| B3 | Extended encodings + re-runs | 7A \|\| 5A→5C | Yes |
| B4 | Post-processing + rendering | 7B \|\| 5D→5F | Yes |
| B5 | Chart ablation + chain | 7C \|\| 5G→5H→5J | Yes |
| B6 | TS augmentation + transfer | 5K \|\| 5M | Yes |
| B7 | Ensemble + compute profiling | 5L \|\| 8B | Yes |
| B8 | Broad dataset evaluation (headline) | 8A (solo GPU) | No |

Hardware: RTX 5070 Ti 16GB, PyTorch 2.12+cu128, W&B 0.25.1

## Config Structure

```yaml
# Experiment config (experiment_5*/6*/7*/8* scripts)
experiment:
  output_dir: results/experiment_6a_encodings
  dataset_root: UCRArchive_2018
  datasets: [GunPoint, ECG5000, FordA]
  seeds: [42, 123, 7]
  chart_types: [line, area, bar, scatter]
  # ... experiment-specific fields

model:
  chart_model: deepcnn
  pretrained: false
training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.001

# Main pipeline config (vtbench CLI)
dataset:
  name: ECG5000
  train_path: UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv
  test_path: UCRArchive_2018/ECG5000/ECG5000_TEST.tsv
model:
  type: single_modal_chart
  chart_model: resnet18
chart_branches:
  branch_1:
    chart_type: line
    color_mode: color
    label_mode: with_label
```

## Coding Conventions

- **Dispatch pattern:** Top-level functions (`train_model`, `evaluate_model`) branch on `model_type` string
- **Config access:** `config['section']['key']` with `.get(key, default)` for optional fields
- **Device:** `device = "cuda" if torch.cuda.is_available() else "cpu"` — defined at module level
- **Naming:** snake_case for functions/variables, PascalCase for classes
- **Error handling:** `raise ValueError(f"Unsupported ...")` for unknown types
- **Max line length:** 99 (flake8 config)
- **Imports:** All shared helpers via `vtbench.utils.experiment_helpers`, NOT from experiment_4

## Common Tasks

### Adding a New Chart Encoder
1. Create `vtbench/models/chart_models/<name>.py` with an `nn.Module` class
2. Set `self.feature_dim` attribute for the feature output size
3. Support `num_classes=None` → return features only (set classifier to `nn.Identity()`)
4. Register it in `vtbench/train/factory.py:get_chart_model()`

### Adding a New Encoding Method
1. Add encoding function in `vtbench/data/ts_image_encodings.py`
2. Register in `get_encoding()` dispatch function
3. Add to pregenerate.py dataset list
4. Reference in experiment config YAML

### Adding a New Dataset
1. Place `.ts`/`.tsv` files in `UCRArchive_2018/<DatasetName>/`
2. Add dataset name to experiment config YAML
3. Run `python scripts/preflight_check.py` to verify images exist

### Running Specific Experiments
```bash
python scripts/orchestrator.py --only 6b 6a     # just these two
python scripts/orchestrator.py --phase B5        # start from phase B5
python scripts/orchestrator.py --resume          # skip completed
```

## Known Quirks
- The `two_branch` trainer uses `test_chart_loaders` for validation during training (misleading variable name)
- `tn, fp, fn, tp` unpacking in `_calculate_metrics` only works for binary; for multiclass it silently uses zeros
- The `specificity` metric is only meaningful for binary classification
- `imbalanced-learn` is in requirements but not actively used in the core pipeline
- Configs have 3 tiers: `_local` (quick test), base (smoke), `_full` (production) — only `_full` variants should be used for paper results
