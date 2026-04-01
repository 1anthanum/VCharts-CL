# Experiment 3B: Chart-Component Ablation

This experiment evaluates how much chart-specific visual details matter by
testing a trained model on ablated chart images (no retraining on ablations).

## Prereqs
- UCRArchive_2018 downloaded under the repo root (default path used below).
- `vtbench-env` or another environment with the repo dependencies installed.

## Quick Run
1) Activate your environment:
```
conda activate vtbench-env
```
2) (Optional) Generate ablated images first:
```
python scripts/generate_ablated_images.py --config vtbench/config/experiment_3b_ablation.yaml
```
3) Run the experiment script:
```
python scripts/experiment_3b_ablation.py --config vtbench/config/experiment_3b_ablation.yaml
```

## Outputs
Results are written to:
- `results/experiment_3b/ablation_results.csv`
- `results/experiment_3b/ablation_results.md`
- `results/experiment_3b/interpretations.txt`
- `results/experiment_3b/plots/*.png`

## Customization
Edit `vtbench/config/experiment_3b_ablation.yaml` to:
- change datasets (5-6 recommended),
- adjust chart types,
- tune ablation strength.

If you want the experiment to use saved ablated images (instead of on-the-fly),
set:
```
experiment.use_saved_ablations: true
```

If your UCRArchive is elsewhere, update:
```
experiment.dataset_root
```
