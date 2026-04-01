# Experiment 3A: Grad-CAM – Where Do Models Look?

This experiment generates Grad-CAM overlays for trained single-chart CNNs
to visualize which chart regions drive predictions.

## Quick Run
1) Activate your environment:
```
conda activate vtbench-env
```
2) Run the script:
```
python scripts/experiment_3a_gradcam.py --config vtbench/config/experiment_3a_gradcam.yaml
```

## Outputs
Results are written to:
- `results/experiment_3a/<dataset>/grid_class0.png`
- `results/experiment_3a/<dataset>/grid_class1.png`
- `results/experiment_3a/gradcam_manifest.csv`
- `results/experiment_3a/encoding_consistency_table.csv`
- `results/experiment_3a/dataset_notes.md`

Each dataset directory also contains per-encoding overlays under:
`results/experiment_3a/<dataset>/<encoding>/class_<0|1>/<correct|incorrect>/`

## Notes
- The grids are organized as: rows = encodings, columns = samples.
- Columns contain 10 correct examples followed by 10 incorrect examples (if available).
- `dataset_notes.md` and `encoding_consistency_table.csv` are templates to fill
  after you inspect the overlays.

## Customization
Edit `vtbench/config/experiment_3a_gradcam.yaml` to:
- change datasets,
- adjust categories,
- tweak grid size or overlay opacity,
- set `cam_target` (`pred` or `true`) and `seed`.
