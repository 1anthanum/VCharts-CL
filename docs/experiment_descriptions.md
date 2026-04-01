# VTBench — Experiment Descriptions

## Phase A: Preprocessing

### Preflight
Pre-generate and validate all chart images and encoding images required by downstream experiments. Ensures cached PNGs exist for every dataset-sample-chart_type combination, preventing on-the-fly generation bottlenecks during training.

---

## Phase B: Ablation Studies (5A–5M)

### Experiment 5A — Single-Modal Chart Classification
Train CNN classifiers on individual chart types (line, bar, area, scatter) across multiple datasets and seeds to establish per-chart-type accuracy baselines for visual time-series classification.

### Experiment 5B — Two-Branch Fusion
Evaluate two-branch architectures that fuse one chart-image branch with one numerical-feature branch, measuring whether combining visual and raw numerical representations improves classification over either modality alone.

### Experiment 5C — Multi-Chart Fusion
Test multi-chart fusion models that combine multiple chart types (e.g., line + bar + area) into a single classifier, determining whether diverse visual representations provide complementary discriminative information.

### Experiment 5D — Rendering Style
Ablate chart rendering parameters including color mode (color vs. grayscale), background style, line thickness, and axis visibility to quantify how visual aesthetics affect CNN classification performance.

### Experiment 5E — Label Mode
Compare chart images rendered with axis labels, tick marks, and titles versus minimal/clean charts to measure whether textual chart annotations help or hinder CNN feature extraction.

### Experiment 5F — Data Scale
Study the effect of training set size on chart-based classification by subsampling datasets at various fractions (10%–100%), revealing data-efficiency characteristics of visual versus numerical approaches.

### Experiment 5G — Chart Type
Systematic comparison across all four chart types (line, bar, area, scatter) on diverse datasets to identify which visual representation best preserves time-series discriminative features per domain.

### Experiment 5H — Resolution Sweep
Evaluate image resolution impact (64×64, 128×128, 224×224, 336×336) on classification accuracy and compute cost, identifying the resolution sweet spot that balances performance and GPU memory.

### Experiment 5I — Backbone Comparison
Benchmark five CNN backbones (SimpleCNN, DeepCNN, ResNet18, EfficientNet-B0, ViT-Tiny) on chart images to determine which architecture best extracts discriminative features from visual time-series representations.

### Experiment 5J — Training Strategy
Ablate training hyperparameters including learning rate schedules, optimizers (Adam, SGD), weight decay, and early stopping patience to identify robust training configurations for chart-based classifiers.

### Experiment 5K — Time-Series Augmentation
Apply time-series-level augmentations (jitter, scaling, time-warp, window-crop) before chart rendering, measuring whether augmenting the raw signal improves visual classification robustness and generalization.

### Experiment 5L — Ensemble Learning
Combine predictions from multiple chart-type classifiers via voting and weighted ensembles, testing whether aggregating diverse visual perspectives yields higher accuracy than any single chart type.

### Experiment 5M — Transfer Learning
Pre-train chart classifiers on a large source dataset (ECG5000/FordA), then fine-tune on smaller target datasets to measure cross-domain transferability of learned visual time-series features.

---

## Phase B: Encoding Methods (6A–6B)

### Experiment 6A — Encoding Method Comparison
Compare nine mathematical encoding methods (GASF, GADF, MTF, RP, CWT, STFT, phase-space combinations) that transform time series into images, benchmarking each encoding's classification accuracy across datasets.

### Experiment 6B — Numerical Baselines
Train purely numerical models (FCN, Transformer, OS-CNN) directly on raw time-series values without any visual representation, establishing baseline accuracies for fair comparison against chart-based approaches.

---

## Phase B: Extended Experiments (7A–7C)

### Experiment 7A — Extended Encodings
Test additional and combined encoding methods beyond the core set, including multi-channel composites (e.g., GASF+RP overlay) and frequency-domain combinations to push encoding diversity further.

### Experiment 7B — Image Post-Processing
Apply image-level post-processing (histogram equalization, CLAHE, edge enhancement, Gaussian blur) to encoding images before classification, measuring whether standard CV preprocessing improves encoding-based accuracy.

### Experiment 7C — Chart Ablation
Systematically remove or occlude chart components (axes, gridlines, data points, fill areas) to identify which visual elements CNNs actually rely on for classification via controlled ablation.

---

## Phase B: Broad Evaluation (8A–8B)

### Experiment 8A — Broad Dataset Evaluation (Headline)
The flagship experiment: evaluate the top-performing encodings and models across 20 UCR datasets with 5 seeds each, producing the main results table for publication-ready accuracy comparison.

### Experiment 8B — Compute Profiling
Measure end-to-end wall-clock time, GPU memory, and throughput for each encoding-model combination, providing a cost-performance analysis to accompany accuracy results in the paper.

---

## Phase C: Advanced Methods (9A)

### Experiment 9A — Advanced Methods
Evaluate accuracy improvement techniques: new backbones (EfficientNet-B0, ViT-Tiny), new encodings (wavelet scattering, signature, persistence), training augmentations (Mixup, CutMix), test-time augmentation (TTA), and channel attention (SE-Net, CBAM).

---

## Cloud-Only Tasks

### Cloud: 5H Full Resolution
Run the full resolution sweep (including 224×224 and 336×336) on a cloud GPU with ≥24GB VRAM, which exceeds local RTX 5070 Ti memory limits for large-batch high-resolution training.

### Cloud: 6A Wafer Remaining
Complete the 23 remaining Wafer dataset runs (seed=7) that repeatedly triggered GPU driver crashes on the local machine, using stable cloud hardware to fill this gap.

### Cloud: 7A Large Datasets
Run extended encoding experiments on FordA and Wafer — large datasets excluded from local execution due to GPU instability under sustained high-memory training loads.

### Cloud: 8A Large Datasets
Execute the broad evaluation on Wafer, TwoPatterns, and FordA — the three largest datasets that require extended training time and higher VRAM than local hardware reliably supports.

### Cloud: 8B FordA Profiling
Profile compute performance specifically on FordA (3,601 training samples), capturing accurate timing data for the largest dataset that local hardware cannot process without crash risk.

### Cloud: 5M FordA Transfer
Use FordA as the source domain for transfer learning experiments, pre-training on its large training set then fine-tuning on four smaller target datasets to test large-source transferability.
