# TinyViT Reproduction Study

This repository contains a PyTorch-based reproduction study of the TinyViT architecture and its training methodologies, as originally presented in the paper [**"TinyViT: Fast Pretraining Distillation for Small Vision Transformers"**](https://arxiv.org/abs/2207.10666) (ECCV 2022). This project focuses specifically on reproducing and evaluating the TinyViT-5M and TinyViT-11M variants under computationally constrained conditions.

---

## Project Overview

The goal of this project was to reproduce the core findings of the TinyViT paper, focusing on:

1. **TinyViT-5M and TinyViT-11M Base Models**: Implementing and training these lightweight ConvNet architectures from scratch.
2. **Knowledge Distillation**: Implementing the Fast Pretraining Distillation technique using precomputed teacher logits to improve the performance and training efficiency of the TinyViT models.

Experiments were conducted using subsets (10% and 30%) of the ImageNet-1K dataset and standard data augmentations due to resource limitations. We explored different teacher models, including ConvNeXt-Base, DeiT-B, and EfficientNet-B0.

---

## Key Features Implemented

- **TinyViT Architectures**:
  - `models/tinyvit.py`: TinyViT-5M
  - `models/tinyvit_11m.py`: TinyViT-11M
  - `models/blocks.py`: Core building blocks

- **Fast Distillation Framework**:
  - Sparse (top-k) teacher logits: `distillation/logit_distill.py`, `distillation/fast_distillation.py`
  - KL divergence (soft targets) + Cross-Entropy (hard targets) loss functions
  - Consistent image ID mapping: `utils/data_utils.py`

- **Training Pipelines**:
  - `train.py`: Training TinyViT-5M (baseline, distillation, CIFAR-100 transfer)
  - `train_tinyvit_11m.py`: Training TinyViT-11M (baseline, distillation with EfficientNet-B0)
  - Features: AdamW, cosine LR schedule, warmup, gradient clipping, mixed-precision (11M)

- **Evaluation**:
  - `evaluate_tinyvit_11m.py`: Evaluate trained TinyViT-11M models

- **Data Handling**:
  - `utils/data_utils.py`: Load ImageNet subsets and CIFAR-100

- **Visualization**:
  - `visualize_results.py`: training/validation curves, comparisons
  - `visualize_attention.py`: attention maps (for conceptual insights)

---

## Setup

```bash
conda env create -f environment.yaml
conda activate tinyvit
```

---

## Usage

> **Note:** Adjust paths like `--train-data`, `--output-dir`, `--logits-path` as needed.

### 1. Precompute Teacher Logits

#### For TinyViT-5M (e.g., ConvNeXt-Base)

```bash
python train.py --get-teacher --teacher-model convnext_base \
  --train-data /path/to/imagenet/subset \
  --logits-path ./logits/convnext_base_logits.pkl \
  --batch-size 64 --subset-fraction 0.1
```

#### For TinyViT-11M (EfficientNet-B0)

```bash
python train_tinyvit_11m.py --get-teacher --teacher-model efficientnet_b0 \
  --data-path /path/to/imagenet/subset \
  --logits-path ./logits/efficientnet_b0_logits_30pct.pkl \
  --batch-size 128 --subset-fraction 0.3 --device cuda
```

---

### 2. Train Baseline Models

#### TinyViT-5M on 10% ImageNet

```bash
python train.py --train-data /path/to/imagenet/subset \
  --name tinyvit_5m_baseline_10pct --batch-size 16 \
  --subset-fraction 0.1 --max-iter 300 --warmup-epochs 20
```

#### TinyViT-11M on 30% ImageNet

```bash
python train_tinyvit_11m.py --data-path /path/to/imagenet/subset \
  --name tinyvit_11m_baseline_30pct --output-dir ./output \
  --log-dir ./runs --batch-size 128 --subset-fraction 0.3 \
  --epochs 200 --warmup-epochs 20 --device cuda --mixed-precision
```

---

### 3. Train Distilled Models

#### TinyViT-5M + DeiT-B on 10% ImageNet

```bash
python train.py --train-data /path/to/imagenet/subset \
  --name tinyvit_5m_distill_deitb_10pct --batch-size 16 \
  --subset-fraction 0.1 --max-iter 300 --warmup-epochs 20 \
  --use-distillation True \
  --logits-path ./logits/deit_base_logits_10pct.pkl
```

#### TinyViT-11M + EfficientNet-B0 on 30% ImageNet

```bash
python train_tinyvit_11m.py --data-path /path/to/imagenet/subset \
  --name tinyvit_11m_distill_effb0_30pct --output-dir ./output \
  --log-dir ./runs --batch-size 128 --subset-fraction 0.3 \
  --epochs 200 --warmup-epochs 20 --use-distillation \
  --logits-path ./logits/efficientnet_b0_logits_30pct.pkl \
  --distill-alpha 0.5 --temperature 1.0 --device cuda --mixed-precision
```

---

### 4. Evaluate Models

```bash
python evaluate_tinyvit_11m.py --data-path /path/to/imagenet/subset \
  --model-path ./output/tinyvit_11m_distill_effb0_30pct_best.pth \
  --subset-fraction 0.3 --batch-size 128 --device cuda
```

---

### 5. Generate Visualizations

```bash
python visualize_results.py
python visualize_attention.py
```

---

## Results Summary

- Successfully reproduced TinyViT-5M and TinyViT-11M architectures.
- **TinyViT-5M** achieved >40% validation accuracy on 10% ImageNet despite limited setup.
- **Knowledge distillation** helped accelerate convergence and improve accuracy.
- Observed overfitting in TinyViT-5M (especially with ConvNeXt teacher) on small datasets.
- **TinyViT-11M + EfficientNet-B0** on 30% ImageNet generalized well with minimal train-val gap.
- Findings align with the paper's claims — TinyViT is competitive, but sensitive to data size, augmentations, and hyperparameters.

---

## Limitations

- **Dataset Size**: Used 10% and 30% ImageNet subsets only.
- **Augmentation**: Basic transformations only (no MixUp/CutMix/RandAugment).
- **Hardware**: Some training on CPU (e.g., MacBook Pro); main GPU used: Google Colab Pro (T4, A100).
- **Distillation**: Used only logit-level distillation (no feature-level or layer-level).

---

## Citation

If you use this work, please cite the original TinyViT paper:

```bibtex
@inproceedings{wu2022tinyvit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Peng, Jiachen and Hou, Qihang and Xiao, Yizeng and Fan, Dongfang and Geng, Zehuan and Xie, Jun},
  booktitle={European Conference on Computer Vision},
  pages={68--85},
  year={2022},
  organization={Springer}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---
