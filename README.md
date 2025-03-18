TinyViT Reproduction



This repository contains a PyTorch implementation of the TinyViT architecture and fast distillation framework as described in the paper "TinyViT: Fast Pretraining Distillation for Small Vision Transformers" (ECCV 2022).



Implementation of TinyViT-5M Architecture



Successfully implemented the TinyViT-5M architecture as described in the paper. The implementation includes:



Building Blocks (models/blocks.py):

PatchEmbed: Converts images to patch embeddings

PatchMerging: Hierarchical feature map downsampling

MBConv: Mobile Inverted Bottleneck blocks for early stages

TransformerBlock: Standard transformer blocks with attention and MLP

Attention: Multi-head self-attention mechanism

Mlp: MLP blocks used in transformer layers



Main Architecture (models/tinyvit.py):

Implemented the full TinyViT model with configurable parameters

Created the TinyViT-5M variant with appropriate embedding dimensions, depths, and heads

Added proper initialization and forward methods

Verified parameter count matches the expected ~5.4M parameters



Fast Distillation Framework



Implemented the fast distillation framework described in the paper:



Logit Distillation (distillation/logit_distill.py):

Created a FastDistillation class that handles:

Precomputing and storing sparse teacher logits (top-k values)

Loading and saving logits to disk for efficiency

Computing distillation loss using the stored logits

Implemented temperature scaling for the distillation loss

Added support for batch processing with unique image IDs



Robust ID Generation (utils/data_utils.py):

Implemented DatasetWithIDs class that assigns permanent IDs to images

Ensures consistent image-to-logit mapping even with shuffled data

Uses file paths as IDs for ImageFolder datasets

Uses hash-based IDs for other datasets like CIFAR



Data Loading Utilities



Implemented data loading utilities in utils/data_utils.py:



Dataset Wrappers:

DatasetWithIDs: Wraps datasets to include permanent image IDs with each item

Supports ImageFolder, CIFAR, and custom datasets



DataLoaders:

get_CIFAR_100: Creates dataloaders for CIFAR-100 with proper transformations

get_ImageNet1K: Creates dataloaders for ImageNet with proper transformations

get_imagenet_subset_loaders: Creates dataloaders for a subset of ImageNet



Training Utilities



Implemented training utilities:



Training Loop (utils/train_utils.py):

Standard training loop with logging and model saving

Evaluation utilities for tracking accuracy



Training with Distillation (train.py):

Specialized training loop that combines cross-entropy and distillation losses

Configurable distillation temperature and weight

Support for different datasets (CIFAR-100, ImageNet)



Verification



All components have been verified with test scripts:



test_model.py: Confirms parameter count and forward pass

test_distillation.py: Tests the distillation framework with consistent image IDs



Usage



Training TinyViT with Distillation



# Precompute teacher logits
python precompute_logits.py --dataset cifar100 --teacher-model resnet50 --k 10 --output-path ./logits/cifar100_logits.pkl

# Train with distillation on CIFAR-100
python train.py --dataset cifar100 --train-data ./data --name tinyvit_5m_distill --use-distillation --logits-path ./logits/cifar100_logits.pkl --distill-alpha 0.5 --distill-temp 1.0



Training TinyViT without Distillation



python train.py --dataset cifar100 --train-data ./data --name tinyvit_5m_baseline



Next Steps



Run experiments on CIFAR-100 and ImageNet subset

Compare performance with and without distillation

Investigate the impact of different distillation temperatures and weights

Extend to other TinyViT variants (TinyViT-11M, TinyViT-21M)