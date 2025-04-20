# train.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import timm
import os # Import os

# Import necessary components
from models.tinyvit import tinyvit_5m, tinyvit_finetune
from distillation.logit_distill import FastDistillation # Uses the one from logit_distill.py
from utils.train_utils import train_loop, train_with_distillation, load_checkpoint # Use checkpointing
from utils.data_utils import get_imagenet_subset_loaders, get_CIFAR_100 # Use specific loaders

# --- Argument Parser ---
def get_args():
    parser = argparse.ArgumentParser(description="TinyViT-5M Training Script")

    # Training setup
    parser.add_argument("--name", type=str, required=True, help="Experiment name (for logs and checkpoints)")
    parser.add_argument("--output-dir", type=str, default='./output', help="Directory to save checkpoints and logs")
    parser.add_argument("--max-epochs", type=int, default=300, help="Maximum number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=20, help="Number of warm-up epochs for LR scheduler")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (adjust based on memory)") # Default reduced from 1024

    # Dataset
    parser.add_argument("--dataset", type=str, default="imagenet_subset", choices=["imagenet_subset", "cifar100"], help="Dataset to use")
    parser.add_argument("--train-data", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--subset-fraction", type=float, default=0.1, help="Fraction of ImageNet subset to use (if dataset is imagenet_subset)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")

    # Model
    parser.add_argument("--load-model-path", type=str, default=None, help="Path to load pre-trained model weights for fine-tuning or resuming (e.g., './output/my_exp_best.pth')")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (default: 1000 for ImageNet, 100 for CIFAR-100)")

    # Optimizer and Scheduler
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate") # 0.001 is common starting point
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay (AdamW)")

    # Distillation
    parser.add_argument("--use-distillation", action='store_true', help="Enable knowledge distillation")
    parser.add_argument("--get-teacher-logits", action='store_true', help="Only precompute teacher logits and exit")
    parser.add_argument("--teacher-model", type=str, default='resnet18', help="Teacher model name (from timm or custom) for distillation (e.g., 'convnext_base', 'deit_base_patch16_224')")
    parser.add_argument("--logits-path", type=str, default=None, help="Path to save/load pre-computed teacher logits (required for distillation or get-teacher)")
    parser.add_argument("--distill-k", type=int, default=10, help="Number of top-K logits to store for distillation")
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="Weight alpha for distillation loss (0 <= alpha <= 1)")
    parser.add_argument("--distill-temp", type=float, default=1.0, help="Temperature for distillation")

    return parser.parse_args()

# --- Main Execution ---
if __name__ == '__main__':
    args = get_args()

    # --- Setup Device ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Determine Number of Classes ---
    if args.num_classes is None:
        if args.dataset == "cifar100":
            num_classes = 100
        else: # Default to ImageNet
            num_classes = 1000
    else:
        num_classes = args.num_classes
    print(f"Training for {num_classes} classes.")

    # --- Setup Dataloaders ---
    # Need IDs if precomputing logits or using distillation
    use_ids_in_dataloader = args.get_teacher_logits or args.use_distillation
    print(f"Using IDs in dataloader: {use_ids_in_dataloader}")

    if args.dataset == "cifar100":
        dataloaders = get_CIFAR_100(args.train_data, args.batch_size, args.num_workers, use_ids=use_ids_in_dataloader)
    elif args.dataset == "imagenet_subset":
        dataloaders = get_imagenet_subset_loaders(args.train_data, args.subset_fraction, args.batch_size, args.num_workers, use_ids=use_ids_in_dataloader)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    print(f"Loaded dataset: {args.dataset}")


    # --- Handle Teacher Logit Precomputation ---
    if args.get_teacher_logits:
        if not args.logits_path:
            raise ValueError("--logits-path is required when using --get-teacher-logits")
        print(f"Precomputing teacher logits using model: {args.teacher_model}")
        try:
            # Load teacher model
            teacher_model = timm.create_model(args.teacher_model, pretrained=True, num_classes=num_classes)
            print(f"Loaded teacher model {args.teacher_model} from timm.")
        except Exception as e:
            print(f"Error loading teacher model {args.teacher_model} from timm: {e}")
            print("Ensure the model name is correct and timm is installed.")
            exit(1)

        # Initialize distiller (from logit_distill)
        distiller = FastDistillation(teacher_model, k=args.distill_k, temperature=args.distill_temp)

        # Run precomputation
        print("Starting logit precomputation...")
        # Use the 'train' split dataloader for precomputing over the training set
        distiller.precompute_teacher_logits(dataloaders["train"], device, save_path=args.logits_path)
        print("Logit precomputation finished. Exiting.")
        exit(0) # Exit after precomputation


    # --- Setup Student Model ---
    if args.dataset == "cifar100":
        # Fine-tuning setup
        print(f"Setting up model for fine-tuning on CIFAR-100.")
        # Load base model weights if specified, freeze body, reinit head
        model = tinyvit_finetune(
            num_classes=num_classes, # Should be 100
            pretrained_model_path=args.load_model_path # Path to base model trained on ImageNet
        )
    else:
        # Training from scratch or resuming on ImageNet subset
        print(f"Setting up TinyViT-5M for {args.dataset}.")
        model = tinyvit_5m(num_classes=num_classes)
        # Load weights only if NOT fine-tuning (handled by tinyvit_finetune)
        # and if a path is provided (could be for resuming training)
        if args.load_model_path and args.dataset != "cifar100":
             try:
                  # Basic loading for resuming (might need full checkpoint later)
                  model.load_state_dict(torch.load(args.load_model_path, map_location='cpu'))
                  print(f"Loaded base weights from {args.load_model_path}")
             except Exception as e:
                  print(f"Warning: Could not load weights from {args.load_model_path}: {e}")


    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {type(model).__name__} initialized with {param_count/1e6:.2f}M trainable parameters.")


    # --- Setup Optimizer and Scheduler ---
    # Fine-tune only the head, otherwise optimize all trainable parameters
    if args.dataset == "cifar100":
        parameters_to_optimize = model.head.parameters()
        print("Optimizing only the head parameters for fine-tuning.")
    else:
        parameters_to_optimize = model.parameters()
        print("Optimizing all trainable parameters.")

    optimizer = optim.AdamW(parameters_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    # Setup LR scheduler: Linear warmup + Cosine decay
    if args.warmup_epochs >= args.max_epochs:
         print("Warning: warmup_epochs >= max_epochs. Using only linear warmup.")
         warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.max_epochs)
         scheduler = warmup_scheduler
    else:
         warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
         cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs - args.warmup_epochs, eta_min=args.lr * 0.01) # Decay to 1% of base LR
         scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    print("Optimizer and LR Scheduler configured.")

    # --- Load Checkpoint for Resuming ---
    # If load_model_path is specified for resuming (not fine-tuning from scratch)
    start_epoch = 0
    if args.load_model_path and args.dataset != "cifar100":
        print(f"Attempting to load checkpoint from {args.output_dir} with name {args.name} to resume...")
        # Use the *experiment name* for checkpointing, load_model_path might just be initial weights
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.name, args.output_dir)


    # --- Setup Distillation (if enabled) ---
    distiller = None
    if args.use_distillation:
        if not args.logits_path:
            raise ValueError("--logits-path is required when using --use-distillation")
        print(f"Setting up distillation using logits from: {args.logits_path}")
        # Initialize distiller (teacher model instance not needed for label generation)
        distiller = FastDistillation(k=args.distill_k, temperature=args.distill_temp)
        distiller.load_logits(args.logits_path)
        print(f"Distillation enabled with alpha={args.distill_alpha}, temp={args.distill_temp}, k={args.distill_k}")


    # --- Start Training ---
    print(f"\nStarting training from epoch {start_epoch} for {args.max_epochs} epochs...")
    if args.use_distillation:
        train_with_distillation(
            model=model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            scheduler=scheduler,
            distiller=distiller, # Pass the initialized distiller
            start_epoch=start_epoch,
            max_epochs=args.max_epochs,
            name=args.name,
            alpha=args.distill_alpha,
            temperature=args.distill_temp,
            output_dir=args.output_dir
        )
    else:
        # Standard training loop needs the loss function
        loss_fn = nn.CrossEntropyLoss()
        train_loop(
            model=model,
            dataloader=dataloaders,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            start_epoch=start_epoch,
            max_epochs=args.max_epochs,
            name=args.name,
            output_dir=args.output_dir
        )

    print("Script finished.")
