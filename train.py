import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from models.tinyvit import tinyvit_5m, tinyvit_finetune
from distillation.logit_distill import FastDistillation
from utils.train_utils import *
from utils.data_utils import *

def get_args():
    parser = argparse.ArgumentParser(description="Training Script")
    
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum number of training iterations")
    parser.add_argument("--warmup-epochs", type=int, default=20, help="Number of warm-up epochs")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight Decay")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--name", type=str, required=True, help="Model architecture name")
    parser.add_argument("--load-model", type=str, default=None, help="Path to pre-trained model (optional)")
    parser.add_argument("--use-distillation", action="store_true", help="Enable distillation")
    parser.add_argument("--logits-path", type=str, default=None, help="Path to pre-computed teacher logits")
    parser.add_argument("--distill-temp", type=float, default=1.0, help="Distillation temperature")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "cifar100"], help="Dataset to use")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    

    # Setup finetune or base model
    if args.dataset == "cifar100":
        # Finetune model
        model = tinyvit_finetune(num_classes=100, pretrained_model_path=args.load_model)
        model = model.to(device)
        dataloaders = get_CIFAR_100(args.train_data, args.batch_size, use_ids=args.use_distillation)
        optimizer = optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Base model
        model = tinyvit_5m(num_classes=1000)
        model = model.to(device)

        if args.load_model:
            load_pth(model, args.load_model)
        dataloaders = get_ImageNet1K(args.train_data, args.batch_size, use_ids=args.use_distillation)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup optimizer and scheduler
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iter - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    
    # Loss function
    ce_loss_fn = nn.CrossEntropyLoss()

    # Setup distillation if needed
    if args.use_distillation:
        if not args.logits_path:
            raise ValueError("Must provide path to pre-computed teacher logits when using distillation")
        
        distiller = FastDistillation(k=10, temperature=args.distill_temp, logit_dir=args.logits_path)
        distiller.load_logits(args.logits_path)
        
        train_with_distillation(model, dataloaders, optimizer, scheduler, ce_loss_fn, 
                              distiller, args.max_iter, args.name)
    else:
        # Use regular training loop without distillation
        train_loop(model, dataloaders, optimizer, scheduler, ce_loss_fn, args.max_iter, args.name)