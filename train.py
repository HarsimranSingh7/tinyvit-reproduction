import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from models.tinyvit import tinyvit_5m
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

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    model = tinyvit_5m()

    if args.load_model:
        load_pth(model, args.load_model)

    dataloader = get_ImageNet1K(args.train_data, args.batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iter - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    loss_fn = nn.CrossEntropyLoss()

    train_loop(model, dataloader, optimizer, scheduler, loss_fn, args.max_iter, args.name)