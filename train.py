import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from models.tinyvit import tinyvit_5m
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
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="Distillation weight")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "cifar100"], help="Dataset to use")

    return parser.parse_args()

def train_with_distillation(model, dataloaders, optimizer, scheduler, ce_loss_fn, 
                          distiller, distill_alpha, max_iter, name):
    writer = SummaryWriter(log_dir=f"runs/{name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    best_acc = 0.0

    for iter in range(max_iter):
        model.train()
        for im, label, ids in dataloaders["train"]:
            im = im.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            pred = model(im)
            
            # Regular cross-entropy loss
            ce_loss = ce_loss_fn(pred, label)
            
            # Distillation loss
            distill_loss = distiller.distillation_loss(pred, ids, device)
            
            # Combined loss
            loss = (1 - distill_alpha) * ce_loss + distill_alpha * distill_loss
            
            accuracy = compute_accuracy(pred, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

        model.eval()
        val_losses = []
        val_accuracies = []
        for im, label, ids in dataloaders["val"]:
            im = im.to(device)
            label = label.to(device)
            
            with torch.no_grad():
                pred = model(im)
                ce_loss = ce_loss_fn(pred, label)
                distill_loss = distiller.distillation_loss(pred, ids, device)
                loss = (1 - distill_alpha) * ce_loss + distill_alpha * distill_loss
                accuracy = compute_accuracy(pred, label)
                val_losses.append(loss.item())
                val_accuracies.append(accuracy)

        avg_val_loss = np.mean(val_losses)
        avg_val_accuracy = np.mean(val_accuracies)
        writer.add_scalar('Loss', avg_val_loss, iter)
        writer.add_scalar('Accuracy', avg_val_accuracy, iter)

        # Print stats and save best model every 10 iterations
        if iter % 10 == 0:
            print(f'Iteration: {iter} | Loss: {avg_val_loss:.4f} | Accuracy: {avg_val_accuracy:.4f}')
            if avg_val_accuracy > best_acc:
                save_pth(model, name)
                best_acc = avg_val_accuracy

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = tinyvit_5m(num_classes=100 if args.dataset == "cifar100" else 1000)
    model = model.to(device)

    if args.load_model:
        load_pth(model, args.load_model)

    # Get dataloaders with IDs if using distillation
    if args.dataset == "cifar100":
        dataloaders = get_CIFAR_100(args.train_data, args.batch_size, use_ids=args.use_distillation)
    else:
        dataloaders = get_ImageNet1K(args.train_data, args.batch_size, use_ids=args.use_distillation)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iter - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    
    # Loss function
    ce_loss_fn = nn.CrossEntropyLoss()

    # Setup distillation if needed
    if args.use_distillation:
        if not args.logits_path:
            raise ValueError("Must provide path to pre-computed teacher logits when using distillation")
        
        distiller = FastDistillation(k=10, temperature=args.distill_temp, logit_dir="./logits")
        distiller.load_logits(args.logits_path)
        
        train_with_distillation(model, dataloaders, optimizer, scheduler, ce_loss_fn, 
                              distiller, args.distill_alpha, args.max_iter, args.name)
    else:
        # Use regular training loop without distillation
        train_loop(model, dataloaders, optimizer, scheduler, ce_loss_fn, args.max_iter, args.name)