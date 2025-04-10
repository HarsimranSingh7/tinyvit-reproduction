import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import timm
from tqdm import tqdm
import numpy as np
import random
import math

# Import our custom modules
from tinyvit_11m import tinyvit_11m
from fast_distillation import FastDistillation, DistillationLoss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args_parser():
    parser = argparse.ArgumentParser('TinyViT-11M training script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--subset-fraction', default=0.3, type=float,
                        help='Fraction of ImageNet to use (default: 0.3)')
    parser.add_argument('--warmup-epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--data-path', default='./data', type=str)
    parser.add_argument('--output-dir', default='./output', type=str)
    parser.add_argument('--log-dir', default='./runs', type=str)
    parser.add_argument('--name', default='tinyvit_11m', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--use-distillation', action='store_true')
    parser.add_argument('--get-teacher', action='store_true')
    parser.add_argument('--teacher-model', default='efficientnet_b0', type=str)
    parser.add_argument('--logits-path', default='./logits/efficientnet_b0_logits.pkl', type=str)
    parser.add_argument('--distill-alpha', default=0.5, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--topk', default=30, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--mixed-precision', action='store_true')
    return parser

def main(args):
    print(f"Starting training with args: {args}")
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(args.log_dir, f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}"))
    
    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    try:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        val_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
        
        # Create subset if requested
        if args.subset_fraction < 1.0:
            num_train = len(train_dataset)
            indices = list(range(num_train))
            random.shuffle(indices)
            split = int(np.floor(args.subset_fraction * num_train))
            train_idx = indices[:split]
            train_dataset = Subset(train_dataset, train_idx)
            
            num_val = len(val_dataset)
            indices = list(range(num_val))
            random.shuffle(indices)
            split = int(np.floor(args.subset_fraction * num_val))
            val_idx = indices[:split]
            val_dataset = Subset(val_dataset, val_idx)
        
        print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False
        )
        
        num_classes = 1000  # ImageNet has 1000 classes
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using simulated data for demonstration...")
        
        # Create dummy datasets for simulation
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=10000, num_classes=1000):
                self.size = size
                self.num_classes = num_classes
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                img = torch.randn(3, 224, 224)
                target = torch.randint(0, self.num_classes, (1,)).item()
                return img, target
        
        train_dataset = DummyDataset(size=int(100000 * args.subset_fraction))
        val_dataset = DummyDataset(size=int(10000 * args.subset_fraction))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        num_classes = 1000
        
        print(f"Using simulated data: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Create model
    print("Creating TinyViT-11M model...")
    model = tinyvit_11m(num_classes=num_classes)
    model = model.to(device)
    
    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_count:,}")
    
    # Setup teacher model and distillation if needed
    if args.get_teacher:
        print(f"Loading teacher model: {args.teacher_model}")
        teacher_model = timm.create_model(args.teacher_model, pretrained=True, num_classes=num_classes)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        
        # Compute and store teacher logits
        distiller = FastDistillation(
            teacher_model=teacher_model,
            k=args.topk,
            temperature=args.temperature,
            device=device
        )
        
        distiller.compute_and_store_logits(train_loader, args.logits_path)
        
        # Evaluate teacher model
        teacher_correct = 0
        teacher_total = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Evaluating teacher"):
                images, targets = images.to(device), targets.to(device)
                outputs = teacher_model(images)
                _, predicted = outputs.max(1)
                teacher_total += targets.size(0)
                teacher_correct += predicted.eq(targets).sum().item()
        
        teacher_accuracy = 100. * teacher_correct / teacher_total
        print(f"Teacher model accuracy: {teacher_accuracy:.2f}%")
        return
    
    # Load teacher logits for distillation
    distiller = None
    if args.use_distillation:
        print(f"Loading teacher logits from {args.logits_path}")
        teacher_model = None  # We don't need the teacher model for training
        distiller = FastDistillation(
            teacher_model=teacher_model,
            k=args.topk,
            temperature=args.temperature,
            device=device
        )
        
        # Just load the logits
        try:
            import pickle
            with open(args.logits_path, 'rb') as f:
                distiller.logits_dict = pickle.load(f)
            print(f"Loaded logits for {len(distiller.logits_dict)} images")
        except Exception as e:
            print(f"Error loading logits: {e}")
            print("Continuing without distillation...")
            args.use_distillation = False
    
    # Define loss function
    if args.use_distillation:
        criterion = DistillationLoss(alpha=args.distill_alpha, temperature=args.temperature)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Define optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs)
    
    # Training loop
    best_acc = 0.0
    
    # Initialize scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(train_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get batch indices for distillation
            batch_indices = list(range(
                batch_idx * train_loader.batch_size,
                min((batch_idx + 1) * train_loader.batch_size, len(train_dataset))
            ))
            
            # Get soft targets if using distillation
            soft_targets = None
            if args.use_distillation and distiller is not None:
                soft_targets = distiller.get_soft_targets(batch_indices, num_classes=num_classes)
            
            optimizer.zero_grad()
            
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, soft_targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets, soft_targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            train_bar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, desc="Validating")):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%, "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, f"{args.name}_best.pth"))
            print(f"Saved best model with accuracy: {val_acc:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, os.path.join(args.output_dir, f"{args.name}_final.pth"))
    
    print(f"Training completed. Best validation accuracy: {best_acc:.2f}%")
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TinyViT-11M training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
