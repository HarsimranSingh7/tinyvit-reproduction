import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import timm

from models.tinyvit import tinyvit_5m, tinyvit_finetune
from distillation.logit_distill import FastDistillation
from utils.train_utils import *
from utils.data_utils import *

# subset training
# python train.py --train-data '~/Desktop/data' --subset-fraction 0.1 --name 'subsetImNet' --batch-size 16

# precompute logits
# python train.py --train-data '~/Desktop/data' --logits-path '/Users/popo/Desktop/data/logits.pkl' --get-teacher True --batch-size 16 --name 'subsetImNetDistillation'

# distillation
# python train.py --train-data '~/Desktop/data' --subset-fraction 0.1 --name 'subsetImNetDistillation' --logits-path '/Users/popo/Desktop/data/logits.pkl' --use-distillation True --batch-size 16

# transfer learning
# python train.py --train-data '~/Desktop/data' --logits-path '~/Desktop/data/C100logits' --get-teacher True --dataset 'cifar100' --load-model 'subsetImNetDistillation' --batch-size 16

# python train.py --train-data '~/Desktop/data' --name 'subsetImNetDistillationTransfer' --logits-path '~/Desktop/data/C100logits' --use-distillation True --dataset 'cifar100' --load-model 'subsetImNetDistillation' --batch-size 16


def get_args():
    parser = argparse.ArgumentParser(description="Training Script")
    
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum number of training iterations")
    parser.add_argument("--warmup-epochs", type=int, default=20, help="Number of warm-up epochs")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data")
    parser.add_argument("--subset-fraction", type=float, default=1.0, help="The fraction of data should use")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight Decay")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--name", type=str, required=True, help="Model architecture name")
    parser.add_argument("--load-model", type=str, default=None, help="Path to pre-trained model (optional)")
    parser.add_argument("--use-distillation", type=bool, default=False, help="Enable distillation")
    parser.add_argument("--get-teacher", type=bool, default=False, help="Calculate teacher model logits")
    parser.add_argument("--logits-path", type=str, default=None, help="Path to pre-computed teacher logits")
    parser.add_argument("--distill-temp", type=float, default=1.0, help="Distillation temperature")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "cifar100"], help="Dataset to use")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

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
        dataloaders = get_imagenet_subset_loaders(args.train_data, args.subset_fraction, args.batch_size, use_ids=(args.use_distillation or args.get_teacher))
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.get_teacher:
        print("start logits computation")
        teacher_model = timm.create_model('resnet18', pretrained=True, num_classes=1000)
        teacher_model = teacher_model.to(device)

        # FIX
        logit_file_path = args.logits_path
        logit_dir = os.path.dirname(logit_file_path)
        
        distiller = FastDistillation(teacher_model, k=5, logit_dir=logit_dir)
        distiller.precompute_teacher_logits(dataloaders["train"], device, save_path=logit_file_path)
        exit()

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
        
        distiller = FastDistillation(k=10, temperature=args.distill_temp)
        distiller.load_logits(args.logits_path)
        
        train_with_distillation(model, dataloaders, optimizer, scheduler, ce_loss_fn, 
                              distiller, args.max_iter, args.name)
    else:
        # Use regular training loop without distillation
        train_loop(model, dataloaders, optimizer, scheduler, ce_loss_fn, args.max_iter, args.name)
