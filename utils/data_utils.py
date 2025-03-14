import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

def get_ImageNet1K(path, batch_size, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize(256),  
        transforms.CenterCrop(224),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    full_dataset = datasets.ImageFolder(root="{path}/ImageNet1K", transform=transform)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders

def get_CIFAR_100(path, batch_size, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean & std
    ])

    datasets = {
        "train": torchvision.datasets.CIFAR100(root=f"{path}/CIFAR100", train=True, transform=transform, download=True),
        "val": torchvision.datasets.CIFAR100(root=f"{path}/CIFAR100", train=False, transform=transform, download=True),
    }

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders