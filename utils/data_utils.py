import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Subset
import os

class DatasetWithIDs(torch.utils.data.Dataset):
    """Wraps a dataset to include permanent image IDs with each item"""
    def __init__(self, dataset):
        self.dataset = dataset
        
        # For ImageFolder, use file paths as IDs
        if isinstance(dataset, torchvision.datasets.ImageFolder):
            self.ids = [os.path.basename(path) for path, _ in dataset.samples]
        # For Subset, look through to the underlying dataset
        elif isinstance(dataset, Subset):
            if isinstance(dataset.dataset, torchvision.datasets.ImageFolder):
                all_samples = dataset.dataset.samples
                self.ids = [os.path.basename(all_samples[i][0]) for i in dataset.indices]
            else:
                # For other datasets, generate hash-based IDs
                self.ids = [f"img_{hash(str(i))}" for i in range(len(dataset))]
        else:
            # For other datasets, generate hash-based IDs
            self.ids = [f"img_{hash(str(i))}" for i in range(len(dataset))]
    
    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, tuple):
            return (*item, self.ids[index])
        else:
            return (item, self.ids[index])
    
    def __len__(self):
        return len(self.dataset)

def get_ImageNet1K(path, batch_size, num_workers=0, use_ids=False):
    """
    Get ImageNet-1K dataloaders for training and validation
    
    Args:
        path: Path to ImageNet dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        use_ids: Whether to include image IDs in batches
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=f"{path}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f"{path}/val", transform=val_transform)

    # Wrap datasets with IDs if needed
    if use_ids:
        train_dataset = DatasetWithIDs(train_dataset)
        val_dataset = DatasetWithIDs(val_dataset)

    # Create dataloaders
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders

def get_CIFAR_100(path, batch_size, num_workers=0, use_ids=False):
    """
    Get CIFAR-100 dataloaders for training and validation
    
    Args:
        path: Path to store CIFAR-100 dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        use_ids: Whether to include image IDs in batches
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR100(root=f"{path}/CIFAR100", train=True, transform=train_transform, download=True)
    val_dataset = torchvision.datasets.CIFAR100(root=f"{path}/CIFAR100", train=False, transform=val_transform, download=True)
    
    # Wrap datasets with IDs if needed
    if use_ids:
        train_dataset = DatasetWithIDs(train_dataset)
        val_dataset = DatasetWithIDs(val_dataset)

    # Create dataloaders
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders

def get_imagenet_subset_loaders(root, subset_fraction=0.1, batch_size=128, num_workers=4, use_ids=False):
    """
    Get ImageNet subset dataloaders for training and validation
    
    Args:
        root: Path to ImageNet dataset
        subset_fraction: Fraction of dataset to use
        batch_size: Batch size
        num_workers: Number of workers for data loading
        use_ids: Whether to include image IDs in batches
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(
        root=f"{root}/train", transform=train_transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=f"{root}/val", transform=val_transform
    )
    
    # Create subsets
    if subset_fraction < 1.0:
        train_size = int(len(train_dataset) * subset_fraction)
        val_size = int(len(val_dataset) * subset_fraction)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        train_indices = torch.randperm(len(train_dataset))[:train_size]
        val_indices = torch.randperm(len(val_dataset))[:val_size]
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    # Wrap datasets with IDs if needed
    if use_ids:
        train_dataset = DatasetWithIDs(train_dataset)
        val_dataset = DatasetWithIDs(val_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return {"train": train_loader, "val": val_loader}