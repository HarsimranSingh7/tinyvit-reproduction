import torch
import torchvision
import torchvision.transforms as transforms
from models.tinyvit import tinyvit_5m
from distillation.logit_distill import FastDistillation
from utils.data_utils import DatasetWithIDs
import timm
import os

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load a small subset of CIFAR-10 for testing
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use a small subset for testing
    subset_size = 100
    subset_indices = torch.randperm(len(trainset))[:subset_size]
    subset = torch.utils.data.Subset(trainset, subset_indices)
    
    # Wrap the subset with our ID generator
    subset_with_ids = DatasetWithIDs(subset)
    
    # Use num_workers=0 to avoid multiprocessing issues
    trainloader = torch.utils.data.DataLoader(
        subset, batch_size=10, shuffle=False, num_workers=0
    )
    
    # Create student model
    student_model = tinyvit_5m(num_classes=10)
    student_model = student_model.to(device)
    
    # Create teacher model (a simple ResNet18 for testing)
    try:
        teacher_model = timm.create_model('resnet18', pretrained=True, num_classes=10)
        teacher_model = teacher_model.to(device)
    except:
        print("Could not load pretrained model. Using random initialization.")
        teacher_model = timm.create_model('resnet18', pretrained=False, num_classes=10)
        teacher_model = teacher_model.to(device)
    
    # Setup distillation
    logit_dir = './logits'
    os.makedirs(logit_dir, exist_ok=True)
    distiller = FastDistillation(teacher_model, k=5, logit_dir=logit_dir)
    
    # Precompute teacher logits
    logit_path = os.path.join(logit_dir, 'cifar10_test_logits.pkl')
    
    # Use the dataset with IDs - now we can enable shuffle if desired
    trainloader_with_ids = torch.utils.data.DataLoader(
        subset_with_ids, batch_size=10, shuffle=True, num_workers=0
    )
    
    # Precompute logits
    distiller.precompute_teacher_logits(trainloader_with_ids, device, save_path=logit_path)
    
    # Test loading logits
    new_distiller = FastDistillation(k=5, logit_dir=logit_dir)
    new_distiller.load_logits(logit_path)
    
    # Test distillation loss
    for inputs, labels, ids in trainloader_with_ids:
        inputs = inputs.to(device)
        student_outputs = student_model(inputs)
        
        # Compute distillation loss
        distill_loss = new_distiller.distillation_loss(student_outputs, ids, device)
        
        print(f"Distillation loss: {distill_loss.item()}")
        break
    
    print("Distillation test completed successfully!")

if __name__ == "__main__":
    main()