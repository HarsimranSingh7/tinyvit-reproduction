import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from tqdm import tqdm
import numpy as np

from utils.train_utils import compute_accuracy

class FastDistillation:
    """
    Fast Distillation using pre-stored sparse teacher logits
    """
    def __init__(self, teacher_model=None, k=10, temperature=1.0, logit_dir='./logits'):
        """
        Args:
            teacher_model: Teacher model for generating logits
            k: Number of top logits to store
            temperature: Distillation temperature
            logit_dir: Directory to store/load logits
        """
        self.teacher_model = teacher_model
        self.k = k
        self.temperature = temperature
        self.logit_dir = logit_dir
        self.logit_storage = {}  # {image_id: (values, indices)}
        
        # Create directory if it doesn't exist
        os.makedirs(logit_dir, exist_ok=True)
    
    def precompute_teacher_logits(self, dataloader, device, save_path=None):
        """
        Precompute and store sparse teacher logits for the dataset
        
        Args:
            dataloader: DataLoader for the dataset
            device: Device to run teacher model on
            save_path: Path to save logits (optional)
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model must be provided for precomputing logits")
        
        self.teacher_model.eval()
        self.teacher_model = self.teacher_model.to(device)
        
        print(f"Precomputing teacher logits with k={self.k}...")
        
        accuracies = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Unpack batch - adjust based on your dataloader
                if len(batch) == 2:
                    # Standard (inputs, labels) format
                    inputs, _ = batch
                    # Generate unique IDs based on batch index and position
                    image_ids = [f"{i}" for i in range(inputs.shape[0])]
                elif len(batch) == 3:
                    # (inputs, labels, image_ids) format
                    inputs, labels, image_ids = batch
                
                inputs = inputs.to(device)
                
                # Get teacher logits
                logits = self.teacher_model(inputs)

                # normalize logits
                logits = F.softmax(logits, dim=1)
                
                # Get top-k values and indices
                values, indices = torch.topk(logits, k=self.k, dim=1)
                
                # Store in dictionary
                for i, img_id in enumerate(image_ids):
                    img_id_key = str(img_id) if not isinstance(img_id, str) else img_id
                    self.logit_storage[img_id_key] = (
                        values[i].cpu().numpy(),
                        indices[i].cpu().numpy()
                    )

                accuracies.append(compute_accuracy(logits.cpu(), labels))
        
        print(f"Precomputed logits for {len(self.logit_storage)} images with accuracy: {np.mean(accuracies)}")
        
        # Save logits if path provided
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(self.logit_storage, f, protocol=pickle.HIGHEST_PROTOCOL)  
            print(f"Saved logits to {save_path}")
    
    def load_logits(self, path):
        """
        Load precomputed logits from file
        """
        with open(path, 'rb') as f:
            self.logit_storage = pickle.load(f)
        print(f"Loaded logits for {len(self.logit_storage)} images from {path}")
    
    def distillation_loss(self, student_logits, image_ids, device):
        """
        Compute distillation loss using pre-stored teacher logits
        
        Args:
            student_logits: Logits from student model [batch_size, num_classes]
            image_ids: List of image IDs corresponding to the batch
            device: Device for computation
        
        Returns:
            Distillation loss
        """
        batch_size = student_logits.size(0)
        num_classes = student_logits.size(1)
        
        # Initialize loss
        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        for i, img_id in enumerate(image_ids):
            img_id_key = str(img_id) if not isinstance(img_id, str) else img_id
            
            if img_id_key in self.logit_storage:
                teacher_values, teacher_indices = self.logit_storage[img_id_key]
                
                # Convert to tensors
                teacher_values = torch.tensor(teacher_values, device=device)
                teacher_indices = torch.tensor(teacher_indices, device=device)
                
                # Create sparse teacher logits
                sparse_teacher = torch.zeros(num_classes, device=device)
                sparse_teacher[teacher_indices] = teacher_values
                
                # KL divergence loss
                student_probs = F.log_softmax(student_logits[i] / self.temperature, dim=0)
                teacher_probs = F.softmax(sparse_teacher / self.temperature, dim=0)
                
                loss = F.kl_div(student_probs, teacher_probs, reduction='sum') * (self.temperature ** 2)
                total_loss += loss
                valid_samples += 1
        
        # Return average loss
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=device)
        
    def distillation_labels(self, student_logits, image_ids, device):
        """
        Get the teacher labels from sparse logits
        
        Args:
            student_logits: Logits from student model [batch_size, num_classes]
            image_ids: List of image IDs corresponding to the batch
            device: Device for computation
        
        Returns:
            Distillation labels
        """
        batch_size, num_classes = student_logits.shape

        sparse_teacher = torch.zeros((batch_size, num_classes), device=device)

        for i, img_id in enumerate(image_ids):
            img_id_key = str(img_id) if not isinstance(img_id, str) else img_id
            teacher_values, teacher_indices = self.logit_storage[img_id_key]

            teacher_values = torch.tensor(teacher_values, device=device)
            teacher_indices = torch.tensor(teacher_indices, device=device)

            sparse_teacher[i, teacher_indices] = teacher_values

            smooth_value = (1 - teacher_values.sum()) / (num_classes - self.k)  # label smoothing

            # Add smoothing to non-selected indices
            mask = torch.ones(num_classes, device=device)
            mask[teacher_indices] = 0  # Ignore teacher-assigned indices
            sparse_teacher[i] += smooth_value * mask 

        return sparse_teacher