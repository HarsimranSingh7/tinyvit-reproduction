# distillation/fast_distillation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from tqdm import tqdm

class FastDistillation:
    """
    Handles pre-computation of sparse teacher logits and generation of
    dense soft targets for knowledge distillation.

    Intended for use with `train_tinyvit_11m.py`.
    """
    def __init__(self, teacher_model, k=30, temperature=1.0, device='cuda'):
        """
        Args:
            teacher_model (nn.Module): The teacher model (needed for precomputation).
            k (int): Number of top logits to store per image. Default 30.
            temperature (float): Temperature for distillation loss. Default 1.0.
            device (str or torch.device): Device for computation.
        """
        self.teacher_model = teacher_model
        self.k = k
        self.temperature = temperature
        self.device = torch.device(device) # Ensure it's a torch.device
        self.logits_dict = {} # Stores {image_idx: {'values': top_k_logits, 'indices': top_k_indices}}
        if self.teacher_model is not None:
            self.teacher_model.eval()
            self.teacher_model.to(self.device)

    def compute_and_store_logits(self, dataloader, logits_path):
        """
        Computes and stores the top-k raw logits for each image in the dataset.
        Uses simple batch indices as keys, assuming dataloader order is fixed
        during this precomputation phase.

        Args:
            dataloader: DataLoader for the dataset (expects inputs as first element).
            logits_path (str): Path to save the computed logits dictionary.
        """
        # Check if logits already exist
        if os.path.exists(logits_path):
            try:
                print(f"Loading existing logits from {logits_path}")
                with open(logits_path, 'rb') as f:
                    self.logits_dict = pickle.load(f)
                print(f"Loaded logits for {len(self.logits_dict)} images.")
                return # Skip re-computation
            except Exception as e:
                 print(f"Warning: Could not load existing logits file {logits_path}. Recomputing. Error: {e}")


        if self.teacher_model is None:
             raise ValueError("Teacher model must be provided to compute logits.")

        print(f"Precomputing teacher raw logits (top {self.k}) using {type(self.teacher_model).__name__}...")
        self.teacher_model.eval()
        self.logits_dict = {} # Ensure fresh computation

        with torch.no_grad():
            global_idx = 0
            for batch in tqdm(dataloader, desc="Precomputing Logits"):
                # Assuming images are the first element in the batch
                images = batch[0].to(self.device)
                batch_size = images.size(0)

                # Get teacher raw logits
                teacher_logits = self.teacher_model(images)

                # Get top-k values (raw logits) and indices
                topk_values, topk_indices = torch.topk(teacher_logits, self.k, dim=1)

                # Store in dictionary using a global index as key
                # Assumes dataloader iteration order is consistent for this run.
                for i in range(batch_size):
                    img_idx = global_idx + i
                    self.logits_dict[img_idx] = {
                        'values': topk_values[i].cpu().numpy(),  # Store raw logits
                        'indices': topk_indices[i].cpu().numpy()
                    }
                global_idx += batch_size

        # Save the logits dictionary
        try:
            os.makedirs(os.path.dirname(logits_path), exist_ok=True)
            with open(logits_path, 'wb') as f:
                pickle.dump(self.logits_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {len(self.logits_dict)} logits to {logits_path}")
        except Exception as e:
             print(f"Error saving computed logits to {logits_path}: {e}")


    def get_soft_targets(self, indices, num_classes=1000):
        """
        Generates dense soft targets for a batch based on precomputed sparse logits,
        following the TinyViT paper's formula (Eq. after Fig 4).

        Applies temperature scaling to the top-k logits *before* softmax,
        then uses label smoothing for non-top-k classes.

        Args:
            indices (list or np.array): Batch indices (0-based) corresponding to the
                                        original dataset order used during precomputation.
            num_classes (int): Total number of classes in the dataset.

        Returns:
            torch.Tensor: Soft targets tensor of shape [batch_size, num_classes] on self.device.
        """
        batch_size = len(indices)
        soft_targets = torch.zeros(batch_size, num_classes, device=self.device)
        all_found = True

        for i, idx in enumerate(indices):
            if idx in self.logits_dict:
                # Get stored top-k raw logits and indices
                values_raw = torch.tensor(self.logits_dict[idx]['values'], device=self.device, dtype=torch.float)
                indices_topk = torch.tensor(self.logits_dict[idx]['indices'], device=self.device, dtype=torch.long)

                # Apply temperature scaling to the raw top-k logits
                values_scaled = values_raw / self.temperature

                # Calculate softmax *only* over the scaled top-k logits
                probs_topk = torch.softmax(values_scaled, dim=0)

                # Assign these probabilities to the correct indices in the dense target
                soft_targets[i].scatter_(0, indices_topk, probs_topk)

                # Calculate remaining probability mass for smoothing
                sum_topk_probs = probs_topk.sum() # Sum of probabilities derived from top-k
                remaining_mass = 1.0 - sum_topk_probs

                # Number of classes not in the top-k
                num_other_classes = num_classes - self.k

                if num_other_classes > 0:
                    # Calculate uniform probability for non-top-k classes
                    smoothing_value = max(0.0, remaining_mass.item() / num_other_classes)

                    # Create mask for non-top-k classes
                    mask = torch.ones(num_classes, device=self.device, dtype=torch.bool)
                    mask[indices_topk] = False

                    # Assign smoothing value
                    soft_targets[i, mask] = smoothing_value

                # Optional check/renormalization if k=num_classes and sum wasn't 1
                elif remaining_mass > 1e-6 or remaining_mass < -1e-6:
                    print(f"Warning: k={self.k} equals num_classes={num_classes}, but sum deviates: {sum_topk_probs} for ID {idx}. Renormalizing.")
                    soft_targets[i] /= sum_topk_probs # Renormalize

            else:
                print(f"Error: Logit data not found for image index: {idx}. Returning zeros for this sample.")
                soft_targets[i, :] = 0.0 # Indicate missing data with zeros
                all_found = False

        if not all_found:
             print("Warning: Some image indices were missing from the logit storage.")

        return soft_targets


class DistillationLoss(nn.Module):
    """
    Combines Cross-Entropy loss with hard targets and Kullback-Leibler (KL)
    divergence loss with soft targets from a teacher model.

    Loss = (1 - alpha) * CE(student_logits, hard_targets) + alpha * KLDiv(student_soft, teacher_soft) * T^2
    """
    def __init__(self, alpha=0.5, temperature=1.0):
        """
        Args:
            alpha (float): Weight for the distillation loss component (0 <= alpha <= 1).
                           Weight for CE loss is (1 - alpha).
            temperature (float): Temperature scaling factor for both student and teacher
                                 outputs in the KL divergence term.
        """
        super().__init__()
        if not 0 <= alpha <= 1:
             raise ValueError(f"alpha must be between 0 and 1, but got {alpha}")
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss() # Standard CE for hard labels

    def forward(self, student_logits, hard_targets, teacher_soft_targets=None):
        """
        Compute the combined distillation loss.

        Args:
            student_logits (torch.Tensor): Raw logits from the student model (B, C).
            hard_targets (torch.Tensor): Ground truth labels (B,).
            teacher_soft_targets (torch.Tensor, optional): Dense soft targets from the
                                                         teacher (B, C). If None, only
                                                         CE loss is computed.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        # Calculate standard Cross-Entropy loss with hard targets
        ce_loss_val = self.ce_loss(student_logits, hard_targets)

        # If no teacher targets provided or alpha is 0, return only CE loss
        if teacher_soft_targets is None or self.alpha == 0:
            return ce_loss_val

        # Calculate KL Divergence loss for distillation
        # Soften student logits and apply log-softmax
        student_log_softmax = F.log_softmax(student_logits / self.temperature, dim=1)

        # Teacher targets are already probabilities (softmax applied during generation)
        # Note: KLDivLoss expects log-probabilities for the input and probabilities for the target.
        distill_loss_val = F.kl_div(
            input=student_log_softmax,
            target=teacher_soft_targets, # Already probabilities
            reduction='batchmean' # Average loss over the batch
        )

        # Scale KL divergence loss by T^2 as per Hinton et al.
        distill_loss_val = distill_loss_val * (self.temperature ** 2)

        # Combine the losses using alpha
        total_loss = (1.0 - self.alpha) * ce_loss_val + self.alpha * distill_loss_val
        return total_loss

# (Keep the __main__ test block as it is useful)
# ... test code ...
