# distillation/logit_distill.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from tqdm import tqdm
import numpy as np

# Assuming compute_accuracy is available (e.g., from utils)
# from utils.train_utils import compute_accuracy
# Placeholder if not directly importable:
def compute_accuracy(prediction, gt_labels):
     preds = torch.argmax(prediction.cpu(), dim=1)
     correct = torch.eq(preds, gt_labels.cpu()).sum().item()
     accuracy = correct / gt_labels.size(0) if gt_labels.size(0) > 0 else 0
     return accuracy

class FastDistillation:
    """
    Handles pre-computation and retrieval of sparse teacher logits for
    knowledge distillation, generating dense soft labels according to
    the TinyViT paper's method.

    Intended for use with `train.py` (TinyViT-5M setup).
    """
    def __init__(self, teacher_model=None, k=10, temperature=1.0, logit_dir='./logits'):
        """
        Args:
            teacher_model: Teacher model instance (required for precomputing).
            k (int): Number of top logits to store per image.
            temperature (float): Temperature for potential scaling (though not used
                                 in the label generation formula itself here).
            logit_dir (str): Directory to store/load precomputed logits.
        """
        self.teacher_model = teacher_model
        self.k = k
        self.temperature = temperature # Store temperature, might be needed by loss function
        self.logit_dir = logit_dir
        self.logit_storage = {}  # Stores {image_id: (top_k_softmax_values, top_k_indices)}

        os.makedirs(logit_dir, exist_ok=True)

    def precompute_teacher_logits(self, dataloader, device, save_path=None):
        """
        Precomputes and stores sparse teacher logits (top-k softmax values and indices).

        Args:
            dataloader: DataLoader providing (inputs, labels, image_ids) or (inputs, labels).
            device: Device to run the teacher model on.
            save_path (str, optional): Path to save the computed logits as a pickle file.
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model must be provided for precomputing logits")

        self.teacher_model.eval()
        self.teacher_model = self.teacher_model.to(device)

        print(f"Precomputing teacher logits (top {self.k} softmax values) using {type(self.teacher_model).__name__}...")

        accuracies = []
        batch_start_idx = 0
        self.logit_storage = {} # Clear previous storage

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Precomputing Logits"):
                # Unpack batch - handle different formats
                image_ids = None
                if len(batch) == 3: # Assume (inputs, labels, image_ids)
                    inputs, labels, image_ids = batch
                    image_ids = [str(img_id) for img_id in image_ids] # Ensure string keys
                elif len(batch) == 2: # Assume (inputs, labels)
                    inputs, labels = batch
                    # Generate simple sequential IDs if not provided (less robust)
                    image_ids = [f"batch_{dataloader.batch_sampler.sampler.first_index if hasattr(dataloader.batch_sampler.sampler, 'first_index') else 0}_{i+batch_start_idx}" for i in range(inputs.shape[0])]
                    # Warning: These IDs are only stable if dataloader order is consistent!
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements.")

                inputs = inputs.to(device)
                labels = labels.to(device) # Keep labels for accuracy calculation

                # Get teacher logits and apply softmax
                logits = self.teacher_model(inputs)
                softmax_probs = F.softmax(logits, dim=1)

                # Get top-k probabilities and indices
                # Use softmax_probs here, not raw logits, as per paper description context
                top_k_values, top_k_indices = torch.topk(softmax_probs, k=self.k, dim=1)

                # Store in dictionary using image ID as key
                for i, img_id in enumerate(image_ids):
                    self.logit_storage[img_id] = (
                        top_k_values[i].cpu().numpy(), # Store probabilities
                        top_k_indices[i].cpu().numpy() # Store indices
                    )

                # Calculate batch accuracy (optional but useful)
                batch_accuracy = compute_accuracy(logits, labels) # Accuracy uses raw logits
                accuracies.append(batch_accuracy)
                batch_start_idx += inputs.shape[0]


        avg_accuracy = np.mean(accuracies) if accuracies else 0
        print(f"Precomputed logits for {len(self.logit_storage)} images. Teacher Top-1 Accuracy: {avg_accuracy*100:.2f}%")

        # Save logits if path provided
        if save_path:
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.logit_storage, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved logits to {save_path}")
            except Exception as e:
                print(f"Error saving logits to {save_path}: {e}")

    def load_logits(self, path):
        """ Loads precomputed logits from a pickle file. """
        try:
            with open(path, 'rb') as f:
                self.logit_storage = pickle.load(f)
            print(f"Loaded logits for {len(self.logit_storage)} images from {path}")
        except FileNotFoundError:
             print(f"Error: Logit file not found at {path}")
             raise # Re-raise error as this is critical
        except Exception as e:
             print(f"Error loading logits from {path}: {e}")
             raise

    def distillation_labels(self, image_ids, num_classes, device):
        """
        Generates dense soft labels based on the stored sparse logits,
        following the formula in the TinyViT paper (Eq. after Fig 4).

        Args:
            image_ids (list): List of image IDs for the current batch.
            num_classes (int): The total number of classes in the dataset.
            device: The device to create the output tensor on.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_classes) containing
                          the dense soft labels for distillation. Returns None if
                          any image ID is missing.
        """
        batch_size = len(image_ids)
        # Initialize dense soft targets tensor
        soft_targets = torch.zeros((batch_size, num_classes), device=device)
        all_found = True

        for i, img_id in enumerate(image_ids):
            img_id_key = str(img_id) # Ensure string key

            if img_id_key in self.logit_storage:
                # Retrieve stored top-k softmax probabilities and indices
                teacher_probs_topk, teacher_indices_topk = self.logit_storage[img_id_key]

                # Convert to tensors on the correct device
                teacher_probs_topk = torch.tensor(teacher_probs_topk, device=device, dtype=torch.float)
                teacher_indices_topk = torch.tensor(teacher_indices_topk, device=device, dtype=torch.long)

                # Assign the stored probabilities to the corresponding indices
                soft_targets[i, teacher_indices_topk] = teacher_probs_topk

                # Calculate the probability mass for the remaining classes (label smoothing)
                sum_topk_probs = teacher_probs_topk.sum()
                remaining_prob_mass = 1.0 - sum_topk_probs

                # Number of classes not in the top-k
                num_other_classes = num_classes - self.k

                if num_other_classes > 0:
                    # Calculate the uniform probability for each non-top-k class
                    smoothing_value = remaining_prob_mass / num_other_classes
                    # Ensure smoothing value is non-negative
                    smoothing_value = max(0.0, smoothing_value.item())

                    # Create a mask for non-top-k classes
                    mask = torch.ones(num_classes, device=device, dtype=torch.bool)
                    mask[teacher_indices_topk] = False

                    # Assign the smoothing value to non-top-k classes
                    soft_targets[i, mask] = smoothing_value
                elif remaining_prob_mass > 1e-6: # Handle case k == num_classes, check if sum wasn't exactly 1
                     print(f"Warning: k={self.k} equals num_classes={num_classes}, but sum of top-k probs is {sum_topk_probs} for ID {img_id_key}. Renormalizing.")
                     soft_targets[i] /= sum_topk_probs # Renormalize if needed

                # Optional: Verify sum is close to 1
                # if not torch.isclose(soft_targets[i].sum(), torch.tensor(1.0, device=device)):
                #     print(f"Warning: Soft target sum is {soft_targets[i].sum()} for ID {img_id_key}")

            else:
                print(f"Error: Logit data not found for image ID: {img_id_key}. Cannot generate soft target.")
                # Handle missing data: return zeros, raise error, or skip sample?
                # Returning zeros for now, but this indicates a problem in precomputation/ID matching.
                soft_targets[i, :] = 0.0 # Or handle as error
                all_found = False
                # Consider raising an error here if all logits MUST be present
                # raise KeyError(f"Logit data not found for image ID: {img_id_key}")

        if not all_found:
             print("Warning: Some image IDs were missing from the logit storage.")
             # Decide how to handle this in the training loop - perhaps skip the batch?
             # For now, the function returns the tensor which might have zero rows.

        return soft_targets

    # Deprecating the old distillation_loss as it didn't match the paper's label method
    # def distillation_loss(self, student_logits, image_ids, device):
    #     """ Deprecated: Calculates KL div against sparse teacher logits.
    #         Use `distillation_labels` to get dense targets for a standard loss function instead.
    #     """
    #     # ... (previous implementation) ...
    #     print("Warning: `distillation_loss` is deprecated. Use `distillation_labels`.")
    #     return torch.tensor(0.0, device=device)
