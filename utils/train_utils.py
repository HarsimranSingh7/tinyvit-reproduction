# utils/train_utils.py
import torch
import torch.nn as nn # Import nn for loss functions
import torch.nn.functional as F # Import functional for KLDiv
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import os # For saving checkpoints

# Determine device globally (consider passing as argument for more flexibility)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

def compute_accuracy(prediction, gt_labels):
    """Computes top-1 accuracy."""
    preds = torch.argmax(prediction.cpu(), dim=1) # Move prediction to CPU for comparison
    gt_labels = gt_labels.cpu() # Ensure labels are also on CPU
    correct = torch.eq(preds, gt_labels).sum().item()
    accuracy = correct / gt_labels.size(0) if gt_labels.size(0) > 0 else 0
    return accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, name, output_dir='.'):
    """Saves a complete checkpoint including model, optimizer, scheduler state."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'{name}_checkpoint.pth')
    best_model_path = os.path.join(output_dir, f'{name}_best.pth')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} (Epoch {epoch}, Best Acc: {best_acc:.4f})")

    # Optionally save just the best model state dict separately
    # torch.save(model.state_dict(), best_model_path)

def load_checkpoint(model, optimizer, scheduler, name, output_dir='.'):
    """Loads a checkpoint and resumes training state."""
    checkpoint_path = os.path.join(output_dir, f'{name}_checkpoint.pth')
    start_epoch = 0
    best_acc = 0.0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device) # Load directly to device
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1 # Start from next epoch
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Loaded checkpoint from {checkpoint_path}. Resuming from Epoch {start_epoch}, Best Acc: {best_acc:.4f}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
    return start_epoch, best_acc

# Deprecating old save/load in favor of checkpointing
# def load_pth(model, name):
#     state = torch.load(f'{name}.pth')
#     model.load_state_dict(state)
# def save_pth(model, name):
#     torch.save(model.state_dict(), f'{name}.pth')


def train_loop(model, dataloader, optimizer, scheduler, loss_fn, start_epoch, max_epochs, name, output_dir='.'):
    """Standard training loop."""
    log_dir_path = os.path.join("runs", f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir_path)
    print(f"TensorBoard logs will be saved to: {log_dir_path}")

    # Load best accuracy from checkpoint if resuming
    _, best_acc = load_checkpoint(model, optimizer, scheduler, name, output_dir) # Use load_checkpoint to get best_acc

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_train_losses = []
        epoch_train_accuracies = []

        batch_iter = tqdm(dataloader["train"], desc=f"Epoch {epoch}/{max_epochs} [Train]")
        for im, label in batch_iter:
            im = im.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(im)
            loss = loss_fn(pred, label)

            loss.backward()
            # Gradient Clipping (as mentioned in report/common practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Calculate accuracy using hard labels
            accuracy = compute_accuracy(pred, label)
            epoch_train_losses.append(loss.item())
            epoch_train_accuracies.append(accuracy)

            # Update tqdm progress bar
            batch_iter.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy*100:.2f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
            })
            del im, label, pred, loss # Free memory

        # Log average training metrics for the epoch
        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_acc = np.mean(epoch_train_accuracies)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', avg_train_acc * 100, epoch) # Log as percentage
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)


        # Validation phase
        model.eval()
        epoch_val_losses = []
        epoch_val_accuracies = []
        batch_iter_val = tqdm(dataloader["val"], desc=f"Epoch {epoch}/{max_epochs} [Val]")
        with torch.no_grad():
            for im, label in batch_iter_val:
                im = im.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                pred = model(im)
                loss = loss_fn(pred, label)
                accuracy = compute_accuracy(pred, label) # Accuracy uses hard labels
                epoch_val_losses.append(loss.item())
                epoch_val_accuracies.append(accuracy)

                batch_iter_val.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy*100:.2f}%"
                 })
                del im, label, pred, loss # Free memory

        avg_val_loss = np.mean(epoch_val_losses)
        avg_val_acc = np.mean(epoch_val_accuracies)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Val', avg_val_acc * 100, epoch) # Log as percentage

        print(f'Epoch: {epoch} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc*100:.2f}%')


        # Update learning rate scheduler
        scheduler.step() # Step scheduler after optimizer step and validation

        # Save checkpoint and update best model if validation accuracy improved
        is_best = avg_val_acc > best_acc
        if is_best:
            best_acc = avg_val_acc
            # Save only the best model's state_dict separately
            best_model_path = os.path.join(output_dir, f'{name}_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"*** Best model saved to {best_model_path} (Val Acc: {best_acc*100:.2f}%) ***")

        # Save checkpoint periodically or at the end
        # save_checkpoint(model, optimizer, scheduler, epoch, best_acc, name, output_dir) # Save every epoch

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, max_epochs - 1, best_acc, name, output_dir)
    writer.close()
    print("Training finished.")


def train_with_distillation(model, dataloaders, optimizer, scheduler,
                            distiller, start_epoch, max_epochs, name,
                            alpha=0.5, temperature=1.0, output_dir='.'):
    """
    Training loop with knowledge distillation using precomputed sparse logits.
    Uses KL divergence loss between student and dense soft teacher targets,
    combined with Cross-Entropy loss against hard labels.

    Args:
        model: The student model.
        dataloaders: Dictionary containing 'train' and 'val' DataLoaders.
                     Train dataloader must yield (image, hard_label, image_id).
                     Val dataloader must yield (image, hard_label).
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        distiller: The FastDistillation instance (from logit_distill.py) with loaded logits.
        start_epoch (int): Epoch to start training from (for resuming).
        max_epochs (int): Total number of epochs to train.
        name (str): Base name for saving checkpoints and logs.
        alpha (float): Weight for the distillation loss (0 <= alpha <= 1). Default 0.5.
        temperature (float): Temperature for softening logits in KL divergence. Default 1.0.
        output_dir (str): Directory to save checkpoints and logs.
    """
    log_dir_path = os.path.join("runs", f"{name}_distill_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir_path)
    print(f"TensorBoard logs will be saved to: {log_dir_path}")

    # Load best accuracy from checkpoint if resuming
    _, best_acc = load_checkpoint(model, optimizer, scheduler, name, output_dir) # Use load_checkpoint

    # Standard Cross-Entropy loss for hard labels
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_train_losses = []
        epoch_train_accuracies = [] # Accuracy against HARD labels

        batch_iter = tqdm(dataloaders["train"], desc=f"Epoch {epoch}/{max_epochs} [Train Distill]")
        for im, hard_label, ids in batch_iter:
            im = im.to(device, non_blocking=True)
            hard_label = hard_label.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Get student predictions (raw logits)
            student_pred = model(im)
            num_classes = student_pred.shape[1] # Get number of classes from output

            # Get dense soft labels from the distiller
            # Ensure distiller.distillation_labels handles missing IDs gracefully if needed
            teacher_soft_labels = distiller.distillation_labels(ids, num_classes, device)

            if teacher_soft_labels is None:
                 print(f"Warning: Skipping batch due to missing teacher logits for IDs: {ids}")
                 continue # Skip batch if critical logits are missing

            # --- Calculate Combined Loss ---
            # 1. Cross-Entropy Loss with hard labels
            loss_ce = ce_loss_fn(student_pred, hard_label)

            # 2. KL Divergence Loss with soft labels
            student_log_softmax = F.log_softmax(student_pred / temperature, dim=1)
            # teacher_soft_labels are already probabilities
            loss_kl = F.kl_div(student_log_softmax, teacher_soft_labels, reduction='batchmean') * (temperature ** 2)

            # 3. Combine losses
            loss = (1.0 - alpha) * loss_ce + alpha * loss_kl

            # Backpropagation
            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Calculate accuracy using HARD labels for monitoring task performance
            accuracy = compute_accuracy(student_pred, hard_label)
            epoch_train_losses.append(loss.item())
            epoch_train_accuracies.append(accuracy)

            # Update tqdm progress bar
            batch_iter.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy*100:.2f}%", # Accuracy vs hard labels
                'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
            })

            del im, hard_label, ids, student_pred, teacher_soft_labels, loss, loss_ce, loss_kl # Free memory

        # Log average training metrics for the epoch
        avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else 0
        avg_train_acc = np.mean(epoch_train_accuracies) if epoch_train_accuracies else 0
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', avg_train_acc * 100, epoch) # Log as percentage
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # Validation phase (using hard labels only)
        model.eval()
        epoch_val_losses = []
        epoch_val_accuracies = []
        batch_iter_val = tqdm(dataloaders["val"], desc=f"Epoch {epoch}/{max_epochs} [Val]")
        # Assume val loader yields (im, label) or (im, label, ids) - only need im, label
        with torch.no_grad():
            for batch_val in batch_iter_val:
                im, label = batch_val[0], batch_val[1] # Extract image and label
                im = im.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                pred = model(im)
                # Use standard CE loss for validation against hard labels
                loss = ce_loss_fn(pred, label)
                accuracy = compute_accuracy(pred, label)
                epoch_val_losses.append(loss.item())
                epoch_val_accuracies.append(accuracy)

                batch_iter_val.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy*100:.2f}%"
                 })
                del im, label, pred, loss # Free memory

        avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else 0
        avg_val_acc = np.mean(epoch_val_accuracies) if epoch_val_accuracies else 0
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Val', avg_val_acc * 100, epoch) # Log as percentage

        print(f'Epoch: {epoch} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc*100:.2f}%')

        # Update learning rate scheduler
        scheduler.step()

        # Save checkpoint and update best model if validation accuracy improved
        is_best = avg_val_acc > best_acc
        if is_best:
            best_acc = avg_val_acc
            best_model_path = os.path.join(output_dir, f'{name}_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"*** Best model saved to {best_model_path} (Val Acc: {best_acc*100:.2f}%) ***")

        # Save checkpoint periodically
        # save_checkpoint(model, optimizer, scheduler, epoch, best_acc, name, output_dir)

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, max_epochs - 1, best_acc, name, output_dir)
    writer.close()
    print("Distillation training finished.")
