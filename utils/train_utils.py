import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def compute_accuracy(prediction,gt_labels):
    preds = torch.argmax(prediction, dim=1)
    correct = torch.eq(preds, gt_labels).sum().item()
    accuracy = correct / gt_labels.size(0)
    return accuracy

def load_pth(model, name):
    state = torch.load(f'{name}.pth')
    model.load_state_dict(state)

def save_pth(model, name):
    torch.save(model.state_dict(), f'{name}.pth')

def train_loop(model, dataloader, optimizer, scheduler, loss_fn, max_iter, name):
    writer = SummaryWriter(log_dir=f"runs/{name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    best_acc = 0.0

    for iter in range(max_iter):
        model.train()
        for im, label in dataloader["train"]:
            im = im.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(im)
            loss = loss_fn(pred,label)
            accuracy = compute_accuracy(pred,label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            del im, label
        scheduler.step()

        model.eval()
        cur_val_losses = []
        cur_val_accuracies = []
        for im, label in dataloader["val"]:
            im = im.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.no_grad():
                pred = model(im)
                loss = loss_fn(pred, label)
                accuracy = compute_accuracy(pred,label)
                cur_val_losses.append(loss.item())
                cur_val_accuracies.append(accuracy)
            del im, label

        avg_val_loss = np.array(cur_val_losses).mean()
        avg_val_accuracies = np.array(cur_val_accuracies).mean()
        writer.add_scalar('Loss', avg_val_loss, iter)
        writer.add_scalar('Accuracy', avg_val_accuracies, iter)

        # print out stats and save best model every 10 its
        if iter % 10 == 0:
            print(f'Iteration: {iter} | Loss: {avg_val_loss} | Accuracy: {avg_val_accuracies}')
            if avg_val_accuracies > best_acc:
                save_pth(model, name)
                best_acc = avg_val_accuracies


def train_with_distillation(model, dataloaders, optimizer, scheduler, loss_fn, 
                          distiller, max_iter, name):
    writer = SummaryWriter(log_dir=f"runs/{name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    best_acc = 0.0

    for iter in range(max_iter):
        model.train()
        for im, label, ids in dataloaders["train"]:
            im = im.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(im)
            
            # Distillation loss
            distillation_labels = distiller.distillation_labels(pred, ids, device)
            loss = loss_fn(pred,distillation_labels)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            del im
        scheduler.step()

        model.eval()
        val_losses = []
        val_accuracies = []
        for im, label, ids in dataloaders["val"]:
            im = im.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            with torch.no_grad():
                pred = model(im)
                loss = loss_fn(pred,label)
                accuracy = compute_accuracy(pred, label)
                val_losses.append(loss.item())
                val_accuracies.append(accuracy)
            
            del im, label

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