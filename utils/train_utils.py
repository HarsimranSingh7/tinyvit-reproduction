import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime

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
    writer = SummaryWriter(log_dir=f"runs/{name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}")

    best_acc = float('inf')

    for iter in range(max_iter):
        model.train()
        for im, label in dataloader["train"]:
            optimizer.zero_grad()
            pred = model(im)
            loss = loss_fn(pred,label)
            accuracy = compute_accuracy(pred,label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

        model.eval()
        cur_val_losses = []
        cur_val_accuracies = []
        for im, label in dataloader["val"]:
            with torch.no_grad():
                pred = model(im)
                loss = loss_fn(pred, label)
                accuracy = compute_accuracy(pred,label)
                cur_val_losses.append(loss.item())
                cur_val_accuracies.append(accuracy.item())

        avg_val_loss = np.array(cur_val_losses).mean()
        avg_val_accuracies = np.array(cur_val_accuracies).mean()
        writer.add_scalar('Loss', avg_val_loss, iter)
        writer.add_scalar('Accuracy', avg_val_accuracies, iter)

        # print out stats and save best model every 10 its
        if iter % 10 == 0:
            print(f'Iteration: {iter} | Loss: {avg_val_loss} | Accuracy: {avg_val_accuracies}')
            if avg_val_accuracies < best_acc:
                save_pth(model, name)
                best_acc = avg_val_accuracies
