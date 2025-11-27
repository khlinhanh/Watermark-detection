import torch
import torch.optim as optim
import torch.nn as nn

def train_epoch(model, loader, optimizer, criterions, device):
    model.train()
    total_loss = 0
    for imgs, labels, severities, bboxes in loader:
        imgs, labels, severities, bboxes = imgs.to(device), labels.to(device), severities.to(device).unsqueeze(1), bboxes.to(device)
        optimizer.zero_grad()
        out_cls, out_reg, out_box = model(imgs)
        loss = criterions['cls'](out_cls, labels) + criterions['reg'](out_reg, severities) + criterions['box'](out_box, bboxes)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
