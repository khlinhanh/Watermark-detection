import torch
import torch.optim as optim

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

def train_model(model, train_loader, epochs=5, lr=1e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterions = {'cls':torch.nn.CrossEntropyLoss(),
                  'reg':torch.nn.MSELoss(),
                  'box':torch.nn.SmoothL1Loss()}
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterions, device)
        print(f"[Epoch {ep+1}/{epochs}] Loss: {loss:.4f}")
