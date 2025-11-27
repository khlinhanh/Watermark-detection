import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class WatermarkDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row['filename'])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        t = int(row['type'])
        s = float(row['severity'])
        bbox = torch.tensor([row['x'], row['y'], row['width'], row['height']], dtype=torch.float32)
        return img, torch.tensor(t, dtype=torch.long), torch.tensor(s, dtype=torch.float32), bbox
