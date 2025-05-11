import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

class ColorizationDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = [p for p in paths if os.path.exists(p)]
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = transforms.Resize((256, 256))(img)
            img = self.transform(img)
            img_np = np.array(img).astype(np.uint8)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab_tensor = torch.from_numpy(lab).float().permute(2, 0, 1)
            L = lab_tensor[0:1] / 50.0 - 1.0
            ab = lab_tensor[1:3] / 110.0
            return L, ab
        except Exception as e:
            print(f"Ошибка загрузки {path}: {e}")
            return self[(idx + 1) % len(self)]