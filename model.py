import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from skimage import color
import random

class ColorizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64))  # Адаптивный пуллинг
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ColorizationDataset(Dataset):
    def __init__(self, paths, output_size=(256, 256)):  # <-- Добавлен конструктор
        self.paths = paths
        self.output_size = output_size
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            
            # Адаптивное изменение размера
            if min(img.size) < 256:
                img = transforms.Resize(256)(img)
                
            # Случайное кадрирование
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=self.output_size
            )
            img = transforms.functional.crop(img, i, j, h, w)
            
            # Применение аугментаций
            img = self.transform(img)
            
            # Преобразование в LAB
            img_lab = color.rgb2lab(np.asarray(img))  # <-- Используем np.asarray
            img_lab = torch.from_numpy(img_lab).float()  # <-- Явное преобразование
            img_lab = img_lab.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            L = img_lab[0:1, :, :] / 50.0 - 1.0
            ab = img_lab[1:3, :, :] / 110.0
            return L.float(), ab.float()
            
        except Exception as e:
            print(f"Ошибка загрузки {self.paths[idx]}: {e}")
            return None

    def __len__(self):
        return len(self.paths)


class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0))
        ])

    def fit(self, train_loader, val_loader, learning_rate=1e-4, epochs=100):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for L, ab in train_loader:
                L = L.to(self.device)
                ab = ab.to(self.device)
                
                optimizer.zero_grad()
                pred_ab = self.model(L)
                loss = criterion(pred_ab, ab)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for L, ab in val_loader:
                    L = L.to(self.device)
                    ab = ab.to(self.device)
                    pred_ab = self.model(L)
                    val_loss += criterion(pred_ab, ab).item()
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}')
        
        torch.save(self.model.state_dict(), 'colorization_model.pth')

    def predict(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        img_lab = color.rgb2lab(image)
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[0:1, :, :].unsqueeze(0).to(self.device) / 50.0 - 1.0
        
        with torch.no_grad():
            pred_ab = self.model(L).cpu().squeeze(0) * 110.0
        
        L = (L.cpu().squeeze(0) + 1.0) * 50.0
        lab_img = torch.cat([L, pred_ab], dim=0).permute(1, 2, 0).numpy()
        
        lab_img[:, :, 0] = np.clip(lab_img[:, :, 0], 0, 100)
        lab_img[:, :, 1:] = np.clip(lab_img[:, :, 1:], -128, 127)
        rgb_img = color.lab2rgb(lab_img) * 255
        rgb_img = rgb_img.astype(np.uint8)
        
        return Image.fromarray(rgb_img)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Пути к данным
    data_root = "C:/Users/mr_bs/.cache/kagglehub/datasets/nelyg8002000/commercial-aircraft-dataset/versions/1/1_Liner TF"
    train_paths = [os.path.join(data_root, fname) for fname in os.listdir(data_root)]
    
    # Создание датасетов
    train_dataset = ColorizationDataset(train_paths[:100])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Инициализация модели
    model = ColorizationModel()
    trainer = Trainer(model, device)
    
    # Обучение
    trainer.fit(train_loader, val_loader=None, learning_rate=1e-4, epochs=50)  # Добавьте val_loader
    
    # Пример предсказания
    colorized = trainer.predict("test.jpg")
    colorized.show()