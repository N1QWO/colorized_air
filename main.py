
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import torch
    from model import ColorizationModel as Model, Trainer
    num_classes = 10  # Количество распознаваемых классов
    learning_rate = 0.001
    batch_size = 32
    epochs = 10

    # 3. Подготовка данных
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root='path/to/train_data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ImageFolder(root='path/to/val_data', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = Model()
    optimazer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model,optimazer,device)