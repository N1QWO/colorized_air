import torch
import torch.optim as optim
from tqdm import tqdm
from utils import visualize_prediction
from PIL import Image
from skimage import color
from torchvision import transforms
import numpy as np

class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def fit(self, train_loader, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for L, ab in progress_bar:
                L, ab = L.to(self.device), ab.to(self.device)
                ab = ab / 110.0  # 🔥 Добавь нормализацию здесь
                pred_ab = self.model(L)
                loss = self.criterion(pred_ab, ab)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
        torch.save(self.model.state_dict(), "models/colorization_model.pth")

    def predict(self, image_path):
        self.model.eval()
        with torch.no_grad():
            img = Image.open(image_path).convert("RGB")
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)

            # Конвертируем в LAB
            img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            lab = color.rgb2lab(img_np)
            L = torch.from_numpy(lab[:, :, 0]).unsqueeze(0).unsqueeze(0).float().to(self.device)
            L_norm = L / 50.0 - 1.0

            pred_ab = self.model(L_norm)
            pred_ab = pred_ab * 110.0

            # Собираем LAB
            lab_pred = torch.cat([L, pred_ab], dim=1).squeeze().permute(1, 2, 0).cpu().numpy()
            lab_pred[:, :, 0] = np.clip(lab_pred[:, :, 0], 0, 100)
            lab_pred[:, :, 1:] = np.clip(lab_pred[:, :, 1:], -128, 127)

            rgb_pred = (color.lab2rgb(lab_pred) * 255).astype(np.uint8)
            return Image.fromarray(rgb_pred)