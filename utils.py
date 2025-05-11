import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
from skimage import color
import cv2

def visualize_prediction(model, image_path, device):
    model.eval()
    model.to(device)  # гарантируем, что модель на нужном устройстве
    
    with torch.no_grad():
        img = Image.open(image_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        # Конвертируем в LAB
        img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        lab = color.rgb2lab(img_np)
        L = torch.from_numpy(lab[:, :, 0]).unsqueeze(0).unsqueeze(0).float().to(device)
        L_norm = L / 50.0 - 1.0  # Нормализация

        pred_ab = model(L_norm)
        pred_ab = pred_ab * 110.0  # Расширяем диапазон a*b*

        # 🔥 Выравниваем размер pred_ab под L
        pred_ab = torch.nn.functional.interpolate(
            pred_ab,
            size=L.shape[2:],  # H x W
            mode='bilinear',
            align_corners=False
        )

        # Теперь можно безопасно объединять
        lab_pred = torch.cat([L, pred_ab], dim=1).squeeze().permute(1, 2, 0).cpu().numpy()

        # Ограничиваем значения
        lab_pred[:, :, 0] = np.clip(lab_pred[:, :, 0], 0, 100)
        lab_pred[:, :, 1:] = np.clip(lab_pred[:, :, 1:], -128, 127)

        rgb_pred = (color.lab2rgb(lab_pred) * 255).astype(np.uint8)

        # Визуализация
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img)
        axs[0].set_title("Оригинал")
        axs[1].imshow(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY), cmap='gray')
        axs[1].set_title("Черно-белое")
        axs[2].imshow(rgb_pred)
        axs[2].set_title("Цветное (предсказание)")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ =='__main__':
    # Сначала создаём модель (замените MyModel() на реальную архитектуру)
    from model import ColorizationModel  # например

    model = ColorizationModel()
    model.load_state_dict(torch.load("models/colorization_model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    visualize_prediction(model, "TF_PLANE_36.jpg", device)