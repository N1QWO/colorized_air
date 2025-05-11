import gradio as gr
from trainer import Trainer
from model import ColorizationModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorizationModel()
model = model.to(device)
model.load_state_dict(torch.load("models/colorization_model.pth"))
model.eval()

trainer = Trainer(model, device)

def predict_and_return(path):
    result = trainer.predict(path)
    return result

# Запуск веб-интерфейса
demo = gr.Interface(
    fn=predict_and_return,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(type="pil"),
    title="Colorization AI",
    description="Загрузите черно-белое изображение, и модель попытается его раскрасить."
)

if __name__ == "__main__":
    demo.launch()