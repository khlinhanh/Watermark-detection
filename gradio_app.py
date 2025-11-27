import gradio as gr
from PIL import Image
from src.predict import predict_and_visualize, LABELS
from src.model import WatermarkNet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WatermarkNet(num_classes=4).to(device)
model.load_state_dict(torch.load("../weights/watermark_model.pth", map_location=device))

def gr_image(img):
    pil = Image.fromarray(img.astype('uint8'))
    overlay, cls, intensity, bbox = predict_and_visualize(model, pil, device, transform=None)
    return overlay, f"{LABELS[cls]} | Intensity: {intensity:.2f}"

interface = gr.Interface(gr_image,
                 inputs=gr.Image(type="numpy", label="Upload Image"),
                 outputs=[gr.Image(label="Overlay"), gr.Textbox(label="Result")],
                 title="Watermark Detection (Image)")

interface.launch(share=True)
