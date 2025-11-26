import gradio as gr
from PIL import Image
import json
import os

dataset_folder = "dataset/my_dataset"
os.makedirs(dataset_folder, exist_ok=True)
labels_file = os.path.join(dataset_folder, "labels.json")

# Load labels cũ nếu có
if os.path.exists(labels_file):
    with open(labels_file, "r") as f:
        labels = json.load(f)
else:
    labels = []

# Hàm xử lý ảnh upload và tạo label
def label_watermark(img, wm_type, x_min, y_min, x_max, y_max, intensity):
    img_name = f"img_{len(labels)+1}.png"
    img_path = os.path.join(dataset_folder, img_name)
    img.save(img_path)
    
    labels.append({
        "image_path": img_path,
        "watermark": {
            "type": wm_type,
            "bbox": [x_min, y_min, x_max, y_max],
            "intensity": intensity
        }
    })
    
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=4)
    return f"Saved {img_name} with label!"

wm_types = ["Text", "Logo", "Pattern", "No Watermark"]
intensities = ["Low", "Medium", "High"]

interface = gr.Interface(
    fn=label_watermark,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(wm_types, label="Watermark Type"),
        gr.Number(label="x_min", value=0),
        gr.Number(label="y_min", value=0),
        gr.Number(label="x_max", value=100),
        gr.Number(label="y_max", value=100),
        gr.Dropdown(intensities, label="Intensity")
    ],
    outputs=gr.Textbox(label="Status"),
    title="Label Watermark Dataset",
    description="Upload ảnh, chọn loại watermark, bounding box và intensity."
)

interface.launch()
