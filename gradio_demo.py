import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import gradio as gr
import torch.nn.functional as F
import matplotlib.pyplot as plt

LABELS = ["Text","Logo","Pattern","No Watermark"]

def predict_and_visualize(model, pil_img, device, show_cam=True):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img_t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out_cls, out_reg, out_box = model(img_t)
    cls_idx = out_cls.argmax(dim=1).item()
    intensity = out_reg.item()
    bbox = out_box[0].cpu().numpy()
    
    img_vis = np.array(pil_img).copy()
    H,W = img_vis.shape[:2]
    x,y,w,h = bbox
    x = int(x/224*W); y=int(y/224*H); w=int(w/224*W); h=int(h/224*H)
    cv2.rectangle(img_vis, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.putText(img_vis, f"{LABELS[cls_idx]} {intensity:.2f}", (x,max(20,y-5)), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    return img_vis, cls_idx, intensity, bbox

def process_video(model, video_path, device):
    cap = cv2.VideoCapture(video_path)
    W,H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (W,H))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        vis, cls, intensity, bbox = predict_and_visualize(model, pil_img, device)
        out.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    cap.release()
    out.release()
    return "output.mp4"

def gr_image(model, img, device):
    pil = Image.fromarray(img.astype('uint8'))
    vis, cls, intensity, bbox = predict_and_visualize(model, pil, device)
    return vis, f"{LABELS[cls]} | Intensity: {intensity:.2f}"

def launch_gradio(model, device):
    interface = gr.TabbedInterface([
        gr.Interface(lambda img: gr_image(model,img,device),
                     inputs=gr.Image(type="numpy", label="Upload Image"),
                     outputs=[gr.Image(label="Overlay"), gr.Textbox(label="Result")],
                     title="Watermark Detection (Image)"),
        gr.Interface(lambda vid: process_video(model, vid, device),
                     inputs=gr.Video(label="Upload Video"),
                     outputs=gr.Video(label="Result Video"),
                     title="Watermark Detection (Video)")
    ], ["Image","Video"])
    interface.launch(share=True)
