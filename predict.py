import torch
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt

LABELS = ["Text","Logo","Pattern","No Watermark"]

def predict_and_visualize(model, pil_img, device, transform, show_cam=True):
    model.eval()
    img_t = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_cls, out_reg, out_box = model(img_t)
        cls_idx = out_cls.argmax(dim=1).item()
        intensity = out_reg.item()
        bbox = out_box[0].cpu().numpy()

    # Vẽ bounding box
    img_vis = np.array(pil_img).copy()
    H,W = img_vis.shape[:2]
    x,y,w,h = bbox
    x = int(x/224*W); y = int(y/224*H)
    w = int(w/224*W); h = int(h/224*H)
    cv2.rectangle(img_vis, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.putText(img_vis, f"{LABELS[cls_idx]} {intensity:.2f}", (x,max(20,y-5)),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    return img_vis, cls_idx, intensity, bbox
