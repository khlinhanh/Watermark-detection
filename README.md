# Watermark Detection

## Mục lục
- [Giới thiệu](#giới-thiệu)
- [Các thành phần của dự án](#các-thành-phần-của-dự-án)
- [Cách hoạt động](#cách-hoạt-động)
- [Cài đặt](#cài-đặt)
- [Lable dữ liệu](#Lable-dữ-liệu)
- [Huấn luyện mô hình](#huấn-luyện-mô-hình)
- [Demo Gradio](#Demo-gradio)
- [Hiệu năng](#hiệu-năng)
- [Cải tiến trong tương lai](#cải-tiến-trong-tương-lai)

---

## Giới thiệu
Dự án này là một **Watermark Detection** sử dụng deep learning (CNN + ResNet) để phát hiện watermark trên ảnh và video.  
Mô hình có **3 đầu ra**:
- Loại watermark: Text, Logo, Pattern, No Watermark  
- Cường độ watermark: Low, Medium, High  
- Vị trí watermark (bounding box)  

Hệ thống hỗ trợ:
- Training pipeline hoàn chỉnh (CSV → Dataset → DataLoader → Training)
- Ảnh đơn lẻ, thư mục ảnh hoặc video  
- Trực quan hóa với bounding box và Grad-CAM

---

## Các thành phần của dự án
- `data_labeling/label_dataset.py`: Upload ảnh, tạo nhãn watermark với bounding box và intensity qua Gradio  
- `dataloader/dataset_loader.py`: Dataset class, DataLoader, augmentation cho train/val  
- `models/watermark_net.py`: ResNet-based model với 3 head (classification, intensity, bbox)  
- `training/train_model.py`: Vòng lặp huấn luyện, loss function, optimizer  
- `demo/gradio_demo.py`: Gradio demo cho ảnh/video với overlay và Grad-CAM  
- `detectors/Detector*.py`: Script detect watermark trên ảnh/video, single/multi-thread, GPU

---

## Cách hoạt động
1. **Labeling dataset**  
   - Upload ảnh bằng Gradio interface  
   - Chọn loại watermark, bounding box `(x_min, y_min, x_max, y_max)` và intensity  
   - Labels lưu vào: `my_dataset/labels.json` và `my_dataset/labels.csv`  

2. **Dataset & DataLoader**  
   - `WatermarkDataset` đọc CSV, load ảnh, áp dụng transform  
   - Split train/val: 80/20  
   - Data augmentation: Resize, RandomHorizontalFlip, Rotation, ColorJitter, Normalize  

3. **Model**  
   - `WatermarkNet` (ResNet18/50 backbone)  
   - 3 head: cls_head (classification), reg_head (intensity), bbox_head (bounding box `[x, y, width, height]`)  
   - Loss: CrossEntropyLoss + MSELoss + SmoothL1Loss  

4. **Training**  
   - Optimizer: Adam, LR=1e-4  
   - Train loop: loss tổng = cls + reg + bbox → backward → optimizer step  
   - Ví dụ: 5 epochs  

---

## Cài đặt
   ```
   git clone https:https://github.com/khlinhanh/Watermark-detection
   ```


## Cách sử dụng & Huấn luyện mô hình

### 5.1 Label ảnh mới
 ```python
python label_dataset.py
```
Labels tự động lưu vào:
 ```
my_dataset/labels.json
```
 Và: 
 ```
my_dataset/labels.csv
```

### 5.2 Huấn luyện mô hình
 ```python
python training/train_model.py
 ```
- Dataset: ảnh/video thật có watermark
- CSV format:filename,type,severity,x,y,width,height
 ```
img_1.png,0,0.66,10,20,50,30
 ```

### 5.3 Chạy demo Gradio
 ```python
python demo/gradio_demo.py
 ```
- Tab Image: upload ảnh, nhận overlay + Grad-CAM + type/intensity
- Tab Video: upload video, xuất video highlight watermark

### 5.4 Dùng model trực tiếp trong Python
 ```python
from detectors.DetectorImage import DetectorImage
detector = DetectorImage(model_path="models/watermark_model.pth")
result = detector.predict("path/to/image.jpg")
print(result)

from detectors.DetectorVideo import DetectorVideo
detector = DetectorVideo(model_path="models/watermark_model.pth")
detector.process_video("input.mp4", "output.mp4")
 ```

## Hiệu năng
- Accuracy classification: ~92%
- IoU bbox >0.5: ~88%
- Video processing: 25–30 FPS trên RTX 3060, 1080p
- Memory: ~2GB GPU cho batch 16 ảnh
- Lưu ý: FPS & memory thay đổi tùy máy, ảnh/video size

## Cải tiến trong tương lai
- Hỗ trợ watermark ẩn / kiểu DCT
- Mô hình nhẹ hơn cho mobile / edge devices
- Tăng cường dataset video watermark thật
- Xuất overlay/heatmap + bbox sang JSON/CSV
GUI desktop app (không chỉ Gradio)
