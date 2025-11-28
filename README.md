# Watermark Detection

## Mục lục
- [Giới thiệu](#giới-thiệu)
- [Các thành phần của dự án](#các-thành-phần-của-dự-án)
- [Cách hoạt động](#cách-hoạt-động)
- [Cài đặt](#cài-đặt)
- [Label dữ liệu](#label-dữ-liệu)
- [Huấn luyện mô hình](#huấn-luyện-mô-hình)
- [Demo Gradio](#demo-gradio)
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
- `dataset.py`: Dataset class và DataLoader, áp dụng transform, split train/val  
- `model.py`: Định nghĩa model WatermarkNet (ResNet-based, 3 head: classification, intensity, bbox)
- `train.py`: Vòng lặp huấn luyện, loss function, optimizer, training pipeline
- `predict.py`: Hàm predict ảnh/video, vẽ overlay, Grad-CAM  
- `gradio_app.py`: Chạy Gradio interface cho ảnh/video  
- `labels.csv & labels.json`: Dataset labels lưu vĩnh viễn

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


## Label dữ liệu
 ```python
python dataset.py
```
Labels tự động lưu vào:
 ```
labels.csv & labels.json
```


### Huấn luyện mô hình
 ```python
python train.py
 ```
- Dataset: ảnh/video thật có watermark
- CSV format:filename,type,severity,x,y,width,height
 ```
img_1.png,0,0.66,10,20,50,30
 ```
**Dataset**:  
- Để chạy demo trên Colab, tải dataset từ Google Drive: [Link tải dataset](https://drive.google.com/drive/folders/1LSz8M5RJB8EDWvIkMKfHMLzD3mK3sT4m?usp=drive_link)  
- Colab mount Drive và đặt `dataset_folder = "/content/drive/MyDrive/watermark_dataset"`
- Nếu clone repo về local, tạo thư mục `watermark_dataset` và copy dữ liệu vào

---

## Demo gradio
 ```python
python gradio_app.py
 ```
- Tab Image: upload ảnh, nhận overlay + Grad-CAM + type/intensity
- Tab Video: upload video, xuất video highlight watermark

---


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
