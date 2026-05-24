import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Khắc phục đường dẫn hệ thống để gọi được các file trong thư mục core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dataset.data_loader import get_multimodal_dataloader
from core.networks.dual_stream import DualStreamFusionModel

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    # Vòng lặp quét qua từng Batch dữ liệu đa phương tiện
    for batch in tqdm(dataloader, desc="Training Batches"):
        # Đẩy dữ liệu vào GPU (RTX 3050)
        videos = batch['video'].to(device)  # Kích thước: (Batch, 3, 30, 300, 300)
        audios = batch['audio'].to(device)  # Kích thước: (Batch, 48000)
        labels = batch['label'].to(device)  # Kích thước: (Batch,)
        
        optimizer.zero_grad()
        
        # Chạy forward mạng hai nhánh + Cross-Attention
        outputs = model(videos, audios)
        loss = criterion(outputs, labels)
        
        # Chạy backward để cập nhật trọng số
        loss.backward()
        optimizer.step()
        
        # Tính toán thống kê độ chính xác
        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds.double() / total_samples
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation Batches"):
            videos = batch['video'].to(device)
            audios = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(videos, audios)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds.double() / total_samples
    return epoch_loss, epoch_acc

def main():
    # 1. Cấu hình phần cứng và tham số huấn luyện
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hệ thống kích hoạt chế độ huấn luyện trên thiết bị: {device}")
    
    # Cấu hình siêu tham số (Hyper-parameters) tối ưu cho cấu hình máy
    BATCH_SIZE = 4       # Tránh tràn VRAM 4GB của RTX 3050
    LEARNING_RATE = 1e-4 # Tốc độ học nhỏ giúp Wav2Vec2 và EfficientNet hội tụ mịn
    EPOCHS = 10
    
    # Đường dẫn tới file chỉ mục CSV từ bước tiền xử lý LAV-DF
    TRAIN_CSV = r"D:\Projects\Multimodal-DFD\data\metadata\lavdf_train_manifest.csv"
    DEV_CSV = r"D:\Projects\Multimodal-DFD\data\metadata\lavdf_dev_manifest.csv"
    WEIGHTS_DIR = r"D:\Projects\Multimodal-DFD\core\weights"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    # 2. Khởi tạo DataLoader
    print("Đang nạp tập dữ liệu đa phương tiện...")
    train_loader = get_multimodal_dataloader(TRAIN_CSV, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_multimodal_dataloader(DEV_CSV, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Khởi tạo Mô hình tổng thể
    model = DualStreamFusionModel(pretrained=True).to(device)
    
    # 4. Định nghĩa Hàm phạt và Thuật toán tối ưu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    best_val_acc = 0.0
    
    # 5. Vòng lặp chạy chính thức qua từng Epoch
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Cơ chế lưu mô hình tốt nhất (Checkpointing)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(WEIGHTS_DIR, "best_multimodal_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f" Đã lưu mô hình tối ưu mới với Val Acc: {val_acc:.4f}")

    print("\nQuá trình huấn luyện hoàn tất thành công!")

if __name__ == '__main__':
    main()