import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Khắc phục đường dẫn hệ thống để gọi được các file trong thư mục core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import MultimodalConfig
from core.dataset.data_loader import get_multimodal_loaders
from core.models.multimodal_model import MultimodalDeepfakeDetector

def evaluate_model(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\n--- Kích hoạt chu trình kiểm thử dữ liệu đa phương tiện (Test Set) ---")
    with torch.no_grad():
        for videos, audios, labels in tqdm(dataloader, desc="Testing Batches"):
            videos = videos.to(device)
            audios = audios.to(device)
            
            # Chạy forward lấy logits 1 chiều từ mô hình mới
            logits, _, _ = model(videos, audios)
            
            # Ép phẳng đồng bộ kích thước ma trận cột
            logits = logits.view(-1)
            
            # Tính toán xác suất Sigmoid nhị phân để đưa ra nhãn dự đoán (0.0 hoặc 1.0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            
            # Thu thập kết quả
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy().astype(int))
            
    return all_labels, all_preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị đánh giá: {device}")
    
    config = MultimodalConfig()
    MODEL_WEIGHTS = os.path.join(config.WEIGHTS_DIR, "best_multimodal_model.pth")
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ LỖI: Không tìm thấy file trọng số tại {MODEL_WEIGHTS}. Bạn cần chạy train.py trước!")
        return

    # 1. Khởi tạo hệ thống DataLoader cho tập Test (Lấy ra loader thứ 3)
    print("-> Đang nạp cấu trúc DataLoader cho tập Test thực nghiệm...")
    _, _, test_loader = get_multimodal_loaders(config)
    
    if test_loader is None:
        print("❌ LỖI: Hàm get_multimodal_loaders chưa cấu hình trả về test_loader. Hãy kiểm tra lại file data_loader.py!")
        return
        
    print(f"   Số lượng mẫu trong tập Test: {len(test_loader.dataset)} mẫu ({len(test_loader)} Batches)")
    
    # 2. Khởi tạo mô hình mới và nạp trọng số đã huấn luyện (best_multimodal_model.pth)
    model = MultimodalDeepfakeDetector().to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    print("🎯 [OK] Nạp trọng số tối ưu (Best Checkpoint) thành công!")
    
    # 3. Chạy đánh giá kiểm thử
    y_true, y_pred = evaluate_model(model, test_loader, device)
    
    # 4. Tính toán các chỉ số thống kê học thuật cho Luận văn tốt nghiệp
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # 5. In kết quả chuẩn cấu trúc báo cáo khoa học gửi Thầy
    print("\n" + "="*20 + " KẾT QUẢ ĐÁNH GIÁ THỰC NGHIỆM (METRICS) " + "="*20)
    print(f" 🔹 Độ chính xác tổng thể (Accuracy): {acc*100:.2f}%")
    print(f" 🔹 Độ chính xác dự đoán đúng (Precision): {precision*100:.2f}%")
    print(f" 🔹 Tỷ lệ bắt trúng Deepfake (Recall): {recall*100:.2f}%")
    print(f" 🔹 Chỉ số cân bằng F1-Score: {f1*100:.2f}%")
    print("="*78)
    
    # Phòng ngừa ma trận nhầm lẫn biên dạng nhỏ
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print("\n[Ma trận nhầm lẫn - Confusion Matrix]")
        print(f" ✅ True Real (Đoán đúng video Real): {tn} | ❌ False Fake (Nghi oan Real thành Fake): {fp}")
        print(f" ❌ False Real (Lọt lưới video Deepfake): {fn} | ✅ True Fake (Bắt trúng video Deepfake): {tp}")
        print("="*78)

if __name__ == '__main__':
    main()