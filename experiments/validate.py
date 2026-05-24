import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Khắc phục đường dẫn hệ thống để gọi được các file trong thư mục core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dataset.data_loader import get_multimodal_dataloader
from core.networks.dual_stream import DualStreamFusionModel

def evaluate_model(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\n--- Kích hoạt chu trình kiểm thử dữ liệu đa phương tiện ---")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing Batches"):
            videos = batch['video'].to(device)
            audios = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            # Dự đoán từ mô hình lai
            outputs = model(videos, audios)
            _, preds = torch.max(outputs, 1)
            
            # Thu thập kết quả để tính toán chỉ số
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị đánh giá: {device}")
    
    # 1. Cấu hình đường dẫn dữ liệu kiểm thử
    # BƯỚC ĐẦU: Chạy trên tập test của LAV-DF
    TEST_CSV = r"D:\Projects\Multimodal-DFD\data\metadata\lavdf_test_manifest.csv"
    MODEL_WEIGHTS = r"D:\Projects\Multimodal-DFD\core\weights\best_multimodal_model.pth"
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"LỖI: Không tìm thấy file trọng số tại {MODEL_WEIGHTS}. Bạn cần chạy train.py trước!")
        return

    # 2. Khởi tạo DataLoader cho tập Test
    test_loader = get_multimodal_dataloader(TEST_CSV, batch_size=4, shuffle=False)
    
    # 3. Khởi tạo mô hình và nạp trọng số đã huấn luyện (Tắt pretrained vì ta sẽ nạp file pth)
    model = DualStreamFusionModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    print(" Nạp trọng số tối ưu (Best Checkpoint) thành công!")
    
    # 4. Chạy kiểm thử
    y_true, y_pred = evaluate_model(model, test_loader, device)
    
    # 5. Tính toán các chỉ số thống kê học thuật cho Luận văn
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    # 6. In kết quả chuẩn cấu trúc báo cáo khoa học
    print("\n" + "="*20 + " KẾT QUẢ ĐÁNH GIÁ (METRICS) " + "="*20)
    print(f" Độ chính xác tổng thể (Accuracy): {acc*100:.2f}%")
    print(f" Độ chính xác dự đoán đúng (Precision): {precision*100:.2f}%")
    print(f" Tỷ lệ bỏ sót Deepfake (Recall): {recall*100:.2f}%")
    print(f" Chỉ số cân bằng F1-Score: {f1*100:.2f}%")
    print("="*68)
    
    print("\n[Ma trận nhầm lẫn - Confusion Matrix]")
    print(f" True Real (Đoán đúng Real): {cm[0][0]} | False Fake (Nghi oan Real thành Fake): {cm[0][1]}")
    print(f" False Real (Lọt lưới Deepfake): {cm[1][0]} | True Fake (Bắt trúng Deepfake): {cm[1][1]}")
    print("="*68)

if __name__ == '__main__':
    main()