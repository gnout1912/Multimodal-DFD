import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
import time

# Khắc phục đường dẫn hệ thống để gọi được các file trong thư mục core Windows
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import MultimodalConfig
from core.dataset.data_loader import get_multimodal_loaders
from core.models.multimodal_model import MultimodalDeepfakeDetector

# =========================================================================
# HÀM LOSS TƯƠNG PHẢN ĐÃ VÁ LỖI TOÁN HỌC CHỐNG NAN TUYỆT ĐỐI
# =========================================================================
def contrastive_loss(v_embeds, a_embeds, labels):
    # Chuẩn hóa ma trận vector L2 chống chia cho số 0
    v_norm = torch.nn.functional.normalize(v_embeds, dim=-1, eps=1e-8)
    a_norm = torch.nn.functional.normalize(a_embeds, dim=-1, eps=1e-8)
    
    # Tính Cosine Similarity chuẩn trong biên an toàn [-1, 1]
    sim = torch.sum(v_norm * a_norm, dim=-1)
    sim = torch.clamp(sim, min=-0.99, max=0.99)
    
    # Logic mục tiêu tối ưu: Real (0) -> Đích 1 (Kéo gần), Fake (1) -> Đích 0 (Đẩy xa)
    target = 1.0 - labels.float()
    
    # Tính toán BCE Loss bọc Sigmoid ổn định đồ thị đạo hàm tối đa
    return torch.nn.functional.binary_cross_entropy_with_logits(sim, target)

# =========================================================================
# HÀM MỤC TIÊU (OBJECTIVE FUNCTION) CHO OPTUNA CHẠY TỪNG TRIAL
# =========================================================================
def objective(trial):
    # Định nghĩa vùng không gian tìm kiếm tham số an toàn cho GPU Laptop
    lr = trial.suggest_float('lr', 1e-5, 1.5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 3e-4, log=True)
    
    # GIỚI HẠN CHUẨN: Chỉ cho phép chọn Batch Size 4 hoặc 6 theo đúng yêu cầu của bạn
    batch_size = trial.suggest_categorical('batch_size', [4, 6]) 
    contrastive_weight = trial.suggest_float('contrastive_weight', 0.01, 0.1)
    
    print(f"\n{'='*90}")
    print(f"🔍 ĐANG CHẠY TRIAL {trial.number + 1}")
    print(f"   🔹 Learning Rate    : {lr:.2e}")
    print(f"   🔹 Weight Decay     : {weight_decay:.2e}")
    print(f"   🔹 Batch Size       : {batch_size}")
    print(f"   🔹 Contrastive W    : {contrastive_weight:.3f}")
    print(f"{'='*90}")
    
    # Nạp cấu hình toàn cục
    config = MultimodalConfig()
    config.LEARNING_RATE = lr
    config.WEIGHT_DECAY = weight_decay
    config.BATCH_SIZE = batch_size
    config.NUM_WORKERS = 0 # Ép bằng 0 trên Windows để chống lỗi đóng băng đa luồng ngầm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() # Xả sạch bộ nhớ đệm trước khi khởi chạy
    
    # Gọi bộ nạp dữ liệu Subset local
    train_loader, dev_loader, _ = get_multimodal_loaders(config)
    
    # Khởi tạo mô hình mạng lai mới đã sửa Classifier phẳng hóa thời gian 30 frames
    model = MultimodalDeepfakeDetector().to(device)
    
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                            lr=lr, weight_decay=weight_decay)
    
    start_time = time.time()
    
    # Huấn luyện nhanh 3 Epoch cho mỗi Trial để đánh giá tốc độ hội tụ của bộ siêu tham số
    for epoch in range(3):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/3", leave=False)
        for video, audio, labels in pbar:
            video = video.to(device)
            audio = audio.to(device)
            # Ép dẹt nhãn tuyệt đối về dạng ma trận cột cố định 2D để triệt tiêu lỗi sập chiều batch cuối
            labels = labels.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            logits, v_emb, a_emb = model(video, audio)
            
            # Ép logits về ma trận cột 2D [Batch_size, 1] ăn khớp hoàn hảo với nhãn đích
            logits = logits.view(-1, 1)
            
            loss_bce = bce_criterion(logits, labels)
            loss_con = contrastive_loss(v_emb, a_emb, labels.view(-1))
            
            # Tính toán Loss tổng hợp kép với trọng số tối ưu của từng trial
            loss = loss_bce + contrastive_weight * loss_con
            
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
                
            loss.backward()
            # Gradient Clipping bảo vệ chống nổ số thực phát sinh nan ngầm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        # Giải phóng bộ nhớ đệm GPU cuối mỗi Epoch để bảo vệ tài nguyên
        torch.cuda.empty_cache()
    
    # Tiến hành kiểm định mô hình (Validation) sau 3 Epoch
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for video, audio, labels in dev_loader:
            video = video.to(device)
            audio = audio.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            logits, _, _ = model(video, audio)
            logits = logits.view(-1, 1)
            
            # Tính chỉ số chính xác qua hàm kích hoạt Sigmoid nhị phân ma trận cột
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    acc = correct / total if total > 0 else 0.0
    total_time = (time.time() - start_time) / 60
    
    print(f"   ➔ [KẾT QUẢ TRIAL] Dev Accuracy: {acc*100:.2f}% | Thời gian cày: {total_time:.1f} phút")
    print(f"{'='*90}\n")
    
    # Dọn dẹp RAM/VRAM chuẩn bị bàn giao cho Trial tiếp theo
    del model, optimizer, train_loader, dev_loader
    torch.cuda.empty_cache()
    
    return acc

# =========================================================================
# TIẾN TRÌNH KHỞI CHẠY CHÍNH THỨC
# =========================================================================
if __name__ == "__main__":
    print("🚀 BẮT ĐẦU CHẠY SIÊU TỰ ĐỘNG TUNING THAM SỐ OPTUNA PHIÊN BẢN BỌC GIÁP AN TOÀN")
    print("Mục tiêu: Quét sạch lỗi, tối ưu Cross-Attention bứt phá Accuracy > 90%\n")
    
    # ĐÃ SỬA CHÍ MẠNG: Thiết lập lưu trữ SQLite cục bộ để ghi nhớ tiến độ vĩnh viễn
    db_path = "sqlite:///optuna_study.db"
    
    study = optuna.create_study(
        study_name="multimodal_dfd_tuning", 
        direction='maximize', 
        sampler=TPESampler(seed=42),
        storage=db_path,
        load_if_exists=True # Nếu phát hiện file .db cũ sẽ tự động bốc đầu cày tiếp tục luôn!
    )
    
    # Thêm cờ catch phòng ngừa: Nếu dính Trial nào bị tràn VRAM khi chọn batch=6, Optuna tự bỏ qua và tiếp tục vĩnh viễn không sập nguồn!
    study.optimize(objective, n_trials=10, catch=(torch.OutOfMemoryError, Exception)) 
    
    print("\n" + "="*90)
    print("🎉 HOÀN THÀNH QUÁ TRÌNH KHẢO SÁT THAM SỐ TỐI ƯU!")
    print(f" 🏆 Độ chính xác Dev tốt nhất tìm được: {study.best_value*100:.2f}%")
    print(" 🏆 Bộ siêu tham số lý tưởng nhất dành cho mô hình:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"   🔹 {key:18}: {value:.2e}" if 'lr' in key or 'decay' in key else f"   🔹 {key:18}: {value:.4f}")
        else:
            print(f"   🔹 {key:18}: {value}")
    print("="*90)