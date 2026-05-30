import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import MultimodalConfig
from core.dataset.data_loader import get_multimodal_loaders
from core.models.multimodal_model import MultimodalDeepfakeDetector


def contrastive_loss(v_embeds, a_embeds, labels):
    # Chuẩn hóa ma trận vector L2 chống tràn số thực
    v_norm = torch.nn.functional.normalize(v_embeds, dim=-1, eps=1e-8)
    a_norm = torch.nn.functional.normalize(a_embeds, dim=-1, eps=1e-8)
    
    # Tính Cosine Similarity chuẩn trong biên [-1, 1]
    sim = torch.sum(v_norm * a_norm, dim=-1)
    sim = torch.clamp(sim, min=-0.99, max=0.99)
    
    # Ép logic mục tiêu tối ưu: Real (0) -> Đích 1 (Đồng bộ), Fake (1) -> Đích 0 (Đẩy xa)
    target = 1.0 - labels.float()
    
    # Tính toán BCE Loss bọc Sigmoid ổn định đồ thị đạo hàm tối đa
    return nn.get_submodule if hasattr(nn, 'get_submodule') else nn.functional.binary_cross_entropy_with_logits(sim, target)


def train_epoch(model, loader, bce_criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (video, audio, labels) in enumerate(tqdm(loader, desc="Training")):
        video = video.to(device)
        audio = audio.to(device)
        labels = labels.to(device).float().view(-1)
        
        optimizer.zero_grad()
        logits, v_emb, a_emb = model(video, audio)
        
        logits = torch.nan_to_num(logits, nan=0.0)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        loss_bce = bce_criterion(logits, labels)
        loss_con = contrastive_loss(v_emb, a_emb, labels)
        
        loss = loss_bce + 0.0362 * loss_con
        
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * video.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 25 == 0:
            print(f"   Batch {batch_idx:3d} | BCE: {loss_bce.item():.4f} | Con: {loss_con.item():.4f} | "
                  f"Logits: {logits.mean().item():.4f} | Pred: {preds.mean().item():.3f}")
    
    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


def validate(model, loader, bce_criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for video, audio, labels in tqdm(loader, desc="Validating"):
            video = video.to(device)
            audio = audio.to(device)
            labels = labels.to(device).float().view(-1)
            
            logits, v_emb, a_emb = model(video, audio)
            logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            loss_bce = bce_criterion(logits, labels)
            loss_con = contrastive_loss(v_emb, a_emb, labels)
            loss = loss_bce + 0.0362 * loss_con
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            total_loss += loss.item() * video.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


def main():
    print("=== BẮT ĐẦU KHỞI CHẠY HỆ THỐNG HUẤN LUYỆN MULTIMODAL DEEPFAKE ===")
    config = MultimodalConfig()
    config.create_required_dirs()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Đang sử dụng thiết bị: {device}")
    
    train_loader, dev_loader, _ = get_multimodal_loaders(config)
    
    model = MultimodalDeepfakeDetector().to(device)
    
    # ==================== RESUME ====================
    best_path = os.path.join(config.WEIGHTS_DIR, "best_multimodal_model.pth")
    if os.path.exists(best_path):
        print(f"🔄 Đang load best model để tiếp tục train...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("✅ Load best model thành công!")
    else:
        print("⚠️ Không tìm thấy best model → Train từ đầu.")
    # ===============================================
    
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_dev_acc = 0.0
    patience = 20                    # ← TĂNG CAO ĐỂ ÍT DỪNG
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n🚀 Epoch [{epoch}/{config.EPOCHS}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, bce_criterion, optimizer, device)
        dev_loss, dev_acc = validate(model, dev_loader, bce_criterion, device)
        
        print(f"🔥 Kết quả Epoch {epoch}:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"   Dev Loss:   {dev_loss:.4f} | Dev Acc:   {dev_acc*100:.2f}%")
        
        scheduler.step(dev_acc)
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"   💾 Lưu best model! (Dev Acc = {dev_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"   Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"⛔ Early Stopping tại epoch {epoch}!")
                break


if __name__ == "__main__":
    main()