import torch
import torch.nn as nn
from core.networks.efficientnet_b3 import VisualEncoder
from core.networks.wav2vec_base import AudioEncoder

class CrossModalAttention(nn.Module):
    def __init__(self, dim_visual=1536, dim_audio=768, dim_shared=512):
        super(CrossModalAttention, self).__init__()
        # Đưa đặc trưng của 2 nhánh về cùng một số chiều (dim_shared) để tính Attention
        self.proj_visual = nn.Linear(dim_visual, dim_shared)
        self.proj_audio = nn.Linear(dim_audio, dim_shared)
        
        # Lớp Multihead Attention: Ép nhánh Ảnh (Query) phải đối chiếu với nhánh Tiếng (Key, Value)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim_shared, num_heads=8, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(dim_shared)

    def forward(self, visual_feats, audio_feats):
        # visual_feats: (Batch_Size, 30, 1536) -> 30 frames mặt
        # audio_feats: (Batch_Size, Time_Steps, 768) -> Chuỗi âm thanh
        
        v_proj = self.proj_visual(visual_feats) # (Batch_Size, 30, 512)
        a_proj = self.proj_audio(audio_feats)   # (Batch_Size, Time_Steps, 512)
        
        # Tính toán sự tương quan giữa Môi (Q) và Tiếng (K, V)
        attn_output, _ = self.multihead_attn(query=v_proj, key=a_proj, value=a_proj)
        
        # Kết hợp với kết nối tắt (Residual Connection) và Chuẩn hóa
        output = self.layer_norm(attn_output + v_proj)
        return output # Kích thước: (Batch_Size, 30, 512)

class DualStreamFusionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(DualStreamFusionModel, self).__init__()
        # Khởi tạo 2 nhánh trích xuất đặc trưng nguyên bản của bạn
        self.visual_encoder = VisualEncoder() # Đầu ra lớp conv cuối là 1536
        self.audio_encoder = AudioEncoder(pretrained=pretrained)   # Đầu ra mặc định là 768
        
        # Lớp ép đồng bộ thời gian (Học tập tư duy của LipFD)
        self.cross_attention = CrossModalAttention(dim_visual=1536, dim_audio=768, dim_shared=512)
        
        # Khối phân loại cuối cùng (Classifier) để đưa ra kết luận Real hay Fake
        self.classifier = nn.Sequential(
            nn.Linear(512 * 30, 1024), # Phẳng hóa (Flatten) 30 frames x 512 đặc trưng
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 2) # Đầu ra là 2 lớp: 0 (Real) và 1 (Fake)
        )

    def forward(self, video_tensor, audio_tensor):
        # video_tensor: (Batch_Size, 3, 30, 300, 300)
        # audio_tensor: (Batch_Size, 48000)
        
        batch_size = video_tensor.size(0)
        
        # 1. Trích xuất đặc trưng hình ảnh cho từng frame một
        # Biến đổi hình học để đưa vào EfficientNet: (Batch_Size * 30, 3, 300, 300)
        t_shapes = video_tensor.permute(0, 2, 1, 3, 4).contiguous()
        t_shapes = t_shapes.view(-1, 3, 300, 300)
        
        visual_features = self.visual_encoder(t_shapes) # (Batch_Size * 30, 1536)
        visual_features = visual_features.view(batch_size, 30, 1536) # Đưa về dạng chuỗi thời gian
        
        # 2. Trích xuất đặc trưng âm thanh từ sóng thô
        audio_features = self.audio_encoder(audio_tensor) # (Batch_Size, Time_Steps, 768)
        
        # 3. Thực hiện Fusion bằng Cross-Attention để bắt lỗi lệch pha
        fused_features = self.cross_attention(visual_features, audio_features) # (Batch_Size, 30, 512)
        
        # 4. Phẳng hóa dữ liệu để đưa qua lớp phân loại nhị phân
        fused_flat = fused_features.view(batch_size, -1) # (Batch_Size, 512 * 30)
        logits = self.classifier(fused_flat)
        
        return logits

if __name__ == '__main__':
    # Đoạn mã kiểm tra nhanh cấu trúc mô hình tổng
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy thử nghiệm mô hình lai Dual-Stream trên: {device}")
    
    # Giả lập dữ liệu đầu vào từ DataLoader: 2 video, mỗi video có 30 ảnh mặt 300x300 và 3 giây tiếng
    dummy_video = torch.randn(2, 3, 30, 300, 300).to(device)
    dummy_audio = torch.randn(2, 48000).to(device)
    
    # Khởi tạo mô hình (Tắt pretrained để chạy test trên CPU/GPU cục bộ cho nhanh)
    model = DualStreamFusionModel(pretrained=False).to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(dummy_video, dummy_audio)
        
    print(f"Kích thước đầu ra của lớp Classifier: {output.shape}")
    print("-> Cấu trúc mạng tổng DualStreamFusionModel đã thông suốt hoàn toàn!")