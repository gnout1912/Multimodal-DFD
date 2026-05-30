import torch
import torch.nn as nn
from timm import create_model
from transformers import Wav2Vec2Model
from core.config import MultimodalConfig

class CrossModalAttention(nn.Module):
    def __init__(self, dim_visual=1536, dim_audio=768, dim_shared=512):
        super().__init__()
        self.proj_v = nn.Linear(dim_visual, dim_shared)
        self.proj_a = nn.Linear(dim_audio, dim_shared)
        self.attn = nn.MultiheadAttention(dim_shared, num_heads=8, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim_shared)
        self.dropout = nn.Dropout(0.1)

    def forward(self, visual_feats, audio_feats):
        v = self.proj_v(visual_feats) # [B, 30, 512]
        a = self.proj_a(audio_feats)  # [B, T_audio, 512]
        
        v = torch.nan_to_num(v, nan=0.0)
        a = torch.nan_to_num(a, nan=0.0)
        
        # Bắt tương quan thời gian chéo giữa chuỗi ảnh và chuỗi tiếng
        attn_output, _ = self.attn(query=v, key=a, value=a)
        output = self.norm(v + self.dropout(attn_output))
        return output # Kích thước chuẩn chuỗi: [B, 30, 512]


class MultimodalDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        print("-> Đang tải Visual Encoder: EfficientNet-B3...")
        self.visual_backbone = create_model('efficientnet_b3', pretrained=True, num_classes=0)
        for param in self.visual_backbone.parameters():
            param.requires_grad = False
        
        # Mở khóa 15 tầng cuối của mạng ảnh để thích ứng với tập dữ liệu mới
        for param in list(self.visual_backbone.parameters())[-15:]:
            param.requires_grad = True

        print("-> Đang tải Audio Encoder: Wav2Vec2...")
        self.audio_backbone = Wav2Vec2Model.from_pretrained(MultimodalConfig.WAV2VEC_MODEL_NAME)
        for param in list(self.audio_backbone.parameters())[:-20]:
            param.requires_grad = False

        self.cross_attn = CrossModalAttention()
        self.audio_proj = nn.Linear(768, 512)
        
        # Classifier phẳng hóa thu nhận toàn bộ đặc trưng thời gian [B, 512 * 30]
        self.classifier = nn.Sequential(
            nn.Linear(512 * MultimodalConfig.MAX_FRAMES, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        nn.init.constant_(self.classifier[-1].bias, 0.0)

    def forward(self, video_frames, audio_waveform):
        B, F, C, H, W = video_frames.shape
        
        # 1. Nhánh trích xuất đặc trưng không gian ảnh
        x = video_frames.view(B * F, C, H, w if 'w' in locals() else W)
        with torch.no_grad():
            v_features = self.visual_backbone(x)
        v_features = v_features.view(B, F, -1) # Giữ nguyên chuỗi [B, 30, 1536]
        v_features = torch.nan_to_num(v_features, nan=0.0)

        # 2. Nhánh trích xuất chuỗi âm thanh thời gian
        with torch.no_grad():
            a_features = self.audio_backbone(audio_waveform).last_hidden_state
        a_features = torch.nan_to_num(a_features, nan=0.0)

        # 3. Ép tương tác không gian thời gian qua Cross-Attention
        fused = self.cross_attn(v_features, a_features)
        fused = torch.nan_to_num(fused, nan=0.0)
        
        # Phẳng hóa cục bộ để đưa vào Classifier đưa ra logits 1 chiều chuẩn BCE
        fused_flat = fused.view(B, -1)
        logits = self.classifier(fused_flat).squeeze(-1)
        
        # Rút trích vector đại diện trung bình phục vụ tính toán Loss tương phản
        v_emb = fused.mean(dim=1)
        a_emb = self.audio_proj(a_features.mean(dim=1))
        
        return logits, v_emb, a_emb