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
        v = self.proj_v(visual_feats)
        a = self.proj_a(audio_feats)
        
        # Bảo vệ NaN trước attention
        v = torch.nan_to_num(v, nan=0.0)
        a = torch.nan_to_num(a, nan=0.0)
        
        attn_output, _ = self.attn(query=v, key=a, value=a)
        output = self.norm(v + self.dropout(attn_output))
        return output


class MultimodalDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Visual Backbone - Freeze để ổn định
        self.visual_backbone = create_model('efficientnet_b3', pretrained=True, num_classes=0)
        for param in self.visual_backbone.parameters():
            param.requires_grad = False
        
        # Chỉ unfreeze classifier cuối
        for param in list(self.visual_backbone.parameters())[-15:]:
            param.requires_grad = True

        # Audio Backbone
        self.audio_backbone = Wav2Vec2Model.from_pretrained(MultimodalConfig.WAV2VEC_MODEL_NAME)
        for param in list(self.audio_backbone.parameters())[:-20]:
            param.requires_grad = False

        self.cross_attn = CrossModalAttention()
        self.audio_proj = nn.Linear(768, 512)
        
        # Classifier đơn giản và ổn định
        self.classifier = nn.Sequential(
            nn.Linear(512 * MultimodalConfig.MAX_FRAMES, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # Khởi tạo bias
        nn.init.constant_(self.classifier[-1].bias, 0.0)

    def forward(self, video_frames, audio_waveform):
        B, F, C, H, W = video_frames.shape
        
        # Visual
        x = video_frames.view(B * F, C, H, W)
        with torch.no_grad():
            v_features = self.visual_backbone(x)
        v_features = v_features.view(B, F, -1)
        v_features = torch.nan_to_num(v_features, nan=0.0)

        # Audio
        with torch.no_grad():
            a_features = self.audio_backbone(audio_waveform).last_hidden_state
        a_features = torch.nan_to_num(a_features, nan=0.0)

        # Fusion
        fused = self.cross_attn(v_features, a_features)
        fused = torch.nan_to_num(fused, nan=0.0)
        
        fused_flat = fused.view(B, -1)
        
        logits = self.classifier(fused_flat).squeeze(-1)
        
        # Embeddings
        v_emb = fused.mean(dim=1)
        a_emb = self.audio_proj(a_features.mean(dim=1))
        
        return logits, v_emb, a_emb