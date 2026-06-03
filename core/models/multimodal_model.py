import torch
import torch.nn as nn
from timm import create_model
from transformers import Wav2Vec2Model

from core.config import MultimodalConfig


class CrossModalAttention(nn.Module):
    def __init__(self, dim_visual=1536, dim_audio=768, dim_shared=512):
        super().__init__()

        self.proj_v = nn.Sequential(
            nn.Linear(dim_visual, dim_shared),
            nn.LayerNorm(dim_shared),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.proj_a = nn.Sequential(
            nn.Linear(dim_audio, dim_shared),
            nn.LayerNorm(dim_shared),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=dim_shared,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim_shared)

        self.ffn = nn.Sequential(
            nn.Linear(dim_shared, dim_shared * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim_shared * 2, dim_shared)
        )

        self.norm2 = nn.LayerNorm(dim_shared)

    def forward(self, visual_feats, audio_feats):
        v = self.proj_v(visual_feats)
        a = self.proj_a(audio_feats)

        attn_output, _ = self.attn(
            query=v,
            key=a,
            value=a,
            need_weights=False
        )

        out = self.norm1(v + attn_output)
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)

        return out


class MultimodalDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        print("-> Loading Visual Encoder: EfficientNet-B3")
        self.visual_backbone = create_model(
            MultimodalConfig.EFFICIENTNET_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        # Freeze gần hết visual backbone để phù hợp RTX 3050.
        for param in self.visual_backbone.parameters():
            param.requires_grad = False

        # Chỉ fine-tune block cuối.
        for name, param in self.visual_backbone.named_parameters():
            if (
                "blocks.6" in name
                or "conv_head" in name
                or "bn2" in name
            ):
                param.requires_grad = True

        print("-> Loading Audio Encoder: Wav2Vec2 Base")
        self.audio_backbone = Wav2Vec2Model.from_pretrained(
            MultimodalConfig.WAV2VEC_MODEL_NAME
        )

        # Freeze toàn bộ trước.
        for param in self.audio_backbone.parameters():
            param.requires_grad = False

        # Chỉ mở 2 layer cuối để giảm overfit và giảm VRAM.
        for name, param in self.audio_backbone.named_parameters():
            if "encoder.layers.10" in name or "encoder.layers.11" in name:
                param.requires_grad = True

        self.cross_attn = CrossModalAttention(
            dim_visual=MultimodalConfig.DIM_VISUAL,
            dim_audio=MultimodalConfig.DIM_AUDIO,
            dim_shared=MultimodalConfig.DIM_SHARED
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(MultimodalConfig.DIM_AUDIO, MultimodalConfig.DIM_SHARED),
            nn.LayerNorm(MultimodalConfig.DIM_SHARED),
            nn.GELU()
        )

        # Attention pooling thay mean pooling.
        # Model tự học frame nào quan trọng.
        self.temporal_score = nn.Sequential(
            nn.Linear(MultimodalConfig.DIM_SHARED, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(MultimodalConfig.DIM_SHARED, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, video_frames, audio_waveform):
        # video_frames: (B, F, C, H, W)
        # audio_waveform: (B, audio_len)

        B, F, C, H, W = video_frames.shape

        x = video_frames.view(B * F, C, H, W)

        v_features = self.visual_backbone(x)
        v_features = v_features.view(B, F, -1)

        a_features = self.audio_backbone(audio_waveform).last_hidden_state

        fused = self.cross_attn(v_features, a_features)

        # Attention pooling thay vì mean pooling.
        attn_weights = torch.softmax(self.temporal_score(fused), dim=1)
        pooled_features = torch.sum(fused * attn_weights, dim=1)

        logits = self.classifier(pooled_features).squeeze(-1)

        # Embedding dùng cho AV contrastive loss.
        v_emb = pooled_features
        a_emb = self.audio_proj(a_features.mean(dim=1))

        return logits, v_emb, a_emb