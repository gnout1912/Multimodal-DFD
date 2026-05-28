import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CrossModalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, visual_embeds, audio_embeds, labels):
        # VÁ LỖI TOÁN HỌC 1: Thêm thuộc tính eps để bảo vệ tuyệt đối không chia cho 0 khi gặp vector rỗng
        visual_norm = F.normalize(visual_embeds, p=2, dim=-1, eps=1e-8)
        audio_norm = F.normalize(audio_embeds, p=2, dim=-1, eps=1e-8)

        # Tính Cosine Similarity
        cosine_sim = torch.sum(visual_norm * audio_norm, dim=-1)

        # VÁ LỖI TOÁN HỌC 2: Đối với BCE Loss bọc Sigmoid, giữ nguyên biên [-1, 1], KHÔNG CHIA cho temperature.
        # Điều này triệt tiêu hoàn toàn hiện tượng tràn đạo hàm Gradient Explosion sinh ra chữ `nan`.
        similarity = cosine_sim 

        # Đảo ngược nhãn logic: Real (0) -> Đích 1 (Kéo gần), Fake (1) -> Đích 0 (Đẩy lệch pha)
        target_similarity = 1.0 - labels.float()
        
        loss = F.binary_cross_entropy_with_logits(similarity, target_similarity)
        return loss