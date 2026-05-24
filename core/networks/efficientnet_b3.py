import torch.nn as nn
import timm

class VisualEncoder(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(VisualEncoder, self).__init__()
        # Load EfficientNet-B3 từ thư viện timm
        self.model = timm.create_model('efficientnet_b3', pretrained=pretrained)
        
        # Lấy số lượng feature đầu vào của lớp classifier gốc
        num_ftrs = self.model.classifier.in_features
        
        # Thay thế lớp classifier để phân loại 2 lớp (Real/Fake)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)