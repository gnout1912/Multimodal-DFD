import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", pretrained=True):
        super(AudioEncoder, self).__init__()
        
        if pretrained:
            # Tải cấu hình và trọng số đã được huấn luyện sẵn của Meta
            self.config = Wav2Vec2Config.from_pretrained(model_name)
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        else:
            self.config = Wav2Vec2Config(model_name)
            self.wav2vec2 = Wav2Vec2Model(self.config)
            
        # Đóng băng các lớp Feature Extractor đầu tiên (CNN) của Wav2Vec2
        # Việc này giúp tiết kiệm VRAM cho RTX 3050 và tránh làm nhiễu bộ trích xuất thô
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        # Thừa hưởng số chiều đầu ra của Wav2Vec2 Base (thường là 768)
        self.output_dim = self.config.hidden_size 

    def forward(self, x):
        """
        Đầu vào x: Tensor âm thanh có kích thước (Batch_Size, 48000)
        Đầu ra: Tensor đặc trưng có kích thước (Batch_Size, Time_Steps, 768)
        """
        # Wav2Vec2 yêu cầu đầu vào dạng float32
        outputs = self.wav2vec2(x)
        
        # Lấy giá trị hidden states ở lớp cuối cùng (Last Hidden State)
        # Kích thước thu được sẽ là: (Batch_Size, Sequence_Length, 768)
        audio_features = outputs.last_hidden_state
        
        return audio_features

if __name__ == '__main__':
    # Đoạn code chạy thử nhanh để kiểm tra chiều dữ liệu (Sanity Check)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy thử nghiệm AudioEncoder trên: {device}")
    
    # Giả lập 1 Batch gồm 2 mẫu âm thanh, mỗi mẫu dài 3 giây (48000 mẫu số)
    dummy_audio = torch.randn(2, 48000).to(device)
    
    model = AudioEncoder().to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(dummy_audio)
        
    print(f"Kích thước đầu vào giả lập: {dummy_audio.shape}")
    print(f"Kích thước đặc trưng đầu ra: {output.shape}")
    print("-> Nhánh Audio hoạt động chính xác!")