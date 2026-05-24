import os
import torch
import pandas as pd
import numpy as np
import librosa
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MultimodalDataset(Dataset):
    def __init__(self, manifest_path, max_frames=30, target_audio_len=48000):
        """
        target_audio_len = 48000 tương ứng với 3 giây âm thanh (16000Hz * 3s)
        để cố định độ dài đầu vào cho Wav2Vec2.
        """
        self.df = pd.read_csv(manifest_path)
        self.max_frames = max_frames
        self.target_audio_len = target_audio_len
        
        # Định nghĩa transform chuẩn hóa ảnh cho EfficientNet-B3
        self.video_transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def _load_video_frames(self, face_folder):
        """Đọc chuỗi ảnh khuôn mặt và chuyển thành Tensor dạng: (C, T, H, W)"""
        frame_files = sorted([f for f in os.listdir(face_folder) if f.endswith('.jpg')])
        
        # Nếu số lượng ảnh nhiều hơn cấu hình, cắt bớt. Nếu thiếu, lặp lại ảnh cuối.
        if len(frame_files) >= self.max_frames:
            frame_files = frame_files[:self.max_frames]
        else:
            frame_files = frame_files + [frame_files[-1]] * (self.max_frames - len(frame_files))
            
        frames = []
        for f in frame_files:
            img_path = os.path.join(face_folder, f)
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.video_transform(img) # Kích thước: (3, 300, 300)
            frames.append(img_tensor)
            
        # Nối lại dọc theo trục thời gian (T) -> Kích thước: (T, 3, 300, 300)
        frames = torch.stack(frames, dim=0)
        # Hoán vị để đưa về chuẩn cấu trúc video (C, T, H, W) -> (3, 30, 300, 300)
        frames = frames.permute(1, 0, 2, 3)
        return frames

    def _load_audio(self, audio_path):
        """Đọc file sóng âm thanh thô và chuẩn hóa độ dài cố định"""
        # Đọc file wav với rate 16000Hz
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Padding (bù số 0) hoặc Truncate (cắt ngắn) để đưa về đúng độ dài target_audio_len
        if len(speech) >= self.target_audio_len:
            speech = speech[:self.target_audio_len]
        else:
            pad_len = self.target_audio_len - len(speech)
            speech = np.pad(speech, (0, pad_len), 'constant')
            
        return torch.FloatTensor(speech)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load nhãn dữ liệu (0: Real, 1: Fake)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        
        # 2. Load chuỗi ảnh mặt
        video_tensor = self._load_video_frames(row['face_folder'])
        
        # 3. Load sóng âm thanh
        audio_tensor = self._load_audio(row['audio_path'])
        
        return {
            'video': video_tensor,  # Kích thước: (3, 30, 300, 300)
            'audio': audio_tensor,  # Kích thước: (48000,)
            'label': label
        }

def get_multimodal_dataloader(manifest_path, batch_size=4, shuffle=True, num_workers=0):
    """Hàm bổ trợ khởi tạo DataLoader nhanh gọn"""
    dataset = MultimodalDataset(manifest_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        drop_last=True # Bỏ qua batch cuối nếu bị lẻ, giúp giữ cấu trúc tensor ổn định
    )
    return dataloader