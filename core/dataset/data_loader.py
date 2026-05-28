import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import librosa
from core.config import MultimodalConfig

class MultimodalDataset(Dataset):
    def __init__(self, manifest_path, config, is_train=True):
        self.config = config
        self.is_train = is_train
        
        # ==================== THÊM TRANSFORM ====================
        self.video_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        # ======================================================
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Không tìm thấy file manifest tại: {manifest_path}")
            
        full_df = pd.read_csv(manifest_path)
        
        # Lấy subset để train nhanh trên Colab
        if self.is_train:
            self.df = full_df.sample(n=min(400, len(full_df)), random_state=42).reset_index(drop=True)
        else:
            self.df = full_df.sample(n=min(100, len(full_df)), random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _clean_manifest_path(self, path_str):
        path_str = str(path_str).replace('\\', '/')
        if "processed/" in path_str:
            path_str = path_str.split("processed/", 1)[1]
        return path_str.strip('/')

    def _load_video_frames(self, face_folder):
        pure_folder = self._clean_manifest_path(face_folder)
        local_folder = os.path.join(self.config.PROCESSED_DATA_DIR, pure_folder)
        
        if not os.path.exists(local_folder) or not os.listdir(local_folder):
            return torch.zeros((self.config.MAX_FRAMES, 3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            
        frame_files = sorted([f for f in os.listdir(local_folder) if f.endswith(('.jpg', '.png'))])
        if len(frame_files) == 0:
            return torch.zeros((self.config.MAX_FRAMES, 3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            
        if len(frame_files) >= self.config.MAX_FRAMES:
            frame_files = frame_files[:self.config.MAX_FRAMES]
        else:
            frame_files = frame_files + [frame_files[-1]] * (self.config.MAX_FRAMES - len(frame_files))
            
        frames = []
        for f in frame_files:
            try:
                img = Image.open(os.path.join(local_folder, f)).convert('RGB')
                frames.append(self.video_transform(img))
            except Exception:
                frames.append(torch.zeros((3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)))
                
        return torch.stack(frames)

    def _load_audio(self, audio_path):
        pure_audio = self._clean_manifest_path(audio_path)
        local_audio = os.path.join(self.config.PROCESSED_DATA_DIR, pure_audio)
        
        if not os.path.exists(local_audio):
            return torch.zeros(self.config.TARGET_AUDIO_LEN)
            
        try:
            speech, sr = librosa.load(local_audio, sr=self.config.AUDIO_SR)
            if len(speech) >= self.config.TARGET_AUDIO_LEN:
                speech = speech[:self.config.TARGET_AUDIO_LEN]
            else:
                speech = np.pad(speech, (0, self.config.TARGET_AUDIO_LEN - len(speech)), 'constant')
            return torch.tensor(speech, dtype=torch.float32)
        except Exception:
            return torch.zeros(self.config.TARGET_AUDIO_LEN)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_tensor = self._load_video_frames(row['face_folder'])
        audio_tensor = self._load_audio(row['audio_path'])
        label = torch.tensor(int(row['label']), dtype=torch.float32)
        
        return video_tensor, audio_tensor, label


def get_multimodal_loaders(config):
    """Trả về cả 3 loader: train, dev, test"""
    train_dataset = MultimodalDataset(config.TRAIN_MANIFEST, config, is_train=True)
    dev_dataset = MultimodalDataset(config.DEV_MANIFEST, config, is_train=False)
    test_dataset = MultimodalDataset(config.TEST_MANIFEST, config, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                             num_workers=config.NUM_WORKERS, pin_memory=True)
    
    return train_loader, dev_loader, test_loader