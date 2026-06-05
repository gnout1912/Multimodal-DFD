import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import soundfile as sf


class MultimodalMouthDataset(Dataset):
    def __init__(self, manifest_path, config, is_train=True):
        self.config = config
        self.is_train = is_train

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Không tìm thấy manifest: {manifest_path}")

        self.df = pd.read_csv(manifest_path)

        required_cols = ["face_folder", "mouth_folder", "audio_path", "label"]

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Manifest thiếu cột bắt buộc: {col}")

        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)

        if self.is_train:
            self.face_transform = transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=3),
                transforms.ColorJitter(
                    brightness=0.08,
                    contrast=0.08,
                    saturation=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            self.mouth_transform = transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=3),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        else:
            self.face_transform = transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            self.mouth_transform = transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        split_name = "Train" if is_train else "Eval"
        print(f"📊 {split_name} mouth samples: {len(self.df)} | Manifest: {manifest_path}")

    def __len__(self):
        return len(self.df)

    def _clean_manifest_path(self, path_str):
        path_str = str(path_str).replace("\\", "/")

        if "processed/" in path_str:
            path_str = path_str.split("processed/", 1)[1]

        if "data/processed/" in path_str:
            path_str = path_str.split("data/processed/", 1)[1]

        return path_str.strip("/")

    def _resolve_processed_path(self, path_str):
        pure_path = self._clean_manifest_path(path_str)

        if os.path.isabs(path_str) and os.path.exists(path_str):
            return path_str

        return os.path.join(self.config.PROCESSED_DATA_DIR, pure_path)

    def _load_frames(self, folder_path, transform):
        local_folder = self._resolve_processed_path(folder_path)

        if not os.path.exists(local_folder):
            return torch.zeros(
                self.config.MAX_FRAMES,
                3,
                self.config.IMAGE_SIZE,
                self.config.IMAGE_SIZE,
                dtype=torch.float32
            )

        frame_files = sorted([
            f for f in os.listdir(local_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(frame_files) == 0:
            return torch.zeros(
                self.config.MAX_FRAMES,
                3,
                self.config.IMAGE_SIZE,
                self.config.IMAGE_SIZE,
                dtype=torch.float32
            )

        if len(frame_files) > self.config.MAX_FRAMES:
            indices = np.linspace(
                0,
                len(frame_files) - 1,
                self.config.MAX_FRAMES,
                dtype=int
            )
            frame_files = [frame_files[i] for i in indices]

        elif len(frame_files) < self.config.MAX_FRAMES:
            frame_files = frame_files + [frame_files[-1]] * (
                self.config.MAX_FRAMES - len(frame_files)
            )

        frames = []

        for fname in frame_files:
            img_path = os.path.join(local_folder, fname)

            try:
                img = Image.open(img_path).convert("RGB")
                img = transform(img)
            except Exception:
                img = torch.zeros(
                    3,
                    self.config.IMAGE_SIZE,
                    self.config.IMAGE_SIZE,
                    dtype=torch.float32
                )

            frames.append(img)

        return torch.stack(frames, dim=0)

    def _load_audio(self, audio_path):
        local_audio = self._resolve_processed_path(audio_path)

        if not os.path.exists(local_audio):
            return torch.zeros(self.config.TARGET_AUDIO_LEN, dtype=torch.float32)

        try:
            speech, sr = sf.read(local_audio, dtype="float32")

            if len(speech.shape) > 1:
                speech = np.mean(speech, axis=1)

            if len(speech) == 0:
                return torch.zeros(self.config.TARGET_AUDIO_LEN, dtype=torch.float32)

            if len(speech) >= self.config.TARGET_AUDIO_LEN:
                speech = speech[:self.config.TARGET_AUDIO_LEN]
            else:
                speech = np.pad(
                    speech,
                    (0, self.config.TARGET_AUDIO_LEN - len(speech)),
                    mode="constant"
                )

            return torch.tensor(speech, dtype=torch.float32)

        except Exception as e:
            print(f"⚠️ Không đọc được audio wav: {local_audio} | Error: {e}")
            return torch.zeros(self.config.TARGET_AUDIO_LEN, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        face_video = self._load_frames(row["face_folder"], self.face_transform)
        mouth_video = self._load_frames(row["mouth_folder"], self.mouth_transform)
        audio = self._load_audio(row["audio_path"])
        label = torch.tensor(int(row["label"]), dtype=torch.float32)

        return face_video, mouth_video, audio, label


def get_mouth_loaders(config):
    train_dataset = MultimodalMouthDataset(
        config.TRAIN_MANIFEST,
        config,
        is_train=True
    )

    dev_dataset = MultimodalMouthDataset(
        config.DEV_MANIFEST,
        config,
        is_train=False
    )

    test_dataset = MultimodalMouthDataset(
        config.TEST_MANIFEST,
        config,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, dev_loader, test_loader