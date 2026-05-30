import os
import sys
import cv2
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

IMAGE_SIZE = 300
MAX_FRAMES = 30
AUDIO_SR = 16000

def preprocess_single_video(video_raw_path, output_face_dir, output_audio_path):
    cap = cv2.VideoCapture(video_raw_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False
        
    indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    os.makedirs(output_face_dir, exist_ok=True)
    frame_count = 0
    current_idx = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if not success:
            break
        if current_idx in indices:
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_crop = frame[y:y+h, x:x+w]
            else:
                face_crop = frame
                
            try:
                face_resized = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE))
                cv2.imwrite(os.path.join(output_face_dir, f"{frame_count:05d}.jpg"), face_resized)
            except Exception:
                pass
        current_idx += 1
    cap.release()
    
    try:
        y, sr = librosa.load(video_raw_path, sr=AUDIO_SR)
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        sf.write(output_audio_path, y, AUDIO_SR)
    except Exception:
        silence = np.zeros(AUDIO_SR * 3)
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        sf.write(output_audio_path, silence, AUDIO_SR)
        
    return True

def main():
    METADATA_CSV = r"D:\Projects\Multimodal-DFD\meta_data.csv"
    RAW_DATA_DIR = r"D:\Projects\FakeAVCeleb_Raw"  
    PROCESSED_DIR = r"D:\Projects\Multimodal-DFD\data\processed" 
    
    df = pd.read_csv(METADATA_CSV)
    # Lấy mẫu thử nghiệm 100 video (Subset) cho tập Test chéo thực chiến
    df_subset = df.sample(n=min(100, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"🎬 Bắt đầu tiền xử lý tự động cho {len(df_subset)} video thô từ FakeAVCeleb...")
    success_count = 0
    
    for idx, row in df_subset.iterrows():
        sub_path = str(row['Unnamed: 9']).replace('\\', '/')
        video_name = str(row['path']).strip()
        video_raw_path = os.path.join(RAW_DATA_DIR, sub_path, video_name)
        
        base_name = video_name.replace('.mp4', '').replace('.avi', '')
        output_face_dir = os.path.join(PROCESSED_DIR, sub_path, base_name)
        output_audio_path = os.path.join(PROCESSED_DIR, sub_path, f"{base_name}.wav")
        
        if os.path.exists(video_raw_path):
            if preprocess_single_video(video_raw_path, output_face_dir, output_audio_path):
                success_count += 1
                
    print(f"🎉 HOÀN THÀNH TIỀN XỬ LÝ! Đã dọn sạch và trích xuất thành công {success_count}/{len(df_subset)} video.")

if __name__ == "__main__":
    main()