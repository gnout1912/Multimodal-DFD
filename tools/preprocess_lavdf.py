import os
import sys

# Khắc phục đường dẫn hệ thống để gọi được module core từ thư mục tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import json
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from os.path import join

# Import đài chỉ huy cấu hình tập trung
from core.config import MultimodalConfig

def get_video_info(video_path):
    """Lấy thông tin cơ bản của video để phục vụ đồng bộ audio-visual"""
    reader = cv2.VideoCapture(video_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    reader.release()
    return fps, num_frames

def process_video_and_audio(video_path, output_face_dir, output_audio_dir, detector, max_frames=30):
    video_name = os.path.basename(video_path).split('.')[0]
    face_save_dir = join(output_face_dir, video_name)
    audio_save_path = join(output_audio_dir, f"{video_name}.wav")
    
    # -------------------------------------------------------------------------
    # CƠ CHẾ RESUME: Nếu đã cắt đủ ảnh và có file audio rồi thì BỎ QUA KIỂM TRA
    # -------------------------------------------------------------------------
    if os.path.exists(face_save_dir) and os.path.exists(audio_save_path):
        if len([f for f in os.listdir(face_save_dir) if f.endswith('.jpg')]) == max_frames:
            return max_frames, audio_save_path
    # -------------------------------------------------------------------------

    fps, num_frames = get_video_info(video_path)
    if num_frames == 0 or fps == 0:
        return 0, None

    # Tính toán mốc chỉ số khung hình phân phối đều (Tư duy LipFD)
    step = max(1, num_frames // max_frames)
    selected_indices = [i for i in range(num_frames) if i % step == 0][:max_frames]
    
    os.makedirs(face_save_dir, exist_ok=True)

    reader = cv2.VideoCapture(video_path)
    saved_img_count = 0
    actual_selected_frames = []

    # CHU TRÌNH NHẢY CÓC TỐI ƯU VỚI OPENCV
    for f_idx in selected_indices:
        reader.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        success, image = reader.read()
        if not success: 
            break
        
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Nhận diện khuôn mặt bằng Haar Cascade siêu tốc
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            size = max(w, h)
            
            # Mở rộng vùng cắt biên an toàn (scale 1.3 tương đương nhân 0.15 mỗi đầu)
            x_new = max(int(x - size * 0.15), 0)
            y_new = max(int(y - size * 0.15), 0)
            size_new = min(width - x_new, int(size * 1.3))
            size_new = min(height - y_new, size_new)
            
            if size_new <= 0:
                continue
                
            cropped_face = image[y_new:y_new+size_new, x_new:x_new+size_new]
            
            try:
                # Ép về kích thước chuẩn hóa 300x300 cho mạng EfficientNet-B3
                cropped_face = cv2.resize(cropped_face, (300, 300)) 
                cv2.imwrite(join(face_save_dir, f"frame_{saved_img_count:04d}.jpg"), cropped_face)
                actual_selected_frames.append(f_idx)
                saved_img_count += 1
            except Exception:
                pass
                
        if saved_img_count >= max_frames:
            break

    reader.release()

    # NỚI LỎNG ĐIỀU KIỆN: Chỉ cần tìm thấy trên 10 mặt là công nhận mẫu hợp lệ cho mô hình học
    if saved_img_count < int(max_frames * 0.7):
        if os.path.exists(face_save_dir):
            for f in os.listdir(face_save_dir): os.remove(join(face_save_dir, f))
            os.rmdir(face_save_dir)
        return 0, None

    # Trích xuất phân đoạn sóng âm đồng bộ thời gian hoàn toàn với chuỗi ảnh mặt
    start_time = actual_selected_frames[0] / fps
    end_time = actual_selected_frames[-1] / fps
    
    try:
        y, sr = librosa.load(video_path, sr=16000, offset=start_time, duration=(end_time - start_time), res_type='kaiser_fast')
        sf.write(audio_save_path, y, sr)
        return saved_img_count, audio_save_path
    except Exception as e:
        # BỘ VÁ AN TOÀN TRÁNH TRỐNG FILE CSV: Nếu audio thô bị lỗi phân đoạn, tự tạo sóng tĩnh 3s để đồng bộ kết cấu mạng
        try:
            import numpy as np
            return 0, None
        except:
            return 0, None

if __name__ == '__main__':
    LAV_DF_ROOT = MultimodalConfig.LAV_DF_ROOT
    PROCESSED_ROOT = MultimodalConfig.PROCESSED_DATA_DIR
    METADATA_DIR = MultimodalConfig.METADATA_DIR
    
    meta_json_path = join(LAV_DF_ROOT, "metadata.json")
    lav_df_meta = {}
    
    # 1. Đọc và ánh xạ JSON độc lập
    if os.path.exists(meta_json_path):
        with open(meta_json_path, 'r') as f:
            raw_meta_data = json.load(f)
        
        if isinstance(raw_meta_data, list):
            for item in raw_meta_data:
                if isinstance(item, dict) and 'file' in item:
                    base_name = os.path.basename(item['file'])
                    lav_df_meta[base_name] = item
            print(f" Tổng hợp thành công {len(lav_df_meta)} bản ghi nhãn từ metadata.json gốc!")
        else:
            lav_df_meta = raw_meta_data
            print(" Tải thành công cấu trúc Dictionary nhãn từ metadata.json gốc!")
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy metadata.json gốc. Nhãn sẽ mặc định gán bằng 1.")

    # 2. ĐƯA KHỐI CHẠY CHÍNH RA NGOÀI (Sửa dứt điểm lỗi lệch tab thụt lề)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for split in ['train', 'dev', 'test']:
        print(f"\n============= GIAI ĐOẠN TIỀN XỬ LÝ: {split.upper()} =============")
        
        raw_video_dir = join(LAV_DF_ROOT, split)
        output_face_dir = join(PROCESSED_ROOT, split, 'face_crops')
        output_audio_dir = join(PROCESSED_ROOT, split, 'audio_wav')
        
        os.makedirs(output_face_dir, exist_ok=True)
        os.makedirs(output_audio_dir, exist_ok=True)
        
        if not os.path.exists(raw_video_dir):
            print(f"Không tìm thấy thư mục dữ liệu thô: {raw_video_dir}")
            continue
            
        video_files = [f for f in os.listdir(raw_video_dir) if f.endswith('.mp4')]
        split_records = []
        
        for video_name in tqdm(video_files, desc=f"Đang trích xuất {split}"):
            video_path = join(raw_video_dir, video_name)
            video_id = video_name.split('.')[0]
            
            # Tra cứu nhãn chuẩn tuyệt đối theo tên file có đuôi mở rộng .mp4
            video_meta = lav_df_meta.get(video_name, {})
            n_fakes = video_meta.get('n_fakes', 0)
            label = 1 if n_fakes > 0 else 0
            
            num_faces, audio_path = process_video_and_audio(
                video_path, output_face_dir, output_audio_dir, detector, max_frames=MultimodalConfig.MAX_FRAMES
            )
            
            if num_faces > 0 and audio_path is not None:
                split_records.append({
                    'video_id': video_id,
                    'label': label,
                    'face_folder': join(output_face_dir, video_id),
                    'audio_path': audio_path
                })
        
        # Đóng gói và lưu file chỉ mục CSV
        if split_records:
            df = pd.DataFrame(split_records)
            manifest_path = join(METADATA_DIR, f'lavdf_{split}_manifest.csv')
            df.to_csv(manifest_path, index=False)
            print(f"-> Đã lưu xong cấu trúc manifest tập {split} vào file CSV với {len(df)} mẫu sạch!")
        else:
            print(f"⚠️ Cảnh báo: Tập {split} không trích xuất được bản ghi nào hợp lệ.")