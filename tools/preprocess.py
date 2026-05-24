import os
import cv2
import dlib
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from os.path import join

def get_video_info(video_path):
    """Lấy thông tin cơ bản của video để phục vụ đồng bộ audio-visual"""
    reader = cv2.VideoCapture(video_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    reader.release()
    return fps, num_frames

def get_boundingbox(face, width, height, scale=1.3):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    
    # Đảm bảo kích thước bounding box không vượt quá giới hạn khung hình
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    # ĐOẠN CỨU CÁNH: Nếu tính toán ra kích thước lỗi hoặc <= 0, trả về 0 hết để chặn lỗi unpack
    if size_bb <= 0:
        return 0, 0, 0

    return x1, y1, size_bb

def process_video_and_audio(video_path, output_face_dir, output_audio_dir, detector, max_frames=30):
    # TẬN DỤNG HÀM CỦA BẠN TẠI ĐÂY:
    fps, num_frames = get_video_info(video_path)
    
    if num_frames == 0 or fps == 0:
        return 0, None

    # Tính khoảng cách giữa các frame để lấy đều cho đủ max_frames (Tư duy LipFD)
    step = max(1, num_frames // max_frames)
    selected_indices = [i for i in range(num_frames) if i % step == 0][:max_frames]
    
    video_name = os.path.basename(video_path).split('.')[0]
    face_save_dir = join(output_face_dir, video_name)
    os.makedirs(face_save_dir, exist_ok=True)

    # Đọc lại video để tiến hành cắt khuôn mặt
    reader = cv2.VideoCapture(video_path)
    frame_num = 0
    saved_img_count = 0
    actual_selected_frames = []

    while reader.isOpened() and saved_img_count < max_frames:
        success, image = reader.read()
        if not success: break
        
        if frame_num in selected_indices:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)

            if len(faces):
                face = faces[0]
                x, y, size = get_boundingbox(face, width, height)
                cropped_face = image[y:y+size, x:x+size]
                
                if size > 0:
                    try:
                        cropped_face = cv2.resize(cropped_face, (300, 300)) # Kích thước cho EfficientNet-B3
                        cv2.imwrite(join(face_save_dir, f"frame_{saved_img_count:04d}.jpg"), cropped_face)
                        actual_selected_frames.append(frame_num)
                        saved_img_count += 1
                    except Exception:
                        pass
        frame_num += 1
    reader.release()

    if saved_img_count == 0:
        return 0, None

    # Trích xuất đoạn Audio khớp chính xác từng mili-giây với chuỗi ảnh mặt
    start_time = actual_selected_frames[0] / fps
    end_time = actual_selected_frames[-1] / fps
    audio_save_path = join(output_audio_dir, f"{video_name}.wav")
    
    try:
        # Load và hạ tần số lấy mẫu về 16kHz chuẩn cho Wav2Vec2
        y, sr = librosa.load(video_path, sr=16000, offset=start_time, duration=(end_time - start_time))
        sf.write(audio_save_path, y, sr)
        return saved_img_count, audio_save_path
    except Exception as e:
        print(f"Lỗi trích xuất audio tại {video_name}: {e}")
        return 0, None
    
if __name__ == '__main__':
    # Cấu hình các đường dẫn tuyệt đối chuẩn theo sơ đồ của bạn
    LAV_DF_ROOT = r'D:\LAV-DF'
    PROCESSED_ROOT = r'D:\Projects\Multimodal-DFD\data\processed'
    METADATA_DIR = r'D:\Projects\Multimodal-DFD\data\metadata'
    
    detector = dlib.get_frontal_face_detector()
    
    # Duyệt qua từng tập dữ liệu phân chia sẵn trong LAV-DF
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
            
            # Chạy trích xuất song song hình ảnh và âm thanh đồng bộ thời gian
            num_faces, audio_path = process_video_and_audio(
                video_path, output_face_dir, output_audio_dir, detector
            )
            
            # Nếu trích xuất thành công và có nhãn (Giả định nhãn dựa trên metadata hoặc quy ước tập dữ liệu)
            # Đối với LAV-DF, bạn có thể mặc định tạm thời label=1 (Fake) hoặc bổ sung logic đọc metadata.json
            # Để đơn giản hóa và loader chạy được luôn, ta tạm để label=1 (Sẽ tối ưu bằng file đọc json sau)
            if num_faces > 0 and audio_path is not None:
                split_records.append({
                    'video_id': video_id,
                    'label': 1,  
                    'face_folder': join(output_face_dir, video_id),
                    'audio_path': audio_path
                })
        
        # Lưu file chỉ mục CSV riêng cho từng tập để DataLoader nạp vào
        if split_records:
            df = pd.DataFrame(split_records)
            df.to_csv(join(METADATA_DIR, f'lavdf_{split}_manifest.csv'), index=False)
            print(f"-> Đã lưu xong cấu trúc manifest tập {split} vào file CSV!")