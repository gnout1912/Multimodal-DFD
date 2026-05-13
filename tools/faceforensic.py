import os
import cv2
import dlib
import pandas as pd
from tqdm import tqdm
from os.path import join

def get_boundingbox(face, width, height, scale=1.3):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def extract_faces(video_path, output_dir, detector, max_frames=30):
    reader = cv2.VideoCapture(video_path)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tính khoảng cách giữa các frame để lấy đều cho đủ max_frames
    step = max(1, num_frames // max_frames)
    
    video_name = os.path.basename(video_path).split('.')[0]
    save_dir = join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    frame_num = 0
    saved_count = 0
    extracted_paths = []

    while reader.isOpened() and saved_count < max_frames:
        success, image = reader.read()
        if not success: break
        
        if frame_num % step == 0:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)

            if len(faces):
                face = faces[0]
                x, y, size = get_boundingbox(face, width, height)
                cropped_face = image[y:y+size, x:x+size]
                
                # Resize 300x300 cho EfficientNet-B3
                if size > 0:
                    cropped_face = cv2.resize(cropped_face, (300, 300))
                    file_name = f"frame_{saved_count:04d}.jpg"
                    save_path = join(save_dir, file_name)
                    cv2.imwrite(save_path, cropped_face)
                    extracted_paths.append(save_path)
                    saved_count += 1
        
        frame_num += 1
    reader.release()
    return extracted_paths

if __name__ == '__main__':
    RAW_DIR = r'D:\Projects\Multimodal-DFD\data\raw'
    PROCESSED_DIR = r'D:\Projects\Multimodal-DFD\data\processed\face_crops'
    METADATA_DIR = r'D:\Projects\Multimodal-DFD  \data\metadata'
    
    # Khởi tạo detector một lần duy nhất để tiết kiệm tài nguyên
    detector = dlib.get_frontal_face_detector()
    
    all_record = []

    for category in ['original', 'Deepfakes']:
        label = 0 if category == 'original' else 1
        input_folder = join(RAW_DIR, category)
        output_folder = join(PROCESSED_DIR, category)
        
        if not os.path.exists(input_folder): continue
        
        videos = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
        print(f"--- Processing: {category} ---")
        
        for video_name in tqdm(videos):
            v_path = join(input_folder, video_name)
            paths = extract_faces(v_path, output_folder, detector)
            
            # Lưu lại thông tin vào danh sách để làm file metadata mới
            if paths:
                all_record.append({
                    'video_id': video_name.split('.')[0],
                    'label': label,
                    'num_faces': len(paths)
                })

    # Xuất ra file CSV để sau này nạp vào mô hình dễ dàng
    df = pd.DataFrame(all_record)
    df.to_csv(join(METADATA_DIR, 'processed_faces_list.csv'), index=False)
    print("Xử lý xong! Metadata đã lưu tại data/metadata/processed_faces_list.csv")