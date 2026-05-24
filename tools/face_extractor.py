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
    
    # Nếu thư mục đã tồn tại và đủ ảnh, bỏ qua để tiết kiệm thời gian
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= max_frames:
        reader.release()
        return [join(save_dir, f) for f in os.listdir(save_dir)]

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
                    try:
                        cropped_face = cv2.resize(cropped_face, (300, 300))
                        file_name = f"frame_{saved_count:04d}.jpg"
                        save_path = join(save_dir, file_name)
                        cv2.imwrite(save_path, cropped_face)
                        extracted_paths.append(save_path)
                        saved_count += 1
                    except Exception as e:
                        print(f"Lỗi resize tại video {video_name}: {e}")
        
        frame_num += 1
    reader.release()
    return extracted_paths

if __name__ == '__main__':
    RAW_DIR = r'D:\Projects\Multimodal-DFD\data\raw'
    PROCESSED_DIR = r'D:\Projects\Multimodal-DFD\data\processed\face_crops'
    METADATA_DIR = r'D:\Projects\Multimodal-DFD\data\metadata'
    CSV_PATH = join(METADATA_DIR, 'processed_faces_list.csv')
    
    detector = dlib.get_frontal_face_detector()
    
    # Đọc lại file CSV cũ nếu có để chạy tiếp (Resume)
    if os.path.exists(CSV_PATH):
        all_record = pd.read_csv(CSV_PATH).to_dict('records')
        processed_ids = set([str(r['video_id']) for r in all_record])
    else:
        all_record = []
        processed_ids = set()

    for category in ['original', 'Deepfakes']:
        label = 0 if category == 'original' else 1
        input_folder = join(RAW_DIR, category)
        output_folder = join(PROCESSED_DIR, category)
        
        if not os.path.exists(input_folder): continue
        
        videos = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
        print(f"\n--- Processing: {category} ---")
        
        for video_name in tqdm(videos):
            video_id = video_name.split('.')[0]
            
            # Kiểm tra nếu đã xử lý rồi thì bỏ qua
            if video_id in processed_ids:
                continue
                
            v_path = join(input_folder, video_name)
            paths = extract_faces(v_path, output_folder, detector)
            
            if paths:
                all_record.append({
                    'video_id': video_id,
                    'label': label,
                    'num_faces': len(paths)
                })
                
                # Lưu CSV liên tục sau mỗi video để tránh mất dữ liệu khi crash
                df = pd.DataFrame(all_record)
                df.to_csv(CSV_PATH, index=False)

    print(f"\n Hoàn thành! Metadata lưu tại: {CSV_PATH}")