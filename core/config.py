import os

class MultimodalConfig:
    # =========================================================================
    # 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG (CHỌN MÔI TRƯỜNG CHẠY)
    # =========================================================================
    
    # --- THẾ TRẬN 1: CẤU HÌNH CHẠY TRÊN LAPTOP CỦA BẠN (LOCAL WINDOWS) ---
    PROJECT_ROOT = r"D:\Projects\Multimodal-DFD"
    PROCESSED_DATA_DIR = r"D:\Projects\Multimodal-DFD\data\processed"
    
    # --- THẾ TRẬN 2: CẤU HÌNH CHẠY TRÊN GOOGLE COLAB (Nếu cần quay xe lên Cloud) ---
    # PROJECT_ROOT = "/content/multimodal_dfd_project/Multimodal-DFD"
    # PROCESSED_DATA_DIR = "/content/datasets_subset/processed"
    
    WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "core", "weights")
    METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
    
    TRAIN_MANIFEST = os.path.join(METADATA_DIR, "lavdf_train_manifest.csv")
    DEV_MANIFEST = os.path.join(METADATA_DIR, "lavdf_dev_manifest.csv")
    TEST_MANIFEST = os.path.join(METADATA_DIR, "lavdf_test_manifest.csv")
    
    # =========================================================================
    # 2. THAM SỐ TRÍCH XUẤT ĐẶC TRƯNG & MÔ HÌNH
    # =========================================================================
    IMAGE_SIZE = 300       # Kích thước ảnh chuẩn hóa đầu vào EfficientNet-B3
    MAX_FRAMES = 30        # Số lượng khung hình cố định cho mỗi chuỗi video
    
    AUDIO_SR = 16000       # Tần số lấy mẫu chuẩn cho mạng nơ-ron Wav2Vec2
    AUDIO_DURATION = 3     # Độ dài tín hiệu âm thanh tối đa (giây)
    TARGET_AUDIO_LEN = AUDIO_SR * AUDIO_DURATION  # Cố định chuỗi 48000 mẫu số
    
    # =========================================================================
    # 3. SIÊU THAM SỐ ĐÃ TỐI ƯU HÓA QUA OPTUNA (ĐẠT HIỆU SUẤT ĐỈNH CAO 86.00%)
    # =========================================================================
    # ĐÃ CHỈNH: Ép cứng Batch Size về 4 - Con số lập kỷ lục, an toàn tuyệt đối cho GPU 4GB
    BATCH_SIZE = 4          
    EPOCHS = 30             # Tăng lên 30 Epoch để lớp học đan chéo Cross-Attention hội tụ sâu
    
    # ĐÃ CHỈNH: Tốc độ học lý tưởng tìm ra từ thuật toán Bayes của Optuna
    LEARNING_RATE = 1.64e-5    
    # ĐÃ CHỈNH: Hệ số phạt chống Overfitting siêu mịn, bảo vệ các tầng unfreeze
    WEIGHT_DECAY = 2.81e-5     
    
    # ĐÃ CHỈNH: Giữ nguyên bằng 0 bắt buộc khi chạy trên Windows để tránh crash luồng ngầm
    NUM_WORKERS = 0        
    
    # =========================================================================
    # 4. KIẾN TRÚC MẠNG MÔ HÌNH (MODEL ARCHITECTURE)
    # =========================================================================
    WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"
    EFFICIENTNET_MODEL_NAME = "efficientnet_b3"
    
    DIM_VISUAL = 1536      # Số kênh đặc trưng đầu ra của EfficientNet-B3
    DIM_AUDIO = 768        # Kích thước trạng thái ẩn của Wav2Vec2-Base
    DIM_SHARED = 512       # Kích thước không gian chiếu chung cho hai miền dữ liệu

    @classmethod
    def create_required_dirs(cls):
        """Tự động kiểm tra và khởi tạo các thư mục hệ thống nếu chưa tồn tại"""
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.METADATA_DIR, exist_ok=True)
        os.makedirs(cls.WEIGHTS_DIR, exist_ok=True)

if __name__ == '__main__':
    MultimodalConfig.create_required_dirs()
    print("-> File core/config.py đã được cấu hình bộ siêu tham số tối ưu thành công!")