import os

class MultimodalConfig:
    # =========================================================================
    # 1. ĐƯỜNG DẪN HỆ THỐNG
    # =========================================================================
    # PROJECT_ROOT = "/content/multimodal_dfd_project/Multimodal-DFD" tren Colab
    PROJECT_ROOT = r"D:\Projects\Multimodal-DFD"
    
    # PROCESSED_DATA_DIR = "/content/datasets_subset/processed" tren Colab
    PROCESSED_DATA_DIR = r"D:\Projects\Multimodal-DFD\data\processed"
    
    WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "core", "weights")
    METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
    
    TRAIN_MANIFEST = os.path.join(METADATA_DIR, "lavdf_train_manifest.csv")
    DEV_MANIFEST = os.path.join(METADATA_DIR, "lavdf_dev_manifest.csv")
    TEST_MANIFEST = os.path.join(METADATA_DIR, "lavdf_test_manifest.csv")
    
    # =========================================================================
    # 2. THAM SỐ TRÍCH XUẤT ĐẶC TRƯNG & MÔ HÌNH
    # =========================================================================
    IMAGE_SIZE = 300       
    MAX_FRAMES = 30        
    
    AUDIO_SR = 16000       
    AUDIO_DURATION = 3     
    TARGET_AUDIO_LEN = AUDIO_SR * AUDIO_DURATION  # 48000
    
    # =========================================================================
    # 3. SIÊU THAM SỐ HUẤN LUYỆN - ĐÃ TỐI ƯU
    # =========================================================================
    BATCH_SIZE = 4          # Giảm từ 8 → tránh OOM + ổn định hơn
    EPOCHS = 20             # Tăng để model có thời gian học
    
    LEARNING_RATE = 5e-5    # Tăng nhẹ để backbone unfreeze học tốt hơn
    WEIGHT_DECAY = 5e-4     # Giảm mạnh so với 1e-2 (trước quá cao)
    
    NUM_WORKERS = 4        
    
    WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"
    EFFICIENTNET_MODEL_NAME = "efficientnet_b3"
    
    DIM_VISUAL = 1536      
    DIM_AUDIO = 768        
    DIM_SHARED = 512       

    @classmethod
    def create_required_dirs(cls):
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.METADATA_DIR, exist_ok=True)
        os.makedirs(cls.WEIGHTS_DIR, exist_ok=True)