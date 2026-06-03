import os


class MultimodalConfig:
    # =========================================================================
    # PROJECT PATH
    # =========================================================================
    PROJECT_ROOT = r"D:\Projects\Multimodal-DFD"
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

    WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "core", "weights")
    METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")

    # =========================================================================
    # FAKEAVCELEB ONLY - GIẢI PHÁP 1
    # =========================================================================
    FAKEAVCELEB_RAW_DIR = r"D:\Projects\FakeAVCeleb_v1.2"
    FAKEAVCELEB_METADATA_CSV = os.path.join(METADATA_DIR, "fakeavceleb_meta_data.csv")

    TRAIN_MANIFEST = os.path.join(METADATA_DIR, "fakeavceleb_train_manifest.csv")
    DEV_MANIFEST = os.path.join(METADATA_DIR, "fakeavceleb_dev_manifest.csv")
    TEST_MANIFEST = os.path.join(METADATA_DIR, "fakeavceleb_test_manifest.csv")

    FAKEAVCELEB_TRAIN_MANIFEST = TRAIN_MANIFEST
    FAKEAVCELEB_DEV_MANIFEST = DEV_MANIFEST
    FAKEAVCELEB_TEST_MANIFEST = TEST_MANIFEST

    # =========================================================================
    # LAPTOP RTX 3050 4GB - SAFE CONFIG
    # =========================================================================
    IMAGE_SIZE = 224
    MAX_FRAMES = 16

    AUDIO_SR = 16000
    AUDIO_DURATION = 3
    TARGET_AUDIO_LEN = AUDIO_SR * AUDIO_DURATION

    # BATCH_SIZE là batch logic, train.py sẽ chạy physical batch = 1 rồi accumulation
    BATCH_SIZE = 3
    EPOCHS = 25

    LEARNING_RATE = 4.4636e-05
    WEIGHT_DECAY = 1.7524e-04
    NUM_WORKERS = 0

    # =========================================================================
    # MODEL
    # =========================================================================
    WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"
    EFFICIENTNET_MODEL_NAME = "efficientnet_b3"

    DIM_VISUAL = 1536
    DIM_AUDIO = 768
    DIM_SHARED = 512

    # =========================================================================
    # TRAINING CONTROL
    # =========================================================================
    CONTRASTIVE_WEIGHT = 0.0719
    GRAD_CLIP_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 6

    BEST_MODEL_NAME = "best_fakeavceleb_model.pth"
    BEST_THRESHOLD_NAME = "best_fakeavceleb_threshold.json"
    HISTORY_NAME = "fakeavceleb_training_history.csv"

    @classmethod
    def create_required_dirs(cls):
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.METADATA_DIR, exist_ok=True)
        os.makedirs(cls.WEIGHTS_DIR, exist_ok=True)