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
    # FAKEAVCELEB LIPSYNC-FOCUSED V2
    # =========================================================================
    FAKEAVCELEB_RAW_DIR = r"D:\Projects\FakeAVCeleb_v1.2"
    FAKEAVCELEB_METADATA_CSV = os.path.join(METADATA_DIR, "fakeavceleb_meta_data.csv")

    TRAIN_MANIFEST = os.path.join(METADATA_DIR, "fakeavceleb_lipsync_train_manifest.csv")
    DEV_MANIFEST = os.path.join(METADATA_DIR, "fakeavceleb_lipsync_dev_manifest.csv")
    TEST_MANIFEST = os.path.join(METADATA_DIR, "fakeavceleb_lipsync_test_manifest.csv")

    FAKEAVCELEB_TRAIN_MANIFEST = TRAIN_MANIFEST
    FAKEAVCELEB_DEV_MANIFEST = DEV_MANIFEST
    FAKEAVCELEB_TEST_MANIFEST = TEST_MANIFEST

    # =========================================================================
    # INPUT CONFIG
    # =========================================================================
    IMAGE_SIZE = 224
    MAX_FRAMES = 16

    AUDIO_SR = 16000
    AUDIO_DURATION = 3
    TARGET_AUDIO_LEN = AUDIO_SR * AUDIO_DURATION

    # =========================================================================
    # TRAIN CONFIG - V2
    # =========================================================================
    BATCH_SIZE = 4
    EPOCHS = 25

    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 2e-4
    NUM_WORKERS = 0

    CONTRASTIVE_WEIGHT = 0.04
    GRAD_CLIP_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 6

    # =========================================================================
    # MODEL
    # =========================================================================
    WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"
    EFFICIENTNET_MODEL_NAME = "efficientnet_b3"

    DIM_VISUAL = 1536
    DIM_AUDIO = 768
    DIM_SHARED = 512

    # =========================================================================
    # OUTPUT FILES - KHÔNG ĐÈ V1
    # =========================================================================
    BEST_MODEL_NAME = "best_fakeavceleb_lipsync_v2_model.pth"
    BEST_THRESHOLD_NAME = "best_fakeavceleb_lipsync_v2_threshold.json"
    HISTORY_NAME = "fakeavceleb_lipsync_v2_training_history.csv"

    TEST_RESULT_NAME = "fakeavceleb_lipsync_v2_test_results.json"
    ERROR_ANALYSIS_NAME = "fakeavceleb_lipsync_v2_error_analysis.csv"

    @classmethod
    def create_required_dirs(cls):
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.METADATA_DIR, exist_ok=True)
        os.makedirs(cls.WEIGHTS_DIR, exist_ok=True)