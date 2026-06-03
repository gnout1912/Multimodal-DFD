import os
import sys
import cv2
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import subprocess
from sklearn.model_selection import train_test_split
from tqdm import tqdm

FFMPEG_PATH = r"C:\Users\tuong\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import MultimodalConfig


def find_video_path(raw_root, sub_path, video_name):
    sub_path = str(sub_path).replace("\\", "/").strip("/")
    video_name = str(video_name).strip()

    sub_path_without_root = sub_path.replace("FakeAVCeleb/", "").strip("/")

    candidates = [
        os.path.join(raw_root, sub_path, video_name),
        os.path.join(raw_root, sub_path_without_root, video_name),
        os.path.join(raw_root, "FakeAVCeleb", sub_path_without_root, video_name),
        os.path.join(raw_root, video_name),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def extract_audio(video_path, output_audio_path, audio_sr, duration):
    """
    Extract audio bằng Gyan FFmpeg, đồng thời loại conda Library/bin khỏi PATH
    để tránh lỗi gdk_pixbuf DLL.
    """

    try:
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

        if not os.path.exists(FFMPEG_PATH):
            print(f"\n❌ Không tìm thấy FFMPEG_PATH: {FFMPEG_PATH}")
            return False

        ffmpeg_dir = os.path.dirname(FFMPEG_PATH)

        # Tạo env sạch hơn cho subprocess:
        # - Đưa Gyan FFmpeg bin lên đầu
        # - Loại conda Library/bin để tránh ffmpeg load nhầm DLL gdk_pixbuf
        env = os.environ.copy()

        old_path_parts = env.get("PATH", "").split(os.pathsep)
        clean_path_parts = []

        for p in old_path_parts:
            p_lower = p.lower()

            if "miniconda3" in p_lower and "library\\bin" in p_lower:
                continue

            if "anaconda3" in p_lower and "library\\bin" in p_lower:
                continue

            clean_path_parts.append(p)

        env["PATH"] = ffmpeg_dir + os.pathsep + os.pathsep.join(clean_path_parts)

        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(audio_sr),
            "-t", str(duration),
            "-loglevel", "error",
            output_audio_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        if result.returncode != 0:
            print(f"\n❌ FFmpeg error:\n{result.stderr}")
            return False

        if not os.path.exists(output_audio_path):
            return False

        if os.path.getsize(output_audio_path) < 1000:
            return False

        # Kiểm tra wav bằng soundfile, không dùng librosa để tránh gọi backend lỗi
        try:
            audio_data, sr = sf.read(output_audio_path)

            if audio_data is None or len(audio_data) == 0:
                return False

        except Exception as e:
            print(f"\n❌ Không đọc được wav sau khi extract: {e}")
            return False

        return True

    except Exception as e:
        print(f"\n❌ Audio exception: {e}")
        return False


def center_crop_frame(frame):
    """
    Fallback khi Haar Cascade không detect được mặt.
    Không tạo ảnh đen.
    Lấy vùng giữa frame, thường vẫn chứa khuôn mặt trong FakeAVCeleb.
    """

    h, w = frame.shape[:2]
    size = min(h, w)

    x1 = max((w - size) // 2, 0)
    y1 = max((h - size) // 2, 0)

    x2 = x1 + size
    y2 = y1 + size

    return frame[y1:y2, x1:x2]


def extract_face_frames(video_path, output_face_dir, image_size, max_frames):
    """
    Trích xuất frame mặt.
    Bản sửa:
    - Ưu tiên detect face bằng Haar.
    - Nếu detect fail, dùng last valid face.
    - Nếu chưa có last valid face, dùng center crop.
    - Không tạo blank frame.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return False

    os.makedirs(output_face_dir, exist_ok=True)

    for fname in os.listdir(output_face_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                os.remove(os.path.join(output_face_dir, fname))
            except Exception:
                pass

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    saved_count = 0
    detected_face_count = 0
    last_valid_face = None

    for f_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        success, frame = cap.read()

        if not success or frame is None:
            continue

        face_crop = None

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(50, 50)
            )

            if len(faces) > 0:
                faces = sorted(
                    faces,
                    key=lambda box: box[2] * box[3],
                    reverse=True
                )

                x, y, w, h = faces[0]

                H, W = frame.shape[:2]
                size = max(w, h)

                x_new = max(int(x - size * 0.18), 0)
                y_new = max(int(y - size * 0.22), 0)
                size_new = int(size * 1.40)

                x2 = min(x_new + size_new, W)
                y2 = min(y_new + size_new, H)

                if x2 > x_new and y2 > y_new:
                    face_crop = frame[y_new:y2, x_new:x2]
                    last_valid_face = face_crop
                    detected_face_count += 1

        except Exception:
            face_crop = None

        if face_crop is None and last_valid_face is not None:
            face_crop = last_valid_face

        if face_crop is None:
            face_crop = center_crop_frame(frame)

        try:
            face_crop = cv2.resize(face_crop, (image_size, image_size))
            save_path = os.path.join(output_face_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(save_path, face_crop)
            saved_count += 1

        except Exception:
            continue

    cap.release()

    # Bây giờ chỉ cần lưu được ít nhất 70% frame là nhận.
    # Vì đã có center crop fallback nên thường sẽ đủ max_frames.
    min_required = int(max_frames * 0.7)

    if saved_count < min_required:
        try:
            for fname in os.listdir(output_face_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    os.remove(os.path.join(output_face_dir, fname))
            os.rmdir(output_face_dir)
        except Exception:
            pass

        return False

    return True


def preprocess_single_video(video_raw_path, output_face_dir, output_audio_path, config):
    ok_video = extract_face_frames(
        video_path=video_raw_path,
        output_face_dir=output_face_dir,
        image_size=config.IMAGE_SIZE,
        max_frames=config.MAX_FRAMES
    )

    if not ok_video:
        return "fail_video"

    ok_audio = extract_audio(
        video_path=video_raw_path,
        output_audio_path=output_audio_path,
        audio_sr=config.AUDIO_SR,
        duration=config.AUDIO_DURATION
    )

    if not ok_audio:
        return "fail_audio"

    return "success"


def balance_real_fake(df, seed=42):
    real_df = df[df["label"] == 0].copy()
    fake_df = df[df["label"] == 1].copy()

    if len(real_df) == 0:
        raise ValueError("Không có sample real label=0 trong metadata.")

    if len(fake_df) == 0:
        raise ValueError("Không có sample fake label=1 trong metadata.")

    n_real = len(real_df)
    n_fake = min(len(fake_df), n_real)

    fake_df = fake_df.sample(n=n_fake, random_state=seed)

    balanced_df = pd.concat([real_df, fake_df], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print("\n📊 Sau khi cân bằng real/fake:")
    print(balanced_df["label"].value_counts())
    print(f"Tổng sample sau cân bằng: {len(balanced_df)}")

    return balanced_df


def split_train_dev_test(df, label_col="label", seed=42):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df[label_col]
    )

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df[label_col]
    )

    print("\n📦 Phân bố split:")
    print(f"Train: {len(train_df)}")
    print(train_df[label_col].value_counts())

    print(f"\nDev: {len(dev_df)}")
    print(dev_df[label_col].value_counts())

    print(f"\nTest: {len(test_df)}")
    print(test_df[label_col].value_counts())

    return train_df, dev_df, test_df


def process_split(split_name, split_df, processed_dir, config):
    print(f"\n🚀 Preprocess split: {split_name.upper()}")

    split_dir = os.path.join(processed_dir, "FakeAVCeleb", split_name)
    os.makedirs(split_dir, exist_ok=True)

    manifest_rows = []

    stats = {
        "success": 0,
        "fail_video": 0,
        "fail_audio": 0,
    }

    for idx, row in tqdm(
        split_df.iterrows(),
        total=len(split_df),
        desc=f"Processing {split_name}"
    ):
        video_raw_path = row["video_raw_path"]
        video_name = str(row["path"]).strip()

        base_name = os.path.splitext(os.path.basename(video_name))[0]

        identity_folder = str(row["Unnamed: 9"]).replace("\\", "/").strip("/")
        identity_name = identity_folder.split("/")[-1]

        safe_name = f"{int(row['label'])}_{idx}_{identity_name}_{base_name}"

        output_face_dir = os.path.join(split_dir, safe_name)
        output_audio_path = os.path.join(split_dir, f"{safe_name}.wav")

        status = preprocess_single_video(
            video_raw_path=video_raw_path,
            output_face_dir=output_face_dir,
            output_audio_path=output_audio_path,
            config=config
        )

        if status in stats:
            stats[status] += 1

        if status == "success":
            manifest_rows.append({
                "face_folder": f"FakeAVCeleb/{split_name}/{safe_name}",
                "audio_path": f"FakeAVCeleb/{split_name}/{safe_name}.wav",
                "label": int(row["label"]),
                "method": str(row.get("method", "")),
                "type": str(row.get("type", "")),
                "category": str(row.get("category", "")),
                "source_path": video_raw_path
            })

    print(f"\n📊 Thống kê preprocess {split_name}:")
    print(f"Success:    {stats['success']}")
    print(f"Fail video: {stats['fail_video']}")
    print(f"Fail audio: {stats['fail_audio']}")

    return manifest_rows


def main():
    print("=== PREPROCESS FAKEAVCELEB - BALANCED REAL/FAKE ===")

    config = MultimodalConfig()
    config.create_required_dirs()

    metadata_csv = config.FAKEAVCELEB_METADATA_CSV
    raw_data_dir = config.FAKEAVCELEB_RAW_DIR
    processed_dir = config.PROCESSED_DATA_DIR
    output_metadata_dir = config.METADATA_DIR

    if not os.path.exists(metadata_csv):
        print(f"❌ Không tìm thấy metadata CSV: {metadata_csv}")
        print("Hãy đặt meta_data.csv vào data/metadata và đổi tên thành fakeavceleb_meta_data.csv")
        return

    if not os.path.exists(raw_data_dir):
        print(f"❌ Không tìm thấy thư mục raw FakeAVCeleb: {raw_data_dir}")
        return

    df = pd.read_csv(metadata_csv)

    required_cols = ["method", "path", "Unnamed: 9"]

    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Metadata thiếu cột bắt buộc: {col}")
            return

    df["label"] = df["method"].apply(
        lambda x: 0 if str(x).strip().lower() == "real" else 1
    )

    print("\n📊 Phân bố label ban đầu trong metadata:")
    print(df["label"].value_counts())

    print("\n🔍 Đang kiểm tra video tồn tại trong raw dataset...")

    valid_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning videos"):
        video_path = find_video_path(
            raw_root=raw_data_dir,
            sub_path=row["Unnamed: 9"],
            video_name=row["path"]
        )

        if video_path is not None:
            row_dict = row.to_dict()
            row_dict["video_raw_path"] = video_path
            valid_records.append(row_dict)

    if len(valid_records) == 0:
        print("❌ Không tìm thấy video hợp lệ nào.")
        print("Hãy kiểm tra lại FAKEAVCELEB_RAW_DIR trong core/config.py")
        return

    df_valid = pd.DataFrame(valid_records).reset_index(drop=True)

    print(f"\n✅ Tìm thấy {len(df_valid)} video tồn tại trên ổ đĩa.")
    print("Phân bố label trước cân bằng:")
    print(df_valid["label"].value_counts())

    df_balanced = balance_real_fake(df_valid, seed=42)

    train_df, dev_df, test_df = split_train_dev_test(
        df_balanced,
        label_col="label",
        seed=42
    )

    splits = {
        "train": train_df,
        "dev": dev_df,
        "test": test_df,
    }

    all_manifest_data = {}

    for split_name, split_df in splits.items():
        rows = process_split(
            split_name=split_name,
            split_df=split_df,
            processed_dir=processed_dir,
            config=config
        )

        all_manifest_data[split_name] = rows

    os.makedirs(output_metadata_dir, exist_ok=True)

    for split_name in ["train", "dev", "test"]:
        manifest_path = os.path.join(
            output_metadata_dir,
            f"fakeavceleb_{split_name}_manifest.csv"
        )

        out_df = pd.DataFrame(all_manifest_data[split_name])
        out_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

        print(f"\n✅ Saved {split_name} manifest:")
        print(manifest_path)
        print(f"Samples sau preprocess: {len(out_df)}")

        if len(out_df) > 0 and "label" in out_df.columns:
            print(out_df["label"].value_counts())

    print("\n🎉 Hoàn thành preprocess FakeAVCeleb balanced.")
    print("\nBước tiếp theo:")
    print("python experiments/train_fakeavceleb.py")


if __name__ == "__main__":
    main()