import os
import sys
import re
import gc
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Cho phép import module từ thư mục gốc project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import MultimodalConfig
from core.dataset.data_loader import MultimodalDataset
from core.models.multimodal_model import MultimodalDeepfakeDetector


# ============================================================
# 0. FIX WINDOWS / CONDA FFMPEG DLL CONFLICT
# ============================================================
def clean_conda_ffmpeg_path():
    """
    Loại conda Library/bin khỏi PATH để tránh lỗi gdk_pixbuf/ffmpeg DLL.
    Dù train hiện đọc wav bằng soundfile, đoạn này vẫn giúp tránh thư viện phụ gọi nhầm ffmpeg conda.
    """
    old_path = os.environ.get("PATH", "")
    path_parts = old_path.split(os.pathsep)

    clean_parts = []

    for p in path_parts:
        p_lower = p.lower()

        if "miniconda3" in p_lower and "library\\bin" in p_lower:
            continue

        if "anaconda3" in p_lower and "library\\bin" in p_lower:
            continue

        clean_parts.append(p)

    os.environ["PATH"] = os.pathsep.join(clean_parts)


clean_conda_ffmpeg_path()


# ============================================================
# 1. CẤU HÌNH TUNING
# ============================================================
N_TRIALS = 15
TUNING_EPOCHS = 4

# Dùng subset để tuning nhanh hơn.
# Train: 200 real + 200 fake = 400 mẫu
# Dev:   60 real + 60 fake = 120 mẫu
TUNING_TRAIN_PER_CLASS = 200
TUNING_DEV_PER_CLASS = 60

# Manifest lipsync-focused, không phụ thuộc config hiện đang trỏ qua dataset nào.
LIPSYNC_TRAIN_MANIFEST_NAME = "fakeavceleb_lipsync_train_manifest.csv"
LIPSYNC_DEV_MANIFEST_NAME = "fakeavceleb_lipsync_dev_manifest.csv"
LIPSYNC_TEST_MANIFEST_NAME = "fakeavceleb_lipsync_test_manifest.csv"


# ============================================================
# 2. HÀM LẤY MẪU CÂN BẰNG
# ============================================================
def balanced_sample(df, n_per_class, seed=42):
    if "label" not in df.columns:
        raise ValueError("Manifest thiếu cột 'label'.")

    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]

    real_n = min(n_per_class, len(real_df))
    fake_n = min(n_per_class, len(fake_df))

    if real_n == 0 or fake_n == 0:
        raise ValueError(
            f"Dataset thiếu class. Real={len(real_df)}, Fake={len(fake_df)}"
        )

    sampled_df = pd.concat([
        real_df.sample(real_n, random_state=seed),
        fake_df.sample(fake_n, random_state=seed)
    ])

    sampled_df = sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return sampled_df


# ============================================================
# 3. LOSS AUDIO-VISUAL CONTRASTIVE
# ============================================================
def av_contrastive_loss(v_embeds, a_embeds, labels, temperature=0.2):
    """
    Real label = 0  -> audio/video nên gần nhau
    Fake label = 1  -> audio/video nên lệch nhau
    """
    v_norm = torch.nn.functional.normalize(v_embeds, dim=-1, eps=1e-8)
    a_norm = torch.nn.functional.normalize(a_embeds, dim=-1, eps=1e-8)

    sim = torch.sum(v_norm * a_norm, dim=-1)
    sim = torch.clamp(sim, min=-0.99, max=0.99)

    target = 1.0 - labels.float()
    logits = sim / temperature

    return nn.functional.binary_cross_entropy_with_logits(logits, target)


# ============================================================
# 4. TÌM THRESHOLD TỐT NHẤT
# ============================================================
def find_best_threshold(labels, probs):
    """
    Chọn threshold theo score cân bằng:
    score = 0.5 * Balanced Accuracy + 0.5 * Macro F1

    Cách này tránh trường hợp model nghiêng hẳn về Fake hoặc Real.
    """
    labels = np.array(labels).astype(int)
    probs = np.array(probs)

    best_t = 0.5
    best_score = -1.0
    best_acc = 0.0
    best_f1 = 0.0
    best_macro_f1 = 0.0
    best_bal_acc = 0.0

    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)

        acc = accuracy_score(labels, preds)
        binary_f1 = f1_score(labels, preds, zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(labels, preds)

        score = 0.5 * bal_acc + 0.5 * macro_f1

        if score > best_score:
            best_score = score
            best_t = float(t)
            best_acc = acc
            best_f1 = binary_f1
            best_macro_f1 = macro_f1
            best_bal_acc = bal_acc

    return {
        "threshold": best_t,
        "score": best_score,
        "accuracy": best_acc,
        "f1": best_f1,
        "macro_f1": best_macro_f1,
        "balanced_acc": best_bal_acc
    }


# ============================================================
# 5. TRAIN 1 EPOCH CHO OPTUNA
# ============================================================
def train_epoch_tuning(
    model,
    loader,
    bce_criterion,
    optimizer,
    device,
    accumulation_steps,
    contrastive_weight,
    grad_clip_norm,
    trial_num,
    epoch
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total = 0

    progress_bar = tqdm(loader, desc=f"Trial {trial_num} | Epoch {epoch}")

    for batch_idx, (video, audio, labels) in enumerate(progress_bar):
        video = video.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, v_emb, a_emb = model(video, audio)

            logits = torch.nan_to_num(
                logits,
                nan=0.0,
                posinf=10.0,
                neginf=-10.0
            )
            logits = torch.clamp(logits, min=-10.0, max=10.0)

            loss_bce = bce_criterion(logits, labels)
            loss_con = av_contrastive_loss(v_emb, a_emb, labels)

            loss = loss_bce + contrastive_weight * loss_con
            scaled_loss = loss / accumulation_steps

        if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaled_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, total)

    return avg_loss


# ============================================================
# 6. VALIDATE CHO OPTUNA
# ============================================================
def validate_tuning(model, loader, bce_criterion, device, contrastive_weight):
    model.eval()

    total_loss = 0.0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validating")

        for video, audio, labels in progress_bar:
            video = video.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, v_emb, a_emb = model(video, audio)

                logits = torch.nan_to_num(
                    logits,
                    nan=0.0,
                    posinf=10.0,
                    neginf=-10.0
                )
                logits = torch.clamp(logits, min=-10.0, max=10.0)

                loss_bce = bce_criterion(logits, labels)
                loss_con = av_contrastive_loss(v_emb, a_emb, labels)

                loss = loss_bce + contrastive_weight * loss_con

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            label_np = labels.detach().cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(label_np)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

    avg_loss = total_loss / max(1, total)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)

    threshold_info = find_best_threshold(all_labels, all_probs)

    preds = (all_probs >= threshold_info["threshold"]).astype(int)

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return {
        "loss": avg_loss,
        "acc": acc,
        "auc": auc,
        "f1": f1,
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "threshold": threshold_info["threshold"],
        "score": threshold_info["score"]
    }


# ============================================================
# 7. OBJECTIVE CHO OPTUNA
# ============================================================
def objective(trial):
    torch.cuda.empty_cache()
    gc.collect()

    config = MultimodalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Các tham số cần tuning
    max_frames = trial.suggest_categorical("max_frames", [16, 20, 24])
    target_batch = trial.suggest_categorical("batch_size", [2, 3, 4, 6])
    lr = trial.suggest_float("learning_rate", 8e-6, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 5e-5, 5e-4, log=True)
    contrastive_weight = trial.suggest_float("contrastive_weight", 0.02, 0.08)

    # Apply cấu hình tạm cho trial
    config.IMAGE_SIZE = 224
    config.MAX_FRAMES = max_frames

    physical_batch = 1
    accumulation_steps = max(1, target_batch // physical_batch)

    temp_train_path = os.path.join(
        config.METADATA_DIR,
        "temp_lipsync_tuning_train.csv"
    )

    temp_dev_path = os.path.join(
        config.METADATA_DIR,
        "temp_lipsync_tuning_dev.csv"
    )

    train_dataset = MultimodalDataset(
        temp_train_path,
        config,
        is_train=True
    )

    dev_dataset = MultimodalDataset(
        temp_dev_path,
        config,
        is_train=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=physical_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=physical_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = MultimodalDeepfakeDetector().to(device)

    bce_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )

    best_trial_score = -1.0
    best_trial_auc = 0.0

    for epoch in range(1, TUNING_EPOCHS + 1):
        train_loss = train_epoch_tuning(
            model=model,
            loader=train_loader,
            bce_criterion=bce_criterion,
            optimizer=optimizer,
            device=device,
            accumulation_steps=accumulation_steps,
            contrastive_weight=contrastive_weight,
            grad_clip_norm=config.GRAD_CLIP_NORM,
            trial_num=trial.number,
            epoch=epoch
        )

        metrics = validate_tuning(
            model=model,
            loader=dev_loader,
            bce_criterion=bce_criterion,
            device=device,
            contrastive_weight=contrastive_weight
        )

        # Score chính để Optuna tối ưu:
        # Ưu tiên AUC, nhưng vẫn tính đến Macro F1 và Balanced Accuracy.
        objective_score = (
            0.60 * metrics["auc"]
            + 0.20 * metrics["macro_f1"]
            + 0.20 * metrics["balanced_acc"]
        )

        print(
            f"\n[Trial {trial.number} | Epoch {epoch}] "
            f"Train Loss={train_loss:.4f} | "
            f"Dev Loss={metrics['loss']:.4f} | "
            f"Dev Acc={metrics['acc']*100:.2f}% | "
            f"Dev AUC={metrics['auc']*100:.2f}% | "
            f"Dev F1={metrics['f1']*100:.2f}% | "
            f"MacroF1={metrics['macro_f1']*100:.2f}% | "
            f"BalAcc={metrics['balanced_acc']*100:.2f}% | "
            f"T={metrics['threshold']:.2f} | "
            f"Score={objective_score:.4f}"
        )

        if objective_score > best_trial_score:
            best_trial_score = objective_score

        if metrics["auc"] > best_trial_auc:
            best_trial_auc = metrics["auc"]

        trial.report(objective_score, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("best_auc", best_trial_auc)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return best_trial_score


# ============================================================
# 8. CẬP NHẬT CONFIG.PY SAU TUNING
# ============================================================
def update_config_file(best_params):
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "core", "config.py")
    )

    if not os.path.exists(config_path):
        print(f"❌ Không tìm thấy config.py tại: {config_path}")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Trỏ lại lipsync-focused dataset
    content = re.sub(
        r'TRAIN_MANIFEST\s*=\s*os\.path\.join\(METADATA_DIR,\s*".*?"\)',
        'TRAIN_MANIFEST = os.path.join(METADATA_DIR, "fakeavceleb_lipsync_train_manifest.csv")',
        content
    )

    content = re.sub(
        r'DEV_MANIFEST\s*=\s*os\.path\.join\(METADATA_DIR,\s*".*?"\)',
        'DEV_MANIFEST = os.path.join(METADATA_DIR, "fakeavceleb_lipsync_dev_manifest.csv")',
        content
    )

    content = re.sub(
        r'TEST_MANIFEST\s*=\s*os\.path\.join\(METADATA_DIR,\s*".*?"\)',
        'TEST_MANIFEST = os.path.join(METADATA_DIR, "fakeavceleb_lipsync_test_manifest.csv")',
        content
    )

    content = re.sub(
        r"IMAGE_SIZE\s*=\s*\d+",
        "IMAGE_SIZE = 224",
        content
    )

    content = re.sub(
        r"MAX_FRAMES\s*=\s*\d+",
        f"MAX_FRAMES = {best_params['max_frames']}",
        content
    )

    content = re.sub(
        r"BATCH_SIZE\s*=\s*\d+",
        f"BATCH_SIZE = {best_params['batch_size']}",
        content
    )

    content = re.sub(
        r"LEARNING_RATE\s*=\s*[\d\.e\-\+]+",
        f"LEARNING_RATE = {best_params['learning_rate']:.4e}",
        content
    )

    content = re.sub(
        r"WEIGHT_DECAY\s*=\s*[\d\.e\-\+]+",
        f"WEIGHT_DECAY = {best_params['weight_decay']:.4e}",
        content
    )

    content = re.sub(
        r"CONTRASTIVE_WEIGHT\s*=\s*[\d\.e\-\+]+",
        f"CONTRASTIVE_WEIGHT = {best_params['contrastive_weight']:.4f}",
        content
    )

    content = re.sub(
        r'BEST_MODEL_NAME\s*=\s*".*?"',
        'BEST_MODEL_NAME = "best_fakeavceleb_lipsync_tuned_model.pth"',
        content
    )

    content = re.sub(
        r'BEST_THRESHOLD_NAME\s*=\s*".*?"',
        'BEST_THRESHOLD_NAME = "best_fakeavceleb_lipsync_tuned_threshold.json"',
        content
    )

    content = re.sub(
        r'HISTORY_NAME\s*=\s*".*?"',
        'HISTORY_NAME = "fakeavceleb_lipsync_tuned_training_history.csv"',
        content
    )

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("\n🎯 Đã cập nhật config.py theo best params:")
    print("   -> Dataset: FakeAVCeleb lipsync-focused")
    print("   -> IMAGE_SIZE = 224")
    print(f"   -> MAX_FRAMES = {best_params['max_frames']}")
    print(f"   -> BATCH_SIZE = {best_params['batch_size']}")
    print(f"   -> LEARNING_RATE = {best_params['learning_rate']:.4e}")
    print(f"   -> WEIGHT_DECAY = {best_params['weight_decay']:.4e}")
    print(f"   -> CONTRASTIVE_WEIGHT = {best_params['contrastive_weight']:.4f}")
    print('   -> BEST_MODEL_NAME = "best_fakeavceleb_lipsync_tuned_model.pth"')

    return True


# ============================================================
# 9. REPORT + CHART
# ============================================================
def generate_report_and_charts(study):
    config = MultimodalConfig()
    best_params = study.best_params
    best_trial = study.best_trial

    report_data = {
        "Chỉ số": [
            "Best Objective Score",
            "Best Dev AUC",
            "Max Frames",
            "Batch Size",
            "Learning Rate",
            "Weight Decay",
            "Contrastive Weight",
            "Trials",
            "Epochs per Trial"
        ],
        "Giá trị": [
            f"{study.best_value:.4f}",
            f"{best_trial.user_attrs.get('best_auc', 0.0):.4f}",
            str(best_params["max_frames"]),
            str(best_params["batch_size"]),
            f"{best_params['learning_rate']:.4e}",
            f"{best_params['weight_decay']:.4e}",
            f"{best_params['contrastive_weight']:.4f}",
            str(N_TRIALS),
            str(TUNING_EPOCHS)
        ]
    }

    df_report = pd.DataFrame(report_data)

    report_path = os.path.join(
        config.METADATA_DIR,
        "lipsync_hyperparameter_tuning_report.csv"
    )

    df_report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("BÁO CÁO HYPERPARAMETER TUNING - LIPSYNC")
    print("=" * 80)
    print(df_report.to_string(index=False))
    print(f"\n📄 Đã lưu report: {report_path}")

    trials_df = study.trials_dataframe()

    chart_path = os.path.join(
        config.PROJECT_ROOT,
        "lipsync_hyperparameter_tuning_history.png"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        trials_df["number"],
        trials_df["value"],
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Objective Score"
    )

    plt.axhline(
        y=study.best_value,
        linestyle="--",
        label=f"Best Score: {study.best_value:.4f}"
    )

    plt.title("FakeAVCeleb Lipsync Hyperparameter Tuning")
    plt.xlabel("Trial")
    plt.ylabel("Objective Score")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Đã lưu chart: {chart_path}")


# ============================================================
# 10. MAIN
# ============================================================
def main():
    print("=== HYPERPARAMETER TUNING - FAKEAVCELEB LIPSYNC-FOCUSED ===")

    config = MultimodalConfig()
    config.create_required_dirs()

    lipsync_train_path = os.path.join(
        config.METADATA_DIR,
        LIPSYNC_TRAIN_MANIFEST_NAME
    )

    lipsync_dev_path = os.path.join(
        config.METADATA_DIR,
        LIPSYNC_DEV_MANIFEST_NAME
    )

    if not os.path.exists(lipsync_train_path):
        print(f"❌ Không tìm thấy: {lipsync_train_path}")
        print("Hãy chạy preprocess_fakeavceleb_lipsync.py trước.")
        return

    if not os.path.exists(lipsync_dev_path):
        print(f"❌ Không tìm thấy: {lipsync_dev_path}")
        print("Hãy chạy preprocess_fakeavceleb_lipsync.py trước.")
        return

    df_train = pd.read_csv(lipsync_train_path)
    df_dev = pd.read_csv(lipsync_dev_path)

    print(f"\n📦 Lipsync Train manifest: {len(df_train)}")
    print(df_train["label"].value_counts())

    print(f"\n📦 Lipsync Dev manifest: {len(df_dev)}")
    print(df_dev["label"].value_counts())

    df_subset_train = balanced_sample(
        df_train,
        n_per_class=TUNING_TRAIN_PER_CLASS,
        seed=42
    )

    df_subset_dev = balanced_sample(
        df_dev,
        n_per_class=TUNING_DEV_PER_CLASS,
        seed=42
    )

    temp_train_path = os.path.join(
        config.METADATA_DIR,
        "temp_lipsync_tuning_train.csv"
    )

    temp_dev_path = os.path.join(
        config.METADATA_DIR,
        "temp_lipsync_tuning_dev.csv"
    )

    df_subset_train.to_csv(temp_train_path, index=False)
    df_subset_dev.to_csv(temp_dev_path, index=False)

    print("\n✅ Tuning train subset:")
    print(df_subset_train["label"].value_counts())
    print(f"Saved: {temp_train_path}")

    print("\n✅ Tuning dev subset:")
    print(df_subset_dev["label"].value_counts())
    print(f"Saved: {temp_dev_path}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=4,
            n_warmup_steps=1
        )
    )

    print("\n🚀 Bắt đầu Optuna tuning...")
    print(f"Trials: {N_TRIALS}")
    print(f"Epochs per trial: {TUNING_EPOCHS}")

    study.optimize(objective, n_trials=N_TRIALS)

    generate_report_and_charts(study)

    best_params = study.best_params

    print("\n🏆 BEST PARAMS:")
    print(f"   Objective Score:    {study.best_value:.4f}")
    print(f"   Max Frames:         {best_params['max_frames']}")
    print(f"   Batch Size:         {best_params['batch_size']}")
    print(f"   Learning Rate:      {best_params['learning_rate']:.4e}")
    print(f"   Weight Decay:       {best_params['weight_decay']:.4e}")
    print(f"   Contrastive Weight: {best_params['contrastive_weight']:.4f}")

    update_config_file(best_params)

    print("\n✅ Tuning xong.")
    print("Bước tiếp theo:")
    print("python experiments\\train.py")
    print("python experiments\\evaluate.py")


if __name__ == "__main__":
    main()