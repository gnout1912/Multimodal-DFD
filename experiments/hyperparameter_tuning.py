import os
import sys
import re
import gc
import json
import subprocess
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import MultimodalConfig
from core.dataset.data_loader import MultimodalDataset
from core.models.multimodal_model import MultimodalDeepfakeDetector

# Import đúng file train FakeAVCeleb
from experiments.train import (
    av_contrastive_loss,
    validate
)


def balanced_sample(df, n_per_class, seed=42):
    if "label" not in df.columns:
        raise ValueError("Manifest thiếu cột label.")

    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]

    real_n = min(n_per_class, len(real_df))
    fake_n = min(n_per_class, len(fake_df))

    if real_n == 0 or fake_n == 0:
        raise ValueError(
            f"Dataset thiếu class. Real={len(real_df)}, Fake={len(fake_df)}"
        )

    sampled = pd.concat([
        real_df.sample(real_n, random_state=seed),
        fake_df.sample(fake_n, random_state=seed),
    ])

    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled


def train_epoch_tuning(
    model,
    loader,
    bce_criterion,
    optimizer,
    device,
    accumulation_steps,
    trial_num,
    epoch,
    contrastive_weight,
    grad_clip_norm
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

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

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")


def objective(trial):
    torch.cuda.empty_cache()
    gc.collect()

    config = MultimodalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tuning nhẹ để phù hợp RTX 3050
    config.IMAGE_SIZE = 224
    config.MAX_FRAMES = 12

    target_batch = trial.suggest_int("batch_size", 2, 4)
    lr = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
    contrastive_weight = trial.suggest_float("contrastive_weight", 0.03, 0.10)

    physical_batch = 1
    accumulation_steps = max(1, target_batch // physical_batch)

    temp_train_path = os.path.join(
        config.METADATA_DIR,
        "temp_fakeavceleb_tuning_train.csv"
    )
    temp_dev_path = os.path.join(
        config.METADATA_DIR,
        "temp_fakeavceleb_tuning_dev.csv"
    )

    train_dataset = MultimodalDataset(temp_train_path, config, is_train=True)
    dev_dataset = MultimodalDataset(temp_dev_path, config, is_train=False)

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

    best_auc_for_trial = 0.0

    for epoch in range(1, 4):
        train_epoch_tuning(
            model=model,
            loader=train_loader,
            bce_criterion=bce_criterion,
            optimizer=optimizer,
            device=device,
            accumulation_steps=accumulation_steps,
            trial_num=trial.number,
            epoch=epoch,
            contrastive_weight=contrastive_weight,
            grad_clip_norm=config.GRAD_CLIP_NORM
        )

        dev_loss, dev_acc, dev_auc, dev_f1, best_t = validate(
            model=model,
            loader=dev_loader,
            bce_criterion=bce_criterion,
            device=device,
            config=config
        )

        print(
            f"\n[Trial {trial.number} | Epoch {epoch}] "
            f"Dev Loss={dev_loss:.4f} | "
            f"Dev Acc={dev_acc*100:.2f}% | "
            f"Dev AUC={dev_auc*100:.2f}% | "
            f"Dev F1={dev_f1*100:.2f}% | "
            f"Best T={best_t:.2f}"
        )

        if dev_auc > best_auc_for_trial:
            best_auc_for_trial = dev_auc

        trial.report(dev_auc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return best_auc_for_trial


def update_config_file(best_lr, best_wd, best_batch, best_contrastive_weight):
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "core", "config.py")
    )

    if not os.path.exists(config_path):
        print(f"❌ Không tìm thấy config.py tại: {config_path}")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(
        r"LEARNING_RATE\s*=\s*[\d\.e\-\+]+",
        f"LEARNING_RATE = {best_lr:.4e}",
        content
    )

    content = re.sub(
        r"WEIGHT_DECAY\s*=\s*[\d\.e\-\+]+",
        f"WEIGHT_DECAY = {best_wd:.4e}",
        content
    )

    content = re.sub(
        r"BATCH_SIZE\s*=\s*\d+",
        f"BATCH_SIZE = {best_batch}",
        content
    )

    content = re.sub(
        r"IMAGE_SIZE\s*=\s*\d+",
        "IMAGE_SIZE = 224",
        content
    )

    content = re.sub(
        r"MAX_FRAMES\s*=\s*\d+",
        "MAX_FRAMES = 16",
        content
    )

    content = re.sub(
        r"CONTRASTIVE_WEIGHT\s*=\s*[\d\.e\-\+]+",
        f"CONTRASTIVE_WEIGHT = {best_contrastive_weight:.4f}",
        content
    )

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("\n🎯 Đã cập nhật config.py:")
    print("   -> IMAGE_SIZE = 224")
    print("   -> MAX_FRAMES = 16")
    print(f"   -> BATCH_SIZE = {best_batch}")
    print(f"   -> LEARNING_RATE = {best_lr:.4e}")
    print(f"   -> WEIGHT_DECAY = {best_wd:.4e}")
    print(f"   -> CONTRASTIVE_WEIGHT = {best_contrastive_weight:.4f}")

    return True


def generate_report_and_charts(study, baseline_lr, baseline_wd):
    config = MultimodalConfig()
    best_params = study.best_params

    data = {
        "Chỉ số cấu hình": [
            "Batch Size",
            "Learning Rate",
            "Weight Decay",
            "Contrastive Weight",
            "Best Dev AUC"
        ],
        "Baseline": [
            str(config.BATCH_SIZE),
            f"{baseline_lr:.2e}",
            f"{baseline_wd:.2e}",
            str(config.CONTRASTIVE_WEIGHT),
            "Chưa tối ưu"
        ],
        "Tối ưu": [
            str(best_params["batch_size"]),
            f"{best_params['learning_rate']:.2e}",
            f"{best_params['weight_decay']:.2e}",
            f"{best_params['contrastive_weight']:.4f}",
            f"{study.best_value:.4f}"
        ]
    }

    df_report = pd.DataFrame(data)

    report_path = os.path.join(
        config.METADATA_DIR,
        "fakeavceleb_hyperparameter_comparison.csv"
    )

    df_report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("BẢNG SO SÁNH HYPERPARAMETER FAKEAVCELEB")
    print("=" * 80)
    print(df_report.to_string(index=False))
    print(f"\n📄 Đã lưu report: {report_path}")

    trials_df = study.trials_dataframe()

    chart_path = os.path.join(
        config.PROJECT_ROOT,
        "fakeavceleb_hyperparameter_tuning_history.png"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        trials_df["number"],
        trials_df["value"],
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Dev AUC"
    )

    plt.axhline(
        y=study.best_value,
        linestyle="--",
        label=f"Best Dev AUC: {study.best_value:.4f}"
    )

    plt.title("FakeAVCeleb Hyperparameter Tuning theo Dev AUC")
    plt.xlabel("Trial")
    plt.ylabel("Dev AUC")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Đã lưu chart: {chart_path}")


def main():
    print("=== HYPERPARAMETER TUNING - FAKEAVCELEB ONLY ===")

    config = MultimodalConfig()
    config.create_required_dirs()

    baseline_lr = config.LEARNING_RATE
    baseline_wd = config.WEIGHT_DECAY

    if not os.path.exists(config.FAKEAVCELEB_TRAIN_MANIFEST):
        print(f"❌ Không tìm thấy train manifest: {config.FAKEAVCELEB_TRAIN_MANIFEST}")
        print("Hãy chạy trước: python tools\\preprocess_fakeavceleb.py")
        return

    if not os.path.exists(config.FAKEAVCELEB_DEV_MANIFEST):
        print(f"❌ Không tìm thấy dev manifest: {config.FAKEAVCELEB_DEV_MANIFEST}")
        print("Hãy chạy trước: python tools\\preprocess_fakeavceleb.py")
        return

    df_train = pd.read_csv(config.FAKEAVCELEB_TRAIN_MANIFEST)
    df_dev = pd.read_csv(config.FAKEAVCELEB_DEV_MANIFEST)

    print(f"FakeAVCeleb Train manifest: {len(df_train)}")
    print(f"FakeAVCeleb Dev manifest:   {len(df_dev)}")

    print("\n📊 Train label:")
    print(df_train["label"].value_counts())

    print("\n📊 Dev label:")
    print(df_dev["label"].value_counts())

    # Tuning nhanh:
    # Train: 100 real + 100 fake = 200 mẫu
    # Dev:   50 real + 50 fake = 100 mẫu
    df_subset_train = balanced_sample(df_train, n_per_class=100, seed=42)
    df_subset_dev = balanced_sample(df_dev, n_per_class=50, seed=42)

    temp_train_path = os.path.join(
        config.METADATA_DIR,
        "temp_fakeavceleb_tuning_train.csv"
    )
    temp_dev_path = os.path.join(
        config.METADATA_DIR,
        "temp_fakeavceleb_tuning_dev.csv"
    )

    df_subset_train.to_csv(temp_train_path, index=False)
    df_subset_dev.to_csv(temp_dev_path, index=False)

    print(f"\n✅ Tuning train subset: {len(df_subset_train)}")
    print(df_subset_train["label"].value_counts())

    print(f"\n✅ Tuning dev subset: {len(df_subset_dev)}")
    print(df_subset_dev["label"].value_counts())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1
        )
    )

    print("\n🚀 Bắt đầu tuning FakeAVCeleb theo Dev AUC...")
    print("Lần đầu nên để n_trials=5. Khi ổn có thể tăng 10 hoặc 15.")

    study.optimize(objective, n_trials=5)

    generate_report_and_charts(study, baseline_lr, baseline_wd)

    best_lr = study.best_params["learning_rate"]
    best_wd = study.best_params["weight_decay"]
    best_batch = study.best_params["batch_size"]
    best_contrastive_weight = study.best_params["contrastive_weight"]

    print("\n🏆 Best Trial:")
    print(f"   Dev AUC:            {study.best_value:.4f}")
    print(f"   Batch Size:         {best_batch}")
    print(f"   Learning Rate:      {best_lr:.4e}")
    print(f"   Weight Decay:       {best_wd:.4e}")
    print(f"   Contrastive Weight: {best_contrastive_weight:.4f}")

    if update_config_file(
        best_lr=best_lr,
        best_wd=best_wd,
        best_batch=best_batch,
        best_contrastive_weight=best_contrastive_weight
    ):
        print("\n✅ Tuning xong. Bây giờ train chính thức bằng:")
        print("python experiments\\train_fakeavceleb.py")

        # Không tự động gọi train để tránh chạy nhầm quá lâu.
        # Bạn tự chạy train sau khi kiểm tra best params.


if __name__ == "__main__":
    main()