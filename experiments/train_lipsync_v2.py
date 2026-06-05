import os

def clean_conda_ffmpeg_path():
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

import sys
import gc
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config_lipsync_v2 import MultimodalConfig
from core.dataset.data_loader import MultimodalDataset
from core.models.multimodal_model import MultimodalDeepfakeDetector


def av_contrastive_loss(v_embeds, a_embeds, labels, temperature=0.2):
    v_norm = torch.nn.functional.normalize(v_embeds, dim=-1, eps=1e-8)
    a_norm = torch.nn.functional.normalize(a_embeds, dim=-1, eps=1e-8)

    sim = torch.sum(v_norm * a_norm, dim=-1)
    sim = torch.clamp(sim, min=-0.99, max=0.99)

    target = 1.0 - labels.float()

    logits = sim / temperature

    return nn.functional.binary_cross_entropy_with_logits(logits, target)


from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

def find_best_threshold(labels, probs):
    labels = np.array(labels).astype(int)
    probs = np.array(probs)

    best_t = 0.5
    best_score = -1
    best_acc = 0.0
    best_f1 = 0.0

    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)

        bal_acc = balanced_accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        binary_f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)

        score = 0.5 * bal_acc + 0.5 * macro_f1

        if score > best_score:
            best_score = score
            best_t = float(t)
            best_acc = acc
            best_f1 = binary_f1

    return best_t, best_f1, best_acc


def train_epoch(model, loader, bce_criterion, optimizer, device, accumulation_steps, config):
    model.train()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="Training")

    for batch_idx, (video, audio, labels) in enumerate(progress):
        video = video.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, v_emb, a_emb = model(video, audio)

            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            logits = torch.clamp(logits, min=-10.0, max=10.0)

            loss_bce = bce_criterion(logits, labels)
            loss_con = av_contrastive_loss(v_emb, a_emb, labels)

            loss = loss_bce + config.CONTRASTIVE_WEIGHT * loss_con
            scaled_loss = loss / accumulation_steps

        if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaled_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.GRAD_CLIP_NORM
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        label_np = labels.detach().cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(label_np)

        total_loss += loss.item() * labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, len(all_labels))

    best_t, best_f1, best_acc = find_best_threshold(all_labels, all_probs)

    try:
        auc = roc_auc_score(np.array(all_labels).astype(int), np.array(all_probs))
    except ValueError:
        auc = 0.0

    return avg_loss, best_acc, auc, best_f1, best_t


def validate(model, loader, bce_criterion, device, config):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        progress = tqdm(loader, desc="Validating")

        for video, audio, labels in progress:
            video = video.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, v_emb, a_emb = model(video, audio)

                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                logits = torch.clamp(logits, min=-10.0, max=10.0)

                loss_bce = bce_criterion(logits, labels)
                loss_con = av_contrastive_loss(v_emb, a_emb, labels)

                loss = loss_bce + config.CONTRASTIVE_WEIGHT * loss_con

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            label_np = labels.detach().cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(label_np)

            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / max(1, len(all_labels))

    best_t, best_f1, best_acc = find_best_threshold(all_labels, all_probs)

    try:
        auc = roc_auc_score(np.array(all_labels).astype(int), np.array(all_probs))
    except ValueError:
        auc = 0.0

    return avg_loss, best_acc, auc, best_f1, best_t


def make_loader(dataset, physical_batch, shuffle, config, drop_last):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=physical_batch,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last
    )


def save_threshold(config, threshold, dev_auc, dev_f1, dev_acc):
    threshold_path = os.path.join(config.WEIGHTS_DIR, config.BEST_THRESHOLD_NAME)

    data = {
        "best_threshold": float(threshold),
        "dev_auc": float(dev_auc),
        "dev_f1": float(dev_f1),
        "dev_acc": float(dev_acc)
    }

    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved best threshold: {threshold_path}")


def main():
    print("=== TRAIN FAKEAVCELEB ONLY - GIẢI PHÁP 1 ===")

    config = MultimodalConfig()
    config.create_required_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Device: {device}")

    if not os.path.exists(config.TRAIN_MANIFEST):
        print(f"❌ Không tìm thấy train manifest: {config.TRAIN_MANIFEST}")
        print("Hãy chạy trước: python tools/preprocess_fakeavceleb.py")
        return

    if not os.path.exists(config.DEV_MANIFEST):
        print(f"❌ Không tìm thấy dev manifest: {config.DEV_MANIFEST}")
        print("Hãy chạy trước: python tools/preprocess_fakeavceleb.py")
        return

    train_dataset = MultimodalDataset(
        config.TRAIN_MANIFEST,
        config,
        is_train=True
    )

    dev_dataset = MultimodalDataset(
        config.DEV_MANIFEST,
        config,
        is_train=False
    )

    # RTX 3050 4GB: chạy batch vật lý 1, batch logic dùng accumulation.
    physical_batch = 1
    target_batch = config.BATCH_SIZE
    accumulation_steps = max(1, target_batch // physical_batch)

    train_loader = make_loader(
        train_dataset,
        physical_batch=physical_batch,
        shuffle=True,
        config=config,
        drop_last=True
    )

    dev_loader = make_loader(
        dev_dataset,
        physical_batch=physical_batch,
        shuffle=False,
        config=config,
        drop_last=False
    )

    print(f"📦 Physical batch: {physical_batch}")
    print(f"📦 Target batch: {target_batch}")
    print(f"📦 Accumulation steps: {accumulation_steps}")

    model = MultimodalDeepfakeDetector().to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")

    bce_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_model_path = os.path.join(config.WEIGHTS_DIR, config.BEST_MODEL_NAME)

    best_dev_auc = 0.0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "train_f1": [],
        "train_threshold": [],
        "dev_loss": [],
        "dev_acc": [],
        "dev_auc": [],
        "dev_f1": [],
        "dev_threshold": [],
        "lr": [],
    }

    for epoch in range(1, config.EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()

        current_lr = optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 80)
        print(f"🚀 Epoch [{epoch}/{config.EPOCHS}] | LR: {current_lr:.2e}")
        print("=" * 80)

        train_loss, train_acc, train_auc, train_f1, train_t = train_epoch(
            model=model,
            loader=train_loader,
            bce_criterion=bce_criterion,
            optimizer=optimizer,
            device=device,
            accumulation_steps=accumulation_steps,
            config=config
        )

        dev_loss, dev_acc, dev_auc, dev_f1, dev_t = validate(
            model=model,
            loader=dev_loader,
            bce_criterion=bce_criterion,
            device=device,
            config=config
        )

        scheduler.step(dev_auc)

        print("\n📌 Epoch Result")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | AUC: {train_auc*100:.2f}% | F1: {train_f1*100:.2f}% | T: {train_t:.2f}")
        print(f"Dev   Loss: {dev_loss:.4f} | Acc: {dev_acc*100:.2f}% | AUC: {dev_auc*100:.2f}% | F1: {dev_f1*100:.2f}% | T: {dev_t:.2f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["train_f1"].append(train_f1)
        history["train_threshold"].append(train_t)
        history["dev_loss"].append(dev_loss)
        history["dev_acc"].append(dev_acc)
        history["dev_auc"].append(dev_auc)
        history["dev_f1"].append(dev_f1)
        history["dev_threshold"].append(dev_t)
        history["lr"].append(current_lr)

        history_path = os.path.join(config.METADATA_DIR, config.HISTORY_NAME)
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

        if dev_auc > best_dev_auc:
            best_dev_auc = dev_auc
            patience_counter = 0

            torch.save(model.state_dict(), best_model_path)
            save_threshold(
                config=config,
                threshold=dev_t,
                dev_auc=dev_auc,
                dev_f1=dev_f1,
                dev_acc=dev_acc
            )

            print(f"💾 Saved best model: {best_model_path}")
            print(f"🎯 Best Dev AUC: {best_dev_auc*100:.2f}%")

        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("⛔ Early stopping triggered.")
                break

    print("\n✅ Train FakeAVCeleb hoàn tất.")
    print(f"Best model: {best_model_path}")
    print(f"Training history: {os.path.join(config.METADATA_DIR, config.HISTORY_NAME)}")


if __name__ == "__main__":
    main()