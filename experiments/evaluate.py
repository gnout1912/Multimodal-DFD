import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import MultimodalConfig
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


def load_best_threshold(config):
    threshold_path = os.path.join(config.WEIGHTS_DIR, config.BEST_THRESHOLD_NAME)

    if not os.path.exists(threshold_path):
        print("⚠️ Không tìm thấy best threshold JSON. Dùng threshold mặc định 0.5.")
        return 0.5

    with open(threshold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    threshold = float(data.get("best_threshold", 0.5))

    print(f"✅ Loaded best threshold: {threshold:.2f}")
    print(f"Dev AUC lúc lưu: {data.get('dev_auc', 0) * 100:.2f}%")
    print(f"Dev F1 lúc lưu: {data.get('dev_f1', 0) * 100:.2f}%")

    return threshold


def evaluate_model(model, loader, device, config, threshold):
    model.eval()

    bce_criterion = nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for videos, audios, labels in tqdm(loader, desc="Testing FakeAVCeleb"):
            videos = videos.to(device, non_blocking=True)
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, v_emb, a_emb = model(videos, audios)

                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                logits = torch.clamp(logits, min=-10.0, max=10.0)

                loss_bce = bce_criterion(logits, labels)
                loss_con = av_contrastive_loss(v_emb, a_emb, labels)

                loss = loss_bce + config.CONTRASTIVE_WEIGHT * loss_con

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            label_np = labels.detach().cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(label_np)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)
    all_preds = (all_probs >= threshold).astype(int)

    preds_05 = (all_probs >= 0.5).astype(int)

    acc_05 = accuracy_score(all_labels, preds_05)
    precision_05, recall_05, f1_05, _ = precision_recall_fscore_support(
        all_labels,
        preds_05,
        average="binary",
        zero_division=0
    )

    try:
        auc_05 = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_05 = 0.0

    cm_05 = confusion_matrix(all_labels, preds_05)

    print("\n" + "=" * 80)
    print("KẾT QUẢ THAM KHẢO VỚI THRESHOLD = 0.50")
    print("=" * 80)
    print(f"Accuracy:  {acc_05 * 100:.2f}%")
    print(f"AUC-ROC:   {auc_05 * 100:.2f}%")
    print(f"Precision: {precision_05 * 100:.2f}%")
    print(f"Recall:    {recall_05 * 100:.2f}%")
    print(f"F1-Score:  {f1_05 * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm_05)
    
    avg_loss = total_loss / max(1, total)

    acc = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0
    )

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "labels": all_labels,
        "probs": all_probs,
        "preds": all_preds,
    }


def main():
    print("=== EVALUATE FAKEAVCELEB ONLY - GIẢI PHÁP 1 ===")

    config = MultimodalConfig()
    config.create_required_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Device: {device}")

    best_model_path = os.path.join(config.WEIGHTS_DIR, config.BEST_MODEL_NAME)

    if not os.path.exists(best_model_path):
        print(f"❌ Không tìm thấy model weight: {best_model_path}")
        print("Hãy chạy trước: python experiments/train_fakeavceleb.py")
        return

    if not os.path.exists(config.TEST_MANIFEST):
        print(f"❌ Không tìm thấy test manifest: {config.TEST_MANIFEST}")
        print("Hãy chạy trước: python tools/preprocess_fakeavceleb.py")
        return

    threshold = load_best_threshold(config)

    test_dataset = MultimodalDataset(
        config.TEST_MANIFEST,
        config,
        is_train=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    model = MultimodalDeepfakeDetector().to(device)

    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    print(f"✅ Loaded model: {best_model_path}")

    results = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        config=config,
        threshold=threshold
    )

    print("\n" + "=" * 80)
    print("KẾT QUẢ TEST FAKEAVCELEB")
    print("=" * 80)
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy'] * 100:.2f}%")
    print(f"AUC-ROC:   {results['auc'] * 100:.2f}%")
    print(f"Precision: {results['precision'] * 100:.2f}%")
    print(f"Recall:    {results['recall'] * 100:.2f}%")
    print(f"F1-Score:  {results['f1'] * 100:.2f}%")
    print(f"Threshold: {threshold:.2f}")
    print("=" * 80)

    cm = results["confusion_matrix"]

    print("\nConfusion Matrix:")
    print(cm)

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()

        print("\nChi tiết:")
        print(f"✅ True Real:  {tn}")
        print(f"❌ False Fake: {fp}")
        print(f"❌ False Real: {fn}")
        print(f"✅ True Fake:  {tp}")

    result_path = os.path.join(
        config.METADATA_DIR,
        "fakeavceleb_test_results.json"
    )

    save_data = {
        "loss": float(results["loss"]),
        "accuracy": float(results["accuracy"]),
        "auc": float(results["auc"]),
        "precision": float(results["precision"]),
        "recall": float(results["recall"]),
        "f1": float(results["f1"]),
        "threshold": float(threshold),
        "confusion_matrix": results["confusion_matrix"].tolist()
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved test result: {result_path}")


if __name__ == "__main__":
    main()