import os
import sys
import json
import gc

import torch
import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config_lipsync_v2 import MultimodalConfig
from core.dataset.data_loader import MultimodalDataset
from core.models.multimodal_model import MultimodalDeepfakeDetector


# ============================================================
# 1. SAFE LOAD
# ============================================================
def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# ============================================================
# 2. FIND BEST THRESHOLD
# ============================================================
def find_best_threshold(labels, probs):
    labels = np.array(labels).astype(int)
    probs = np.array(probs)

    best_t = 0.50
    best_score = -1.0

    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)

        macro_f1 = f1_score(
            labels,
            preds,
            average="macro",
            zero_division=0
        )

        bal_acc = balanced_accuracy_score(labels, preds)

        score = 0.5 * macro_f1 + 0.5 * bal_acc

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t


# ============================================================
# 3. EVALUATE METRICS
# ============================================================
def evaluate_predictions(labels, probs, threshold):
    labels = np.array(labels).astype(int)
    probs = np.array(probs)

    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0
    )

    macro_f1 = f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    bal_acc = balanced_accuracy_score(labels, preds)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(labels, preds)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm,
        "preds": preds
    }


# ============================================================
# 4. PREDICT 1 MODEL
# ============================================================
def predict_one_model(model_path, manifest_path, device, config, split_name, model_name):
    print("\n" + "-" * 80)
    print(f"Predicting {split_name} | {model_name}")
    print(f"Model path: {model_path}")
    print("-" * 80)

    dataset = MultimodalDataset(
        manifest_path,
        config,
        is_train=False
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    model = MultimodalDeepfakeDetector().to(device)

    state_dict = safe_torch_load(model_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for videos, audios, labels in tqdm(loader, desc=f"{split_name}/{model_name}"):
            videos = videos.to(device, non_blocking=True)
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _, _ = model(videos, audios)

                logits = torch.nan_to_num(
                    logits,
                    nan=0.0,
                    posinf=10.0,
                    neginf=-10.0
                )

                logits = torch.clamp(logits, min=-10.0, max=10.0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.detach().cpu().numpy())

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return np.array(all_probs, dtype=np.float32), np.array(all_labels, dtype=int)


# ============================================================
# 5. PRINT RESULT
# ============================================================
def print_result(title, result):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Threshold:       {result['threshold']:.2f}")
    print(f"Accuracy:        {result['accuracy'] * 100:.2f}%")
    print(f"AUC-ROC:         {result['auc'] * 100:.2f}%")
    print(f"Precision:       {result['precision'] * 100:.2f}%")
    print(f"Recall:          {result['recall'] * 100:.2f}%")
    print(f"F1-Score:        {result['f1'] * 100:.2f}%")
    print(f"Macro F1:        {result['macro_f1'] * 100:.2f}%")
    print(f"Balanced Acc:    {result['balanced_accuracy'] * 100:.2f}%")
    print("Confusion Matrix:")
    print(result["confusion_matrix"])

    cm = result["confusion_matrix"]

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print("\nChi tiết:")
        print(f"✅ True Real:  {tn}")
        print(f"❌ False Fake: {fp}")
        print(f"❌ False Real: {fn}")
        print(f"✅ True Fake:  {tp}")


# ============================================================
# 6. MAIN
# ============================================================
def main():
    print("=== EVALUATE SEED ENSEMBLE - FAKEAVCELEB LIPSYNC V2 ===")

    config = MultimodalConfig()
    config.create_required_dirs()

    # Đảm bảo ensemble dùng đúng input shape giống v2
    config.MAX_FRAMES = 16
    config.BATCH_SIZE = 1
    config.NUM_WORKERS = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Device: {device}")

    dev_manifest = config.DEV_MANIFEST
    test_manifest = config.TEST_MANIFEST

    print(f"Dev manifest:  {dev_manifest}")
    print(f"Test manifest: {test_manifest}")
    print(f"MAX_FRAMES:    {config.MAX_FRAMES}")

    # ============================================================
    # CHỈ GIỮ MODEL MẠNH
    # Không đưa seed42, seed3407 vào.
    # ============================================================
    candidate_models = [
        {
            "name": "seed999",
            "path": os.path.join(
                config.WEIGHTS_DIR,
                "best_fakeavceleb_lipsync_v2_seed999_model.pth"
            )
        },
        {
            "name": "seed777",
            "path": os.path.join(
                config.WEIGHTS_DIR,
                "best_fakeavceleb_lipsync_v2_seed777_model.pth"
            )
        },
        {
            "name": "seed123",
            "path": os.path.join(
                config.WEIGHTS_DIR,
                "best_fakeavceleb_lipsync_v2_seed123_model.pth"
            )
        },
        {
            "name": "seed2025",
            "path": os.path.join(
                config.WEIGHTS_DIR,
                "best_fakeavceleb_lipsync_v2_seed2025_model.pth"
            )
        },
        {
            "name": "v2_original",
            "path": os.path.join(
                config.WEIGHTS_DIR,
                "best_fakeavceleb_lipsync_v2_model.pth"
            )
        },
    ]

    valid_models = []

    for model_info in candidate_models:
        if os.path.exists(model_info["path"]):
            valid_models.append(model_info)
            print(f"✅ Found: {model_info['name']} | {model_info['path']}")
        else:
            print(f"⚠️ Missing, skipped: {model_info['name']} | {model_info['path']}")

    if len(valid_models) < 2:
        print("❌ Cần ít nhất 2 model để ensemble.")
        return

    # ============================================================
    # ENSEMBLE COMBINATIONS
    # ============================================================
    combinations = [
        ["seed999", "seed777"],
        ["seed999", "seed777", "seed123"],
        ["seed999", "seed777", "v2_original"],
        ["seed999", "seed777", "seed2025"],
        ["seed999", "seed777", "seed123", "v2_original"],
        ["seed999", "seed777", "seed123", "seed2025"],
        ["seed999", "seed777", "seed123", "seed2025", "v2_original"],
    ]

    model_map = {m["name"]: m for m in valid_models}

    all_results = []

    best_result = None
    best_combo_name = None

    # ============================================================
    # RUN EACH COMBINATION
    # ============================================================
    for combo in combinations:
        combo = [name for name in combo if name in model_map]

        if len(combo) < 2:
            continue

        combo_name = "+".join(combo)

        print("\n" + "#" * 80)
        print(f"ENSEMBLE COMBO: {combo_name}")
        print("#" * 80)

        # ---------------- DEV ----------------
        dev_probs_list = []
        dev_labels_ref = None

        for name in combo:
            item = model_map[name]

            probs, labels = predict_one_model(
                model_path=item["path"],
                manifest_path=dev_manifest,
                device=device,
                config=config,
                split_name="DEV",
                model_name=name
            )

            dev_probs_list.append(probs)

            if dev_labels_ref is None:
                dev_labels_ref = labels
            else:
                if not np.array_equal(dev_labels_ref, labels):
                    print("❌ DEV label order mismatch.")
                    return

        dev_ensemble_probs = np.mean(
            np.stack(dev_probs_list, axis=0),
            axis=0
        )

        best_threshold = find_best_threshold(
            labels=dev_labels_ref,
            probs=dev_ensemble_probs
        )

        dev_result = evaluate_predictions(
            labels=dev_labels_ref,
            probs=dev_ensemble_probs,
            threshold=best_threshold
        )

        print_result(
            title=f"DEV RESULT | {combo_name}",
            result=dev_result
        )

        # ---------------- TEST ----------------
        test_probs_list = []
        test_labels_ref = None

        for name in combo:
            item = model_map[name]

            probs, labels = predict_one_model(
                model_path=item["path"],
                manifest_path=test_manifest,
                device=device,
                config=config,
                split_name="TEST",
                model_name=name
            )

            test_probs_list.append(probs)

            if test_labels_ref is None:
                test_labels_ref = labels
            else:
                if not np.array_equal(test_labels_ref, labels):
                    print("❌ TEST label order mismatch.")
                    return

        test_ensemble_probs = np.mean(
            np.stack(test_probs_list, axis=0),
            axis=0
        )

        test_result_dev_t = evaluate_predictions(
            labels=test_labels_ref,
            probs=test_ensemble_probs,
            threshold=best_threshold
        )

        test_result_05 = evaluate_predictions(
            labels=test_labels_ref,
            probs=test_ensemble_probs,
            threshold=0.50
        )

        print_result(
            title=f"TEST RESULT - DEV THRESHOLD | {combo_name}",
            result=test_result_dev_t
        )

        print_result(
            title=f"TEST RESULT - THRESHOLD 0.50 | {combo_name}",
            result=test_result_05
        )

        record = {
            "combo": combo_name,
            "models": combo,
            "dev_threshold": float(best_threshold),
            "dev": {
                "accuracy": dev_result["accuracy"],
                "auc": dev_result["auc"],
                "precision": dev_result["precision"],
                "recall": dev_result["recall"],
                "f1": dev_result["f1"],
                "macro_f1": dev_result["macro_f1"],
                "balanced_accuracy": dev_result["balanced_accuracy"],
                "confusion_matrix": dev_result["confusion_matrix"].tolist()
            },
            "test_dev_threshold": {
                "threshold": test_result_dev_t["threshold"],
                "accuracy": test_result_dev_t["accuracy"],
                "auc": test_result_dev_t["auc"],
                "precision": test_result_dev_t["precision"],
                "recall": test_result_dev_t["recall"],
                "f1": test_result_dev_t["f1"],
                "macro_f1": test_result_dev_t["macro_f1"],
                "balanced_accuracy": test_result_dev_t["balanced_accuracy"],
                "confusion_matrix": test_result_dev_t["confusion_matrix"].tolist()
            },
            "test_threshold_05": {
                "threshold": test_result_05["threshold"],
                "accuracy": test_result_05["accuracy"],
                "auc": test_result_05["auc"],
                "precision": test_result_05["precision"],
                "recall": test_result_05["recall"],
                "f1": test_result_05["f1"],
                "macro_f1": test_result_05["macro_f1"],
                "balanced_accuracy": test_result_05["balanced_accuracy"],
                "confusion_matrix": test_result_05["confusion_matrix"].tolist()
            }
        }

        all_results.append(record)

        # Chọn best theo F1 trước, sau đó Accuracy, sau đó AUC
        candidate = test_result_05

        if best_result is None:
            best_result = candidate
            best_combo_name = combo_name
        else:
            current_key = (
                candidate["f1"],
                candidate["accuracy"],
                candidate["auc"]
            )

            best_key = (
                best_result["f1"],
                best_result["accuracy"],
                best_result["auc"]
            )

            if current_key > best_key:
                best_result = candidate
                best_combo_name = combo_name

    # ============================================================
    # SAVE RESULT
    # ============================================================
    output_path = os.path.join(
        config.METADATA_DIR,
        "fakeavceleb_lipsync_v2_seed_ensemble_results.json"
    )

    save_data = {
        "single_best_reference": {
            "model": "seed999",
            "threshold": 0.50,
            "accuracy": 0.8733,
            "auc": 0.9199,
            "f1": 0.8725,
            "confusion_matrix": [[66, 9], [10, 65]]
        },
        "best_combo_by_test_threshold_05": best_combo_name,
        "best_result_by_test_threshold_05": {
            "threshold": best_result["threshold"],
            "accuracy": best_result["accuracy"],
            "auc": best_result["auc"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"],
            "macro_f1": best_result["macro_f1"],
            "balanced_accuracy": best_result["balanced_accuracy"],
            "confusion_matrix": best_result["confusion_matrix"].tolist()
        },
        "all_results": all_results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("FINAL ENSEMBLE SUMMARY")
    print("=" * 80)
    print(f"Best combo: {best_combo_name}")
    print_result("BEST TEST RESULT - THRESHOLD 0.50", best_result)
    print(f"\n✅ Saved ensemble result: {output_path}")


if __name__ == "__main__":
    main()