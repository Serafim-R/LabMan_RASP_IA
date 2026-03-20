import torch
import time

from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import create_model
from datasets import create_valid_dataset, create_valid_loader
from torchvision.ops import box_iou

import os
import matplotlib.pyplot as plt


def validate(valid_data_loader, model):

    print("Validating...")
    model.eval()

    score_threshold = 0.25
    iou_threshold = 0.5

    preds = []
    target = []

    TP = 0
    FP = 0
    FN = 0

    total_inference_time = 0
    total_images = 0

    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    all_scores = []
    all_tp_flags = []

    for data in prog_bar:

        images, targets = data

        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():

            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

        total_inference_time += end_time - start_time
        total_images += len(images)

        for i in range(len(images)):

            scores = outputs[i]['scores'].detach().cpu()
            boxes = outputs[i]['boxes'].detach().cpu()
            labels = outputs[i]['labels'].detach().cpu()

            true_boxes = targets[i]['boxes'].detach().cpu()
            true_labels = targets[i]['labels'].detach().cpu()

            # ordenar por score
            order = scores.argsort(descending=True)

            scores = scores[order]
            boxes = boxes[order]
            labels = labels[order]

            # aplicar filtro de confiança
            keep = scores > score_threshold

            pred_boxes = boxes[keep]
            pred_scores = scores[keep]
            pred_labels = labels[keep]

            # -------- mAP --------

            preds_dict = {}
            true_dict = {}

            preds_dict['boxes'] = pred_boxes
            preds_dict['scores'] = pred_scores
            preds_dict['labels'] = pred_labels

            true_dict['boxes'] = true_boxes
            true_dict['labels'] = true_labels

            preds.append(preds_dict)
            target.append(true_dict)

            # -------- Precision / Recall --------

            if len(pred_boxes) == 0:
                FN += len(true_boxes)
                continue

            ious = box_iou(pred_boxes, true_boxes)

            matched_gt = set()

            for p in range(len(pred_boxes)):

                best_iou, gt_idx = ious[p].max(0)

                if (
                    best_iou >= iou_threshold
                    and pred_labels[p] == true_labels[gt_idx]
                    and gt_idx.item() not in matched_gt
                ):
                    TP += 1
                    matched_gt.add(gt_idx.item())
                    all_tp_flags.append(1)
                else:
                    FP += 1
                    all_tp_flags.append(0)
                all_scores.append(pred_scores[p].item())
            FN += len(true_boxes) - len(matched_gt)

    # -------- mAP --------

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()

    # -------- Precision / Recall --------

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    # -------- tempo de inferência --------

    avg_inference_time = total_inference_time / total_images
    fps = 1 / avg_inference_time

    # =============Gráficos=================
    import numpy as np

    scores = np.array(all_scores)
    tp_flags = np.array(all_tp_flags)

    # ordenar por score
    order = np.argsort(-scores)

    scores = scores[order]
    tp_flags = tp_flags[order]

    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum(1 - tp_flags)

    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall_curve = tp_cumsum / (TP + FN + 1e-6)

    plt.figure()

    plt.plot(recall_curve, precision_curve)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")

    plt.grid(True)

    plt.savefig("outputs_val/precision_recall_curve.png")

    plt.close()

    return metric_summary, precision, recall, f1, avg_inference_time, fps


if __name__ == "__main__":

    model = create_model(num_classes=NUM_CLASSES, size=600)

    checkpoint = torch.load("outputs/last_model.pth", map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE).eval()

    test_dataset = create_valid_dataset(
        "data/labman_rasp_IA/test"
    )

    test_loader = create_valid_loader(
        test_dataset,
        num_workers=NUM_WORKERS
    )

    os.makedirs("outputs_val", exist_ok=True)

    metric_summary, p, r, f1, t, fps = validate(test_loader, model)

    print("\nEvaluation Results\n")

    print(f"Precision (IoU=0.5): {p*100:.2f}")
    print(f"Recall (IoU=0.5): {r*100:.2f}")
    print(f"F1-score: {f1*100:.2f}")

    print(f"mAP@0.5: {metric_summary['map_50']*100:.2f}")
    print(f"mAP@0.5:0.95: {metric_summary['map']*100:.2f}")

    print(f"Inference time: {t*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")

    results_path = "outputs_val/validation_results.txt"

    with open(results_path, "w") as f:

        f.write("Validation Results\n\n")

        f.write(f"Precision (IoU=0.5): {p*100:.2f}\n")
        f.write(f"Recall (IoU=0.5): {r*100:.2f}\n")
        f.write(f"F1-score: {f1*100:.2f}\n")

        f.write(f"mAP@0.5: {metric_summary['map_50']*100:.2f}\n")
        f.write(f"mAP@0.5:0.95: {metric_summary['map']*100:.2f}\n")

        f.write(f"Inference time: {t*1000:.2f} ms\n")
        f.write(f"FPS: {fps:.2f}\n")