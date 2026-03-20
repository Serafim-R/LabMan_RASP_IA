import torch

from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import create_model
from datasets import create_valid_dataset, create_valid_loader
from torchvision.ops import box_iou

# Evaluation function
def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    score_threshold = 0.25
    TP = 0
    FP = 0
    FN = 0

    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images)

        iou_threshold = 0.5

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            
            scores = outputs[i]['scores'].detach().cpu()
            boxes = outputs[i]['boxes'].detach().cpu()
            labels = outputs[i]['labels'].detach().cpu()

            # filtro de confiança
            keep = scores > score_threshold

            pred_boxes = boxes[keep]
            pred_labels = labels[keep]
            pred_scores = scores[keep]

            true_boxes = targets[i]['boxes'].detach().cpu()
            true_labels = targets[i]['labels'].detach().cpu()

            if len(pred_boxes) == 0:
                FN += len(true_boxes)
                continue

            ious = box_iou(pred_boxes, true_boxes)

            matched_gt = set()

            for p in range(len(pred_boxes)):
                best_iou, gt_idx = ious[p].max(0)

                if best_iou >= iou_threshold and pred_labels[p] == true_labels[gt_idx] and gt_idx.item() not in matched_gt:
                    TP += 1
                    matched_gt.add(gt_idx.item())
                else:
                    FP += 1

            FN += len(true_boxes) - len(matched_gt)


            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()

            # preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            # preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            # preds_dict['labels'] = outputs[i]['labels'].detach().cpu()

            preds_dict['boxes'] = pred_boxes
            preds_dict['scores'] = pred_scores
            preds_dict['labels'] = pred_labels

            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    print("Boxes before filter:", len(boxes))
    print("Boxes after filter:", len(pred_boxes))

    return metric_summary, precision, recall

if __name__ == '__main__':
    # Load the best model and trained weights.
    model = create_model(num_classes=NUM_CLASSES, size=600)
    checkpoint = torch.load('outputs/last_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    test_dataset = create_valid_dataset(
        'data/labman_rasp_IA/test'
    )
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)

    metric_summary, p, r = validate(test_loader, model)
    print(f"mAP_50: {metric_summary['map_50']*100:.2f}")
    print(f"mAP_50_95: {metric_summary['map']*100:.2f}")
    print(f"Precision (IoU = 0.5) = {p*100:.2f}")
    print(f"Recall (IoU = 0.5) = {r*100:.2f}")