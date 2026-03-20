from sklearn.metrics import precision_score, recall_score
import torch
import numpy as np

def calculate_precision_and_recall(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    
    return precision, recall

def evaluate_model(model, dataloader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for target, output in zip(targets, outputs):
                true_labels.append(target["labels"].cpu().numpy())
                predicted_labels.append(output["labels"].cpu().numpy())
    
    true_labels = np.concatenate(true_labels)
    predicted_labels = np.concatenate(predicted_labels)
    
    precision, recall = calculate_precision_and_recall(true_labels, predicted_labels)
    
    return precision, recall

