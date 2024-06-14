import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

def get_metrics(num_classes : int, device : torch.device) -> dict[str, torchmetrics.Metric]:
    metrics = {'accuracy': MulticlassAccuracy(num_classes=num_classes),
               'precision': MulticlassPrecision(num_classes=num_classes),
               'recall': MulticlassRecall(num_classes=num_classes),
               'f1': MulticlassF1Score(num_classes=num_classes)}
    
    for metric in metrics.values():
        metric.to(device)

    return metrics