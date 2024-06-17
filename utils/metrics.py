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

def add_batch(metrics : dict[str, torchmetrics.Metric], batch_predictions : torch.Tensor, batch_labels : torch.Tensor):
    for metric in metrics.values():
        metric(batch_predictions, batch_labels)

def get_metric_summary(metrics : dict[str, torchmetrics.Metric]) -> dict[str, float]:
    metric_summary = {}
    for name, metric in metrics.items():
        metric_summary = {**metric_summary, name : metric.compute().item()}
    return metric_summary

def reset_metrics(metrics : dict[str, torchmetrics.Metric]):
    for metric in metrics.values():
        metric.reset()