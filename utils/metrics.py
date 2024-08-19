from pathlib import Path
import json

import torch
import torchmetrics
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall#, MulticlassAccuracy

class Metrics:
    def __init__(self, num_classes : int, device : torch.device):
        self.metrics = {#"accuracy": MulticlassAccuracy(num_classes=num_classes),
                   "precision": MulticlassPrecision(num_classes=num_classes),
                   "recall": MulticlassRecall(num_classes=num_classes),
                   "micro_f1": MulticlassF1Score(num_classes=num_classes, average="micro"),
                   "macro_f1": MulticlassF1Score(num_classes=num_classes, average="macro")}
    
        for metric in self.metrics.values():
            metric.to(device)

    def add_batch(self, batch_predictions : torch.Tensor, batch_labels : torch.Tensor) -> None:
        for metric in self.metrics.values():
            metric(batch_predictions, batch_labels)

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
    
    def get_summary(self) -> dict[str, float]:
        metric_summary = {}
        for name, metric in self.metrics.items():
            metric_summary = {**metric_summary, name : metric.compute().item()}
        return metric_summary
    
    def write_summary(self, path : Path, key : str):
        if path.exists() and path.is_file():
            with open(path, "r") as file:
                past_summaries = json.load(file)
        else:
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
            past_summaries = {}
        
        summary = self.get_summary()
        past_summaries[key] = summary

        # organize keys (either languages or training epochs) by alphabetical order if not already
        summary_keys = list(past_summaries.keys())
        sorted_keys = sorted(summary_keys)
        if summary_keys != sorted_keys:
            sorted_summary = {key : past_summaries[key] for key in sorted_keys}
            past_summaries = sorted_summary

        with open(path, "w") as file:
            json.dump(past_summaries, file, indent=4)