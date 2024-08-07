from pathlib import Path
import json
import numpy as np
import pandas as pd

import torch
import torchmetrics
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall#, MulticlassAccuracy
from sklearn.metrics import confusion_matrix

from utils.corpus_load import REGISTERS

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


# modified from https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
def save_cfm(preds : torch.Tensor, 
                   labels : torch.Tensor, 
                   out_path : Path
                   ) -> None:
    cf_matrix = confusion_matrix(labels, preds, labels=range(len(REGISTERS)))
    register_totals = np.sum(cf_matrix, axis=1)
    cf_matrix = np.divide(cf_matrix, register_totals[:, None], where=cf_matrix!=0)

    row_labels = [f"{reg}\n(total:{total})" for reg, total in zip(REGISTERS, register_totals)]
    df_cm = pd.DataFrame(cf_matrix, index=row_labels, columns=REGISTERS)

    if not out_path.parent.exists(): # cfm directory does not exist yet
        out_path.parent.mkdir(parents=True)

    with open(out_path, "w") as file: # overwrites existing cfm matrix, if exists
        json.dump(df_cm.to_dict(), file, indent=4)