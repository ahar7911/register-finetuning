import os
import glob
import json
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import torch
import torchmetrics
from torchmetrics.classification import MulticlassF1Score#, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from sklearn.metrics import confusion_matrix

from utils.corpus_load import REGISTERS

class Metrics:
    def __init__(self, num_classes : int, device : torch.device):
        self.metrics = {#"accuracy": MulticlassAccuracy(num_classes=num_classes),
                   #"precision": MulticlassPrecision(num_classes=num_classes),
                   #"recall": MulticlassRecall(num_classes=num_classes),
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
    
    def write_summary(self, filepath : str, key : str):
        if os.path.exists(filepath):
            with open(filepath, "r") as file:
                past_summaries = json.load(file)
        else:
            past_summaries = {}
        
        summary = self.get_summary()
        past_summaries[key] = summary

        with open(filepath, "w") as file:
            json.dump(past_summaries, file, indent=4)


# modified from https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
def save_cf_matrix(preds : torch.Tensor, 
                   labels : torch.Tensor, 
                   output_filepath : str
                   ) -> None:
    cf_matrix = confusion_matrix(labels, preds, labels=range(len(REGISTERS)))
    cf_matrix = np.divide(cf_matrix, np.sum(cf_matrix, axis=1)[:, None], where=cf_matrix!=0)
    df_cm = pd.DataFrame(cf_matrix, index=REGISTERS, columns=REGISTERS)

    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, cmap="Purples")
    plt.savefig(output_filepath, bbox_inches="tight")