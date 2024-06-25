import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.metrics import confusion_matrix

from utils.corpus_load import REGISTERS

def get_metrics(num_classes : int) -> dict[str, torchmetrics.Metric]:
    metrics = {'accuracy': MulticlassAccuracy(num_classes=num_classes),
               'precision': MulticlassPrecision(num_classes=num_classes),
               'recall': MulticlassRecall(num_classes=num_classes),
               'f1': MulticlassF1Score(num_classes=num_classes)}
    
    # for metric in metrics.values():
    #     metric.to(device)
    return metrics

def add_batch(metrics : dict[str, torchmetrics.Metric], 
              batch_predictions : torch.Tensor, 
              batch_labels : torch.Tensor
              ) -> None:
    for metric in metrics.values():
        metric(batch_predictions, batch_labels)

def get_metric_summary(metrics : dict[str, torchmetrics.Metric]) -> dict[str, float]:
    metric_summary = {}
    for name, metric in metrics.items():
        metric_summary = {**metric_summary, name : metric.compute().item()}
    return metric_summary

def reset_metrics(metrics : dict[str, torchmetrics.Metric]) -> None:
    for metric in metrics.values():
        metric.reset()

# modified from https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
def save_cf_matrix(preds : torch.Tensor, 
                   labels : torch.Tensor, 
                   output_filepath : str
                   ) -> None:
    cf_matrix = confusion_matrix(labels, preds, labels=range(len(REGISTERS)))
    cf_matrix = np.divide(cf_matrix, np.sum(cf_matrix, axis=1)[:, None], where=cf_matrix!=0)
    df_cm = pd.DataFrame(cf_matrix, index=REGISTERS, columns=REGISTERS)

    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(output_filepath)