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
    sn.heatmap(df_cm, annot=True)
    plt.savefig(output_filepath, bbox_inches="tight")


def save_tvt_matrix(model : str, avg : str = "micro"): # tvt : train (lang) vs. test (lang)
    filepaths = glob.glob("output/" + model + "-*/eval.json")
    if not filepaths:
        print("Directory 'output' does not exist, or no (evaluations of) finetuned models exist")
    
    metric_dict = {}
    for filepath in filepaths:
        dirname = os.path.dirname(filepath)
        model_train_lang = os.path.basename(dirname)
        _, train_lang = model_train_lang.split("-")

        with open(filepath, "r") as file:
            metric_summary = json.load(file)
        
        for test_lang, metrics in metric_summary.items():
            metric_dict[train_lang][test_lang] = metrics[f"{avg}_f1"]
    
    df = pd.DataFrame.from_dict(metric_dict)
    plt.figure(figsize=(10, 8))
    sn.heatmap(df, annot=True, cbar_kws={'label': f'{avg}_f1'})
    plt.title(f"{model} {avg} f1")
    plt.xlabel('train lang')
    plt.ylabel('test lang')
    plt.savefig(f"output/{model}-{avg}_f1.png", bbox_inches="tight")

save_tvt_matrix("mbert")