import os
import glob
import json

from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn 

def save_tvt_matrix(model : str, avg : str = "micro"): # tvt : train (lang) vs. test (lang)
    filepaths = glob.glob("output/" + model + "-*/eval.json")
    if not filepaths:
        print("Directory 'output' does not exist, or no (evaluations of) finetuned models exist")
    
    metric_dict = defaultdict(dict)
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

if __name__ == "__main__":
    save_tvt_matrix("mbert", "micro")
    # for model in ["mbert", "xlmr", "glot500"]:
    #     for avg in ["micro", "macro"]:
    #         save_tvt_matrix(model, avg)