import os
import glob
import json

from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn 

# tvt : train (lang) vs. test (lang)
def save_tvt_matrix(model : str, 
                    avg : str = "micro",
                    train_langs : list[str] = ["en", "fr", "fi", "id", "sv", "tr"],
                    eval_langs : list[str] = ["en", "fr", "fi", "id", "sv", "tr", "al"]):    
    
    df = pd.DataFrame(index=eval_langs, columns=train_langs).astype(float)
    for train_lang in train_langs:
        with open(f"output/{model}-{train_lang}/eval.json", "r") as file:
            metric_summary = json.load(file)
        
        for eval_lang, metrics in metric_summary.items():
            df.at[eval_lang, train_lang] = metrics[f"{avg}_f1"]

    plt.figure(figsize=(10, 8))
    sn.heatmap(df, vmin=0.0, vmax=1.0, cmap="Purples", annot=True)
    plt.title(f"{model} {avg} f1")
    plt.xlabel('train lang')
    plt.ylabel('test lang')
    plt.savefig(f"output/tvt/{model}-{avg}.png", bbox_inches="tight")

if __name__ == "__main__":
    for model in ["mbert", "xlmr", "glot500"]:
        for avg in ["micro", "macro"]:
            save_tvt_matrix(model, avg)