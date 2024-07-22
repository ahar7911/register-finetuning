import os
import glob
import json

from collections import defaultdict
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn 

# tvt : train (lang) vs. test (lang)
def plot_tvt_matrix(ax : matplotlib.axes.Axes,
                    model : str, 
                    avg : str = "micro",
                    train_langs : list[str] = ["en", "fr", "fi", "id", "sv", "tr"],
                    eval_langs : list[str] = ["en", "fr", "fi", "id", "sv", "tr", "al"],
                    ) -> None:    
    df = pd.DataFrame(index=eval_langs, columns=train_langs).astype(float)
    for train_lang in train_langs:
        with open(f"output/{model}-{train_lang}/eval.json", "r") as file:
            metric_summary = json.load(file)
        
        for eval_lang, metrics in metric_summary.items():
            df.at[eval_lang, train_lang] = metrics[f"{avg}_f1"]

    sn.heatmap(df, vmin=0.0, vmax=1.0, cmap="Purples", annot=True, cbar=False, ax=ax)
    ax.title(f"{model} {avg} f1")
    ax.xlabel('train lang')
    ax.ylabel('test lang')

def get_langs(model : str, 
              basedir : str = "output"
              ) -> tuple[list[str], list[str]]:
    train_langs = []
    test_langs = set()
    
    for folder in os.listdir(basedir):
        if not os.path.isdir(os.path.join(basedir, folder)):
            continue
        
        parts = folder.split('-')
        if len(parts) == 2 and parts[0] == model:
            lang = parts[1]
            train_langs.append(lang)
            
            json_path = os.path.join(basedir, folder, 'eval.json')
            if os.path.isfile(json_path):
                with open(json_path, 'r') as file:
                    metrics = json.load(file)
                    test_langs.update(list(metrics.keys()))

    only_test_langs = list(test_langs - set(train_langs))
    train_langs = sorted(train_langs)
    test_langs = train_langs + sorted(only_test_langs)
    
    return train_langs, test_langs

def main(models : list[str] = ["mbert", "xlmr", "glot500"],
         avgs : list[str] = ["micro", "macro"]
         ) -> None:
    
    fig, axs = plt.subplots(len(avgs), len(models))
    fig.title = "train vs. eval lang"
    for model_ind, model in enumerate(models):
        train_langs, test_langs = get_langs(model)
        for avg_ind, avg in enumerate(avgs):
            ax_ind = (model_ind) + (avg_ind * 3)
            plot_tvt_matrix(axs[ax_ind], model, avg, train_langs=train_langs, test_langs=test_langs)
    fig.savefig(f"output/tvt.png", bbox_inches="tight")

if __name__ == "__main__":
    main()