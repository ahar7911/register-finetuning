import os
import json
import math

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn 


def confusion_matrices(base_dir : str = "output"):
    for folder in os.listdir(base_dir):
        if not os.path.isdir(os.path.join(base_dir, folder)):
            continue

        cfm_folder = os.path.join(base_dir, folder, "cfm")
        model, train_lang = folder.split("-")

        files = []
        for file in os.listdir(cfm_folder):
            if os.path.isfile(os.path.join(cfm_folder, file)) and os.path.splitext(file)[1] == ".json":
                files.append(file)

        same_lang_file = f"{train_lang}.json"
        if same_lang_file in files:
            files.remove(same_lang_file)
            files.insert(0, same_lang_file)

        num_cols = math.ceil(math.sqrt(len(files)))
        num_rows = math.ceil(len(files) / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows))
        fig.suptitle(f"confusion matrices for {model} trained on {train_lang}", fontsize=20)
        axes = axes.flatten()

        for ax, filepath in zip(axes, files):
            eval_lang, _ = os.path.splitext(filepath)
            filepath = os.path.join(cfm_folder, filepath)
            with open(filepath, "r") as file:
                cfm_dict = json.load(file)
                cfm_df = pd.DataFrame.from_dict(cfm_dict)
            
            sn.heatmap(cfm_df.T, vmin=0.0, vmax=1.0, cmap="Purples", annot=True, ax=ax)
            ax.set_title(eval_lang)
            ax.set_xlabel("predicted")
            ax.set_ylabel("expected/actual (true labels)")
        
        for i in range(len(files), num_rows * num_cols):
            fig.delaxes(axes[i])
        
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, folder, "cfm.png"), bbox_inches="tight")
        print(f"confusion matrices for {model} trained on {train_lang} saved")



def plot_tve_matrix(ax : matplotlib.axes.Axes,
                model : str, 
                metric : str,
                train_langs : list[str], # = ["en", "fr", "fi", "id", "sv", "tr"],
                eval_langs : list[str], # = ["en", "fr", "fi", "id", "sv", "tr", "al"],
                ) -> None:    
    df = pd.DataFrame(index=eval_langs, columns=train_langs).astype(float)
    for train_lang in train_langs:
        with open(f"output/{model}-{train_lang}/eval.json", "r") as file:
            metric_summary = json.load(file)
        
        for eval_lang, metrics in metric_summary.items():
            df.at[eval_lang, train_lang] = metrics[f"{metric}"]

    sn.heatmap(df, vmin=0.0, vmax=1.0, cmap="Purples", annot=True, cbar=False, ax=ax)
    ax.set_title(f"{model} {metric}")
    ax.set_xlabel("train lang")
    ax.set_ylabel("eval lang")


def get_all_train_eval_langs(model : str, 
              base_dir : str = "output"
              ) -> tuple[list[str], list[str]]:
    train_langs = []
    eval_langs = set()
    
    for folder in os.listdir(base_dir):
        if not os.path.isdir(os.path.join(base_dir, folder)):
            continue
        
        parts = folder.split("-")
        if len(parts) == 2 and parts[0] == model:
            lang = parts[1]
            train_langs.append(lang)
            
            json_path = os.path.join(base_dir, folder, "eval.json")
            with open(json_path, "r") as file:
                metrics = json.load(file)
                eval_langs.update(list(metrics.keys()))

    only_eval_langs = list(eval_langs - set(train_langs))
    train_langs = sorted(train_langs)
    eval_langs = train_langs + sorted(only_eval_langs)
    
    return train_langs, eval_langs


def tve_matrices(models : list[str] = ["mbert", "xlmr", "glot500"],
                 metrics : list[str] = ["micro_f1", "macro_f1"]
                 ) -> None:
    fig, axes = plt.subplots(nrows=len(metrics), 
                             ncols=len(models),
                             figsize=(4 * len(models), 5 * len(metrics)))
    fig.suptitle("train vs. eval lang", fontsize=20)

    for model_ind, model in enumerate(models):
        train_langs, eval_langs = get_all_train_eval_langs(model)
        for metr_ind, metric in enumerate(metrics):
            plot_tve_matrix(axes[metr_ind, model_ind], model, metric, train_langs, eval_langs)
            print(f"tve matrix for {model} on metric {metric} plotted")

    fig.tight_layout()
    fig.savefig("output/tve.png", bbox_inches="tight")
    print("tve matrices saved")



def main():
    confusion_matrices()
    print()
    tve_matrices()

if __name__ == "__main__":
    main()