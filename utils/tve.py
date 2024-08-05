import sys
from pathlib import Path
import json

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn

def plot_tve_matrix(base_dir : Path,
                    ax : matplotlib.axes.Axes,
                    model : str, 
                    metric : str,
                    train_langs : list[str], # = ["en", "fr", "fi", "id", "sv", "tr"],
                    eval_langs : list[str] # = ["en", "fr", "fi", "id", "sv", "tr", "al"],
                    ) -> None:
    # constructing train vs. eval dataframe for specified model and metric   
    tve_df = pd.DataFrame(index=train_langs, columns=eval_langs).astype(float)
    for train_lang in train_langs:
        metrics_path = base_dir / f"{model}-{train_lang}/eval.json"
        # don't test if metrics_path exists since get_all_train_eval_langs finds train langs through searching folders
        with open(metrics_path, "r") as file:
            metric_summary = json.load(file)
        
        for eval_lang, metrics in metric_summary.items():
            tve_df.at[train_lang, eval_lang] = metrics[f"{metric}"]

    sn.heatmap(tve_df, vmin=0.0, vmax=1.0, cmap="Purples", annot=True, cbar=False, ax=ax)
    ax.set_title(f"{model} {metric}")
    ax.set_xlabel("eval lang")
    ax.set_ylabel("train lang")


def get_all_train_eval_langs(base_dir : Path, model : str) -> tuple[list[str], list[str]]:
    train_langs = []
    eval_langs = set()
    
    for model_folder in base_dir.iterdir():
        if not model_folder.is_dir():
            continue
        
        parts = model_folder.name.split("-", maxsplit=1)
        if parts[0] == model:
            train_langs.append(parts[1])
            
            json_path = model_folder / "eval.json"
            with open(json_path, "r") as file:
                metrics = json.load(file)
                eval_langs.update(list(metrics.keys()))

    # order langs so that diagonal forms of same train and eval lang
    train_langs = sorted(train_langs)

    train_langs_in_eval = list(set(train_langs) & eval_langs)
    only_eval_langs = list(eval_langs - set(train_langs))
    eval_langs = sorted(train_langs_in_eval) + sorted(only_eval_langs)
    
    return train_langs, eval_langs

def check_valid_langs(default_langs : list[str], new_langs : list[str], lang_type : str) -> list[str]:
    if new_langs is None:
        return default_langs
    elif len(set(new_langs) - set(default_langs)) == 0:
        print(f"some tve {lang_type} langs specified ({' '.join(new_langs)}) do not exist for specified or default models", file=sys.stderr)
        print(f"languages that do exist are: {' '.join(default_langs)}", file=sys.stderr)
        sys.exit(1)
    else:
        return new_langs


def tve_matrices(base_dir : Path,
                 models : list[str] = ["mbert", "xlmr", "glot500"],
                 metrics : list[str] = ["micro_f1", "macro_f1"],
                 train_langs : list[str] = None,
                 eval_langs : list[str] = None
                 ) -> None:
    fig, axes = plt.subplots(nrows=len(metrics), 
                             ncols=len(models),
                             figsize=(10 * len(models), 4 * len(metrics)))
    fig.suptitle("train vs. eval lang", fontsize=20)

    for model_ind, model in enumerate(models):
        default_train_langs, default_eval_langs = get_all_train_eval_langs(base_dir, model)
        model_train_langs = check_valid_langs(default_train_langs, train_langs, "train")
        model_eval_langs = check_valid_langs(default_eval_langs, eval_langs, "eval")      

        for metr_ind, metric in enumerate(metrics):
            plot_tve_matrix(base_dir, axes[metr_ind, model_ind], model, metric, model_train_langs, model_eval_langs)
            print(f"tve matrix for {model} on metric {metric} plotted")

    fig.tight_layout()
    fig.savefig(base_dir / "tve.png", bbox_inches="tight")
    print("tve matrices saved")