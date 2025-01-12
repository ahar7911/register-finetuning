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
        
        for eval_lang, metrics in metric_summary.items(): # add to dataframe
            tve_df.at[train_lang, eval_lang] = metrics[f"{metric}"]

    sn.heatmap(tve_df, vmin=0.0, vmax=1.0, cmap="Purples", annot=True, cbar=False, ax=ax)
    ax.set_title(f"{model} {metric}")
    ax.set_xlabel("Eval lang")
    ax.set_ylabel("Train lang")


def get_all_train_eval_langs(base_dir : Path, model : str) -> tuple[list[str], list[str]]:
    train_langs = set()
    eval_langs = set()
    
    for model_folder in base_dir.iterdir():
        if not model_folder.is_dir():
            continue
        
        parts = model_folder.name.split("-", maxsplit=1) # of form model-lang1-lang2-...
        if parts[0] == model:
            train_langs.add(parts[1])
            
            json_path = model_folder / "eval.json"
            with open(json_path, "r") as file:
                metrics = json.load(file)
                eval_langs.update(list(metrics.keys())) # keys of eval.json are the eval language abbreviations

    # order langs so that diagonal forms of same train and eval lang, then the rest by alphabetical
    shared_langs = train_langs & eval_langs
    only_train_langs = list(train_langs - shared_langs)
    only_eval_langs = list(eval_langs - shared_langs)

    shared_langs = sorted(list(shared_langs))
    train_langs = shared_langs + sorted(only_train_langs)
    eval_langs = shared_langs + sorted(only_eval_langs)
    
    return train_langs, eval_langs


def check_valid_langs(default_langs : list[str], new_langs : list[str], lang_type : str) -> list[str]:
    if new_langs is None: # no languages were specified to tve_matrices, use default
        return default_langs
    elif len(set(new_langs) - set(default_langs)) == 0: # some languages specified aren't in default
        print(f"Some or all of the specified TVE {lang_type} langs ({' '.join(new_langs)}) do not exist for specified or default models", file=sys.stderr)
        print(f"Languages that do exist are: {' '.join(default_langs)}", file=sys.stderr)
        sys.exit(1)
    else:
        return new_langs


def tve_matrices(base_dir : Path,
                 models : list[str] = ["mbert", "xlmr", "glot500"],
                 metrics : list[str] = ["micro_f1", "macro_f1"],
                 train_langs : list[str] = None,
                 eval_langs : list[str] = None
                 ) -> None:
    # set up figure and axes
    fig, axes = plt.subplots(nrows=len(metrics), 
                             ncols=len(models),
                             figsize=(10 * len(models), 4 * len(metrics)))
    fig.suptitle("Train vs. eval lang", fontsize=20)

    for model_ind, model in enumerate(models):
        # get languages that have been trained and evaluated
        default_train_langs, default_eval_langs = get_all_train_eval_langs(base_dir, model)
        model_train_langs = check_valid_langs(default_train_langs, train_langs, "train")
        model_eval_langs = check_valid_langs(default_eval_langs, eval_langs, "eval")      

        for metr_ind, metric in enumerate(metrics):
            plot_tve_matrix(base_dir, axes[metr_ind, model_ind], model, metric, model_train_langs, model_eval_langs)
            print(f"TVE matrix for {model} on metric {metric} plotted")

    fig.tight_layout()
    fig.savefig(base_dir / "tve.png", bbox_inches="tight")
    plt.close(fig)
    print("TVE matrices saved")