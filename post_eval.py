import sys
from pathlib import Path
import json
import math

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn 

from utils.corpus_load import CORPUS_PATH

base_dir = Path("output")

def confusion_matrices():
    for model_folder in base_dir.iterdir():
        if not model_folder.is_dir():
            continue

        cfm_folder = model_folder / "cfm"
        parts = model_folder.name.split("-")
        if len(parts) != 2:
            print(f"subfolder {model_folder.name} of base directory {base_dir} is not of the generated type MODEL-TRAIN_LANG (did you manually add this folder?)", file=sys.stderr)
            continue
        model, train_lang = parts

        cfm_paths = []
        for path in cfm_folder.iterdir():
            if path.is_file() and path.suffix == ".json":
                cfm_paths.append(path)

        # move cfm of evaluation on the same language as the training language to the front
        same_lang_path = cfm_folder / f"{train_lang}.json"
        if same_lang_path in cfm_paths:
            cfm_paths.remove(same_lang_path)
            cfm_paths.insert(0, same_lang_path)

        num_plots = len(cfm_paths) + 1 # 1 more since we are plotting the corpus summary statistics
        num_cols = math.ceil(math.sqrt(num_plots))
        num_rows = math.ceil(num_plots / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows))
        fig.suptitle(f"confusion matrices for {model} trained on {train_lang}", fontsize=20)
        axes = axes.flatten()

        # obtaining corpus summary statistics
        summary_path = CORPUS_PATH / f"summaries/{train_lang}.json"
        if summary_path.exists():
            with open(summary_path) as summary_file:
                train_lang_summary = json.load(summary_file)["counts"]
        else:
            print(f"invalid path to summary json file {summary_path} for lang {train_lang}", file=sys.stderr)
            print(f"check if CORPUS_PATH in utils/corpus_load.py correctly directs to register-corpus directory, or rerun analyze_dist.sh in register-corpus", file=sys.stderr)
            sys.exit(1)
        
        # plotting corpus summary statistics
        axes[0].bar(*zip(*train_lang_summary.items()))
        axes[0].set_title(f"number of texts per register in {train_lang} corpus")
        axes[0].set_xlabel("register")
        axes[0].set_ylabel("number of texts")

        # plotting cfms
        for ax, cfm_path in zip(axes[1:], cfm_paths):
            eval_lang = cfm_path.stem
            with open(cfm_path, "r") as file:
                cfm_dict = json.load(file)
                cfm_df = pd.DataFrame.from_dict(cfm_dict)
                            
            sn.heatmap(cfm_df, vmin=0.0, vmax=1.0, cmap="Purples", annot=True, ax=ax)
            ax.set_title(eval_lang)
            ax.set_xlabel("predicted")
            ax.set_ylabel("expected/actual (true labels)")
        
        # deleting unused axes
        for i in range(len(cfm_paths) + 1, num_rows * num_cols):
            fig.delaxes(axes[i])
        
        fig.tight_layout()
        fig.savefig(model_folder / "cfm.png", bbox_inches="tight")
        print(f"confusion matrices for {model} trained on {train_lang} saved")


def plot_tve_matrix(ax : matplotlib.axes.Axes,
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


def get_all_train_eval_langs(model : str) -> tuple[list[str], list[str]]:
    train_langs = []
    eval_langs = set()
    
    for model_folder in base_dir.iterdir():
        if not model_folder.is_dir():
            continue
        
        parts = model_folder.name.split("-")
        if len(parts) == 2 and parts[0] == model:
            train_langs.append(parts[1])
            
            json_path = model_folder / "eval.json"
            with open(json_path, "r") as file:
                metrics = json.load(file)
                eval_langs.update(list(metrics.keys()))

    # order langs so that diagonal forms of same train and eval lang
    only_eval_langs = list(eval_langs - set(train_langs))
    train_langs = sorted(train_langs)
    eval_langs = train_langs + sorted(only_eval_langs)
    
    return train_langs, eval_langs


def tve_matrices(models : list[str] = ["mbert", "xlmr", "glot500"],
                 metrics : list[str] = ["micro_f1", "macro_f1"]
                 ) -> None:
    fig, axes = plt.subplots(nrows=len(metrics), 
                             ncols=len(models),
                             figsize=(10 * len(models), 4 * len(metrics)))
    fig.suptitle("train vs. eval lang", fontsize=20)

    for model_ind, model in enumerate(models):
        train_langs, eval_langs = get_all_train_eval_langs(model)
        for metr_ind, metric in enumerate(metrics):
            plot_tve_matrix(axes[metr_ind, model_ind], model, metric, train_langs, eval_langs)
            print(f"tve matrix for {model} on metric {metric} plotted")

    fig.tight_layout()
    fig.savefig(base_dir / "tve.png", bbox_inches="tight")
    print("tve matrices saved")


def main():
    confusion_matrices()
    print()
    tve_matrices()

if __name__ == "__main__":
    if not base_dir.exists() or not base_dir.is_dir() or not any(base_dir.iterdir()):
        print(f"current filepath to output directory ({base_dir}) does not exist, is not a directory, or is empty", file=sys.stderr)
        print("run evaluate.py on a finetuned model and an evaluation language, or run run_eval.sh", file=sys.stderr)
        sys.exit(1)
    main()