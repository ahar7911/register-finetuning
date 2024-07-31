from argparse import ArgumentParser
import sys
from pathlib import Path
from typing import Callable
import inspect
import json
import math

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn 

from utils.corpus_load import CORPUS_PATH

base_dir = Path("output")

def plot_summary(ax : matplotlib.axes.Axes, train_lang : str) -> None:
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
    ax.bar(*zip(*train_lang_summary.items()))
    ax.set_title(f"number of texts per register in {train_lang} corpus")
    ax.set_xlabel("register")
    ax.set_ylabel("number of texts")


def plot_cfm(ax : matplotlib.axes.Axes, cfm_path : Path, eval_lang : str) -> None:
    with open(cfm_path, "r") as file:
        cfm_dict = json.load(file)
        cfm_df = pd.DataFrame.from_dict(cfm_dict)
    
    # if len(list(cfm_dict.keys())[0].split("\n")) != 1:
    #     cfm_df = cfm_df.T
    #     new_cfm_dict = cfm_df.to_dict()
    #     with open(cfm_path, "w") as file:
    #         json.dump(new_cfm_dict, file, indent=4)
                    
    sn.heatmap(cfm_df, vmin=0.0, vmax=1.0, cmap="Purples", annot=True, ax=ax)
    ax.set_title(eval_lang)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expected/actual (true labels)")


def confusion_matrices(ignore_summary : bool = False,
                       models : list[str] = None,
                       train_langs : list[str] = None,
                       eval_langs : list[str] = None
                       ) -> None:
    for model_folder in base_dir.iterdir():
        if not model_folder.is_dir():
            continue

        cfm_folder = model_folder / "cfm"
        model, train_lang = model_folder.name.split("-", maxsplit=1)
        if (models is not None and model not in models) or (train_langs is not None and train_lang not in train_langs):
            continue

        cfm_paths = []
        for path in cfm_folder.iterdir():
            if path.is_file() and path.suffix == ".json":
                cfm_paths.append(path)
        cfm_paths = sorted(cfm_paths, key=lambda path: path.name)

        # move cfm of evaluation on the same language as the training language to the front
        same_lang_path = cfm_folder / f"{train_lang}.json"
        if same_lang_path in cfm_paths:
            cfm_paths.remove(same_lang_path)
            cfm_paths.insert(0, same_lang_path)

        cfm_start_ax = 0 if ignore_summary else 1
        num_plots = len(cfm_paths) + cfm_start_ax # 1 more if we are plotting the corpus summary statistics
        num_cols = math.ceil(math.sqrt(num_plots))
        num_rows = math.ceil(num_plots / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows))
        fig.suptitle(f"confusion matrices for {model} trained on {train_lang}", fontsize=20)
        axes = axes.flatten()

        if not ignore_summary:
            plot_summary(axes[0], train_lang)

        # plotting cfms
        for ax, cfm_path in zip(axes[cfm_start_ax:], cfm_paths):
            eval_lang = cfm_path.stem
            if eval_langs is not None and eval_lang not in eval_langs:
                continue
            plot_cfm(ax, cfm_path, eval_lang)
        
        # deleting unused axes
        for i in range(len(cfm_paths) + cfm_start_ax, num_rows * num_cols):
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


def tve_matrices(models : list[str] = ["mbert", "xlmr", "glot500"],
                 metrics : list[str] = ["micro_f1", "macro_f1"],
                 train_langs : list[str] = None,
                 eval_langs : list[str] = None
                 ) -> None:
    fig, axes = plt.subplots(nrows=len(metrics), 
                             ncols=len(models),
                             figsize=(10 * len(models), 4 * len(metrics)))
    fig.suptitle("train vs. eval lang", fontsize=20)

    for model_ind, model in enumerate(models):
        default_train_langs, default_eval_langs = get_all_train_eval_langs(model)
        if train_langs is None:
            model_train_langs = default_train_langs
        elif len(set(train_langs) - set(default_train_langs)) == 0:
            print(f"some tve train langs specified ({' '.join(train_langs)}) do not exist for specified or default models", file=sys.stderr)
            print(f"languages that do exist are: {' '.join(default_train_langs)}", file=sys.stderr)
            sys.exit(1)
        else:
            model_train_langs = train_langs
        
        if eval_langs is None:
            model_eval_langs = default_eval_langs
        elif len(set(eval_langs) - set(default_eval_langs)) == 0:
            print(f"some tve eval langs specified ({' '.join(eval_langs)})do not exist for specified or default models", file=sys.stderr)
            print(f"languages that do exist are: {' '.join(default_eval_langs)}", file=sys.stderr)
            sys.exit(1)
        else:
            model_eval_langs = eval_langs
        

        for metr_ind, metric in enumerate(metrics):
            plot_tve_matrix(axes[metr_ind, model_ind], model, metric, model_train_langs, model_eval_langs)
            print(f"tve matrix for {model} on metric {metric} plotted")

    fig.tight_layout()
    fig.savefig(base_dir / "tve.png", bbox_inches="tight")
    print("tve matrices saved")


def get_argument_names(func : Callable) -> list[str]:
    signature = inspect.signature(func)
    return [param.name for param in signature.parameters.values()]


def main(args):
    args_dict = vars(args)
    if not args.ignore_cfm:
        cfm_arg_names = get_argument_names(confusion_matrices)
        cfm_args = {}
        for arg_name, value in args_dict.items():
            cfm_arg_name = arg_name.replace("cfm_", "")
            if cfm_arg_name != arg_name and cfm_arg_name in cfm_arg_names and value is not None:
                cfm_args[cfm_arg_name] = value
        
        confusion_matrices(**cfm_args)
        print()
    if not args.ignore_tve:
        tve_arg_names = get_argument_names(tve_matrices)
        tve_args = {}
        for arg_name, value in args_dict.items():
            tve_arg_name = arg_name.replace("tve_", "")
            if tve_arg_name != arg_name and tve_arg_name in tve_arg_names and value is not None:
                tve_args[tve_arg_name] = value
        
        tve_matrices(**tve_args)

if __name__ == "__main__":
    parser = ArgumentParser(prog="Post-evaluation analysis",
                            description="Plots confusion matrices for each model, train lang, and eval lang combo, as well as a train vs. evaluation language (tve) plot of metrics")
    parser.add_argument("--ignore_cfm", action="store_true",
                        help="Will not create confusion matrices (no matter what other arguments for confusion matrices are provided)")
    parser.add_argument("--cfm_ignore_summary", action="store_true",
                        help="Whether to plot summary statistics of the train language as the first plot of the confusion matrices")
    parser.add_argument("--cfm_models", nargs="+", choices=["mbert", "xlmr", "glot500"],
                        help="Models to calculate confusion matrices for")
    parser.add_argument("--cfm_train_langs", nargs="+",
                        help="Train languages to calculate confusion matrices for")
    parser.add_argument("--cfm_eval_langs", nargs="+",
                        help="Evaluation languages to calculate confusion matrices for")

    parser.add_argument("--ignore_tve", action="store_true",
                        help="Will not create train vs. eval lang metric (tve) matrix (not matter what other arguments for tve matrix)")
    parser.add_argument("--tve_models", nargs="+", choices=["mbert", "xlmr", "glot500"],
                        help="Models to include in train vs. eval lang metric matrix")
    parser.add_argument("--tve_metrics", nargs="+", choices=["micro_f1", "macro_f1"],
                        help="Metrics to include in train vs. eval lang metric matrix")
    parser.add_argument("--tve_train_langs", nargs="+",
                        help="Train languages to include in train vs. eval lang metric matrix")
    parser.add_argument("--tve_eval_langs", nargs="+",
                        help="Evaluation languages to include in train vs. eval lang metric matrix")

    args = parser.parse_args()
    if not base_dir.exists() or not base_dir.is_dir() or not any(base_dir.iterdir()):
        print(f"current filepath to output directory ({base_dir}) does not exist, is not a directory, or is empty", file=sys.stderr)
        print("run evaluate.py on a finetuned model and an evaluation language, or run run_eval.sh", file=sys.stderr)
        sys.exit(1)
    main(args)