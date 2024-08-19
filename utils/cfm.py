import sys
from pathlib import Path
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix

from utils.corpus_load import get_texts_regs, REGISTERS


# modified from https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
def save_cfm(preds : np.ndarray, labels : np.ndarray, out_path : Path) -> None:
    cf_matrix = confusion_matrix(labels, preds, labels=range(len(REGISTERS)))
    register_totals = np.sum(cf_matrix, axis=1)
    cf_matrix = np.divide(cf_matrix, register_totals[:, None], where=cf_matrix!=0)

    row_labels = [f"{reg}\n(total:{total})" for reg, total in zip(REGISTERS, register_totals)]
    df_cm = pd.DataFrame(cf_matrix, index=row_labels, columns=REGISTERS)

    if not out_path.parent.exists(): # cfm directory does not exist yet
        out_path.parent.mkdir(parents=True)

    with open(out_path, "w") as file: # overwrites existing cfm matrix, if exists
        json.dump(df_cm.to_dict(), file, indent=4)

#---------------------------------------------------------------------------------------------------

def plot_summary(ax : matplotlib.axes.Axes, train_lang : str) -> None:
    train_lang_tsvs = [Path(f"train/{train_lang}.tsv") for train_lang in train_lang.split("-")]
    registers = []
    for train_lang_tsv in train_lang_tsvs:
        _, train_lang_regs = get_texts_regs(train_lang_tsv)
        registers.extend(train_lang_regs)
    register_counts = Counter(registers)
    
    # plotting corpus summary statistics
    ax.bar(*zip(*register_counts.items()))
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


def confusion_matrices(base_dir : Path,
                       ignore_summary : bool = False,
                       models : list[str] = None,
                       train_langs : list[str] = None,
                       eval_langs : list[str] = None
                       ) -> None:
    for model_folder in base_dir.iterdir():
        if not model_folder.is_dir():
            continue
        
        if (model_folder / "cfm.png").exists():
            print(f"saved cfm already exists in {model_folder}")
            continue

        cfm_folder = model_folder / "cfm"
        model, train_lang = model_folder.name.split("-", maxsplit=1)
        if (models is not None and model not in models) or (train_langs is not None and train_lang not in train_langs):
            continue
        if not cfm_folder.exists():
            print(f"no cfm folder exists in folder {model_folder}")
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
        plt.close(fig)
        print(f"confusion matrices for {model} trained on {train_lang} saved")
    print("all confusion matrices saved")