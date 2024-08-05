from argparse import ArgumentParser
import sys
from pathlib import Path
from typing import Callable, Any
import inspect

from utils.cfm import confusion_matrices
from utils.tve import tve_matrices

base_dir = Path("output")

def get_argument_names(func : Callable) -> list[str]:
    signature = inspect.signature(func)
    return [param.name for param in signature.parameters.values()]

def get_defined_args(all_args : dict[str, Any], func : Callable, lead_str : str) -> dict[str, Any]:
    func_arg_names = get_argument_names(func)
    args = {}
    for arg_name, value in all_args.items():
        new_arg_name = arg_name.replace(lead_str, "")
        if new_arg_name == arg_name:
            continue
        elif new_arg_name in func_arg_names and value is not None:
            args[new_arg_name] = value
    return args

def main(args):
    args_dict = vars(args)

    if not args.ignore_cfm:
        cfm_args = get_defined_args(args_dict, confusion_matrices, "cfm_") 
        cfm_args["base_dir"] = base_dir      
        confusion_matrices(**cfm_args)
        print()
    if not args.ignore_tve:
        tve_args = get_defined_args(args_dict, tve_matrices, "tve_") 
        tve_args["base_dir"] = base_dir       
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