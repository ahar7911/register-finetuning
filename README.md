# Fine-tuning Multilingual LLMs for Textual Register Classification
[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/ahar7911/register-finetuning/blob/master/README.md)
[![fr](https://img.shields.io/badge/lang-fr-blue.svg)](https://github.com/ahar7911/register-finetuning/blob/master/README.fr.md)

This repository contains scripts to fine-tune pre-trained multilingual large language models (LLMs) from HuggingFace on a classification task of textual register. Models are fine-tuned to classify a text's register in one or more languages, and then evaluated on their classification ability in languages that they were and were not trained on. Included code generates confusion matrices of the fine-tuned model's ability to classify textual register per language, and matrices that compare fine-tuned models' performances in different languages based on the language(s) they were trained on.

All bash scripts were designed to submit jobs using Slurm on the [University of Strasbourg's Computing Center (CCUS)](https://hpc.pages.unistra.fr/).

The register classification system is assumed to be that of the [`register-corpus` repository](https://github.com/ahar7911/register-corpus) (or at least the structure of said repository is assumed; another classification system could probably be used if the file structure is retained). The code assumes that this repository and the `register-corpus` repository are located within the same folder. If not, `CORPUS_PATH` in `utils/corpus_load.py` should be updated with the correct relative filepath.

### `utils` folder

This folder contains the necessary helper functions to load the corpus into a PyTorch `Dataset` (`corpus_load.py`), calculate and save metrics during fine-tuning and evaluation (`metrics.py`), and calculate confusion matrices (`cfm.py`) and train lang vs. eval lang performance matrices (`tve.py`) after the models have been "evaluated".

The models used for finetuning, their associated abbreviations used within this code, and their HuggingFace checkpoints are saved in `model2chckpt.json`. Additional HuggingFace models could be added here for training; the abbreviations used here could be added to the `models` in `submit_train_jobs.sh`/`finetune.py` and `submit_eval_jobs.sh`/`evaluate.py`. (The `choices` for the `model_name` command line argument for both Python scripts will also have to be edited).

A Python script `download_model.py` downloads all models in `model2chckpt.json` for offline fine-tuning and evaluation.

## Model fine-tuning

To fine-tune the models on the computing cluster, the user should run `submit_train_jobs.sh`, updating the `models`, `langs`, or `subfolder` variables at the top as needed:
 - `models`: which models to fine-tune; abbreviations should line up with those used in `model2chckpt.json`
 - `langs`: which languages to fine-tune the model on; the abbreviations need to line up with those used in `register-corpus/info/lang2name.json`
 - `subfolder`: if specified, models will be saved to `models/subfolder/` and training metric outputs will be saved to `outputs/subfolder`; use for experiments with different constraints

All possible fine-tuning combinations of models and languages will be used.

`submit_train_jobs.sh` uses Slurm to submit each model's fine-tuning as a separate job on the computing cluster, using `run_train.sh` which calls `finetune.py` with the appropriate arguments. `run_train.sh` can be modified to change the job parameters (e.g. number of GPUs, mail address to notify in case of job failure, etc.).


Alternatively, a single finetuning job can be submitted by running the Python script `finetune.py`. Run `python finetune.py --help` for more information about the command line arguments. `model-name` and `train-langs` are the two required arguments, mimicking the functionality of the `models` and `langs` arguments mentioned above.


All outputs will be saved to the `outputs` folder and all fine-tuned models will be saved to the `models` folder (unless a subfolder is specified, see above).

## Model evaluation

To evaluate the fine-tuned models (saved in the `models` folder) on the computing cluster, the user should run `submit_eval_jobs.sh`, updating the `models`, `langs`, or `subfolder` variables at the top as needed:
 - `models`: which fine-tuned models should be evaluated; abbreviations should line up with those used in `model2chckpt.json`
 - `langs`: which languages the models have been fine-tuned on; the abbreviations need to line up with those used in `register-corpus/info/lang2name.json`
 - `subfolder`: if specified, the outputs of evaluation metrics will be saved to `outputs/subfolder`; use for experiments with different constraints

In `run_eval.sh`, an additional argument `eval_langs` contains a **modifiable** list of languages that the fine-tuned models will be evaluated on.

All possible combinations of models, fine-tuning language(s), and evaluation language(s) will be used.

`submit_eval_jobs.sh` uses Slurm to submit each evaluation as a separate job on the computing cluster, using `run_eval.sh` which calls `evaluate.py` with the appropriate arguments. 


Alternatively, a single evaluation job can be submitted by running the Python script `evaluate.py`. Run `python evaluate.py --help` for more information about the command line arguments. `model-name`, `train-langs`, and `eval-lang` are the three required argument, which respectively mimick the functionality of the `models`, `langs`, and `eval_langs` arguments mentioned above.


All evaluation outputs will be saved to the `outputs` folder (unless a subfolder is specified, see above).

## Post-model evaluation

After evaluation of the models has been run, the saved statistics can be used to generate confusion matrices and train lang vs. eval lang performance matrices using `post_eval.py`. By default, you can run `python post_eval.py` to generate confusion and train vs. eval performance matrices for the saved statistics of all evaluated models. Alternatively, you can not generate either the confusion or the train vs. eval performance matrices, or you can specify which models, train and eval langs to have the matrices generated for. More customization is available including specifying the folder in which the statistics are contained. Run `python post_eval.py --help` for more information.