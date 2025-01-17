# Réglage fin des grands modèles de langue multilingues pour la classification de registres de textes
[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/ahar7911/register-finetuning/blob/master/README.md)
[![fr](https://img.shields.io/badge/lang-fr-blue.svg)](https://github.com/ahar7911/register-finetuning/blob/master/README.fr.md)

Ce dépôt Git contient des scripts pour le réglage fin/l'ajustement des grands modèles de langue (LLMs) multilingues pré-entraînés, tirés de la plate-forme HuggingFace, pour une tâche de classification de registre de texte. Les modèles pré-entraînés sont « ajustés » afin qu'ils puissent classer le registre d'un texte dans une ou plusieurs langues, et puis ils sont évalués sur leur capacité de classification dans des langues sur lesquelles ils n'ont pas été « entraînés ». Ce dépôt comprend des scripts pour la génération de matrices de confusion concernant la capacité du modèle ajusté à classer le registre d'un texte par langue, et des matrices qui comparent les performances des modèles ajustés dans différentes langues, en fonction de la ou des langue(s) sur lesquelles ils ont été entrainés.

Tous les scripts bash ont été écrits pour soumettre des jobs à l'aide de Slurm sur [le Centre de Calcul de l'Université de Strasbourg (CCUS)](https://hpc.pages.unistra.fr/).

Le système de classification de registre est supposé être celui du dépôt [`register-corpus`](https://github.com/ahar7911/register-corpus) (ou du moins la structure de ce dépôt est supposée; un autre système de classification pourrait probablement être utilisé si la structure des fichiers est maintenue). Ce code part du principe que ce dépôt et le dépôt `register-corpus`se situent dans le même dossier. Sinon, `CORPUS_PATH` dans le fichier `utils/corpus_load.py` doit être mis à jour avec le chemin d'accès relatif correct.

## `utils` dossier

Ce dossier comprend les fonctions utilitaires nécessaires pour charger le corpus dans un PyTorch `Dataset` (`corpus_load.py`), pour calculer et enregistrer les mesures lors de l'entraînement et de l'évaluation (`metrics.py`), et, après « l'évaluation » des modèles, pour générer les matrices de confusion et les matrices comparant les différences de performance des modèles dans les langues d'évaluation en fonction des langues d'entraînement.

Les modèles utilisés pour le réglage fin, leur abréviations associées utilisées dans ce code, et leur checkpoints HuggingFace sont enregistrés dans le fichier `model2chckpt.json`. D'autres modèles HuggingFace y pourraient être ajoutés pour le réglage fin ; les abréviations utilisées pourraient être ajoutées à la liste `models` dans le script `submit_train_jobs.sh` et `submit_eval_jobs.sh`. (La liste `choices` pour le paramètre de ligne de commande `model_name` dans les scripts python `finetune.py` et `evaluate.py` devra également être modifiée).

Le script python `download_model.py` télécharge tous les modèles énumérés dans le fichier `model2chckpt.json` pour le réglage fin et l'évaluation hors ligne.

## Réglage fin des modèles

Pour faire le réglage fin des modèles sur le centre de calcul, l'utilisateur doit exécuter le script `submit_train_jobs.sh`, en mettant à jour les paramètres `models`, `langs`, ou `subfolder` en haut au besoin:
 - `models`: quels modèles à entraîner/ajuster; les abréviations doit correspondre avec celles utilisées dans le fichier `utils/model2chckpt.json`
 - `langs`: quelles langues sur lesquelles entraîner/ajuster les modèles; les abréviations doit correspondre avec celles utilisées dans le fichier `register-corpus/info/lang2name.json`
 - `subfolder`: si ce paramètre n'est pas vide, les modèles seront enregistrés dans le dossier `models/subfolder/` et les résultats de mesures d'entraînement seront enregistrés dans le dossier `outputs/subfolder` ; ce paramètre serait utile pour des expériences soumises à des contraintes différentes.

Toutes les combinations possibles pour le réglage fin des modèles énumérés sur les langues énumérées seront utilisées.

Le script `submit_train_jobs.sh` soumet le réglage fin de chaque modèle en tant que job unique au centre de calcul, en utilisant le script `run_train.sh` qui appele le script python `finetune.py` à l'aide de Slurm avec les paramètres d'entrée appropriés. Le script `run_train.sh` peut être modifié pour changer les paramètres du job (e.g. la quantité des GPUs, l'adresse courriel à informer en cas de l'échec du job, etc.).

Ou sinon, l'utilisateur peut effectuer le réglage fin d'un modèle unique sur une ou plusieurs langues en exécutant le script Python `finetune.py`. Exécuter `python finetune.py --help` pour plus d'informations sur les paramètres de ligne de commande obligatoires et optionnels. `model-name` et `train-langs` sont les deux paramètres d'entrée obligatoires, qui imitent la fonctionnalité des paramètres `models` et `langs` mentionnés ci-dessus.

Toutes les données de sortie seront enregistrées dans le dossier `outputs` et tous les modèles entraînés seront enregistrés dans le dossier `models` (à moins qu'un sous-dossier soit specifié, voir ci-dessus).

## Évaluation des modèles

Pour évaluer les modèles ajustés (enregistrés dans le dossier `models`) sur le centre de calcul, l'utilisateur doit exécuter le script `submit_eval_jobs.sh`, en mettant à jour les paramètres `models`, `langs`, ou `subfolder` en haut au besoin:
 - `models`: quels modèles ajustés à évaluer; les abréviations doit correspondre avec celles utilisées dans le fichier `utils/model2chckpt.json`
 - `langs`: quelles langues sur lesquelles les modèles ont été ajustés; les abréviations doit correspondre avec celles utilisées dans le fichier `register-corpus/info/lang2name.json`
 - `subfolder`: si ce paramètre n'est pas vide, les résulats des mesures d'évaluation seront enregistrés dans le dossier `outputs/subfolder/` ; ce paramètre serait utile pour des expériences soumises à des contraintes différentes.

Dans le script `run_eval.sh`, un autre paramètre `eval_langs` comprend une liste **modifiable** des langues sur lesquelles **tous** les modèles ajustés seront évalués.

Toutes les combinations possibles des modèles, des langues d'entraînement, et des langues d'évaluation seront utilisées.

Le script `submit_eval_jobs.sh` soumet l'évaluation de chaque modèle en tant que job unique au centre de calcul, en utilisant le script `run_eval.sh` qui appele le script python `evaluate.py` à l'aide de Slurm avec les paramètres d'entrée appropriés. Le script `run_eval.sh` peut être modifié pour changer les paramètres du job (e.g. la quantité des GPUs, l'adresse courriel à informer en cas de l'échec du job, etc.).

Ou sinon, l'utilisateur peut effectuer l'évaluation d'un modèle ajusté spécifique sur une langue d'évaluation en exécutant le script Python `evaluate.py`. Exécuter `python evaluate.py --help` pour plus d'informations sur les paramètres de ligne de commande obligatoires et optionnels. `model-name`, `train-langs`, et `eval-lang` sont les trois paramètres d'entrée obligatoires, qui imitent respectivement la fonctionnalité des paramètres `models`, `langs`, et `eval_langs` mentionnés ci-dessus.

Toutes les données de sortie de l'évaluation seront enregistrées dans le dossier `outputs` (à moins qu'un sous-dossier soit specifié, voir ci-dessus).

## Après l'évaluation des modèles

Après l'évaluation des modèles, les mesures enregistrées peuvent être utilisées pour générer des matrices de confusion et des matrices de comparaison de performance des modèles en éxecutant le script python `post_eval.py`. Par défaut, l'utilisateur peut lancer `python post_eval.py` dans l'interface de ligne de commande pour générer les deux genres de matrices à partir des mesures enregistrées de tous les modèles évalués dans le dossier `outputs`. Ou sinon, l'utilisateur peut choisir de ne pas générer soit les matrices de confusion soit les matrices de comparaison de performance, ou l'utilisateur peut spécifier quels modèles, quelles langues d'entraînement ou quelles langues d'évaluation pour lesquel(le)s générer des matrices. D'autres personnalisations sont possibles, y compris la spécification du dossier dans lequel les mesures sont enregistrées. L'utilisateur peut lancer `python post_eval.py --help` pour plus d'informations.