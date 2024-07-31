#! /bin/bash
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time 00:29:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=harbison@unistra.fr

set -e # if python script fails, bash script fails

source load_env.sh

model=$1
train_lang=$2
eval_langs=("en" "fi" "fr" "sv" "ar" "ca" "es" "fa" "hi" "id" "jp" "no" "pt" "ru" "tr" "ur" "zh" "al" "de")

echo "evaluating $model trained on $train_lang"
for eval_lang in "${eval_langs[@]}"; do
    echo "evaluating on $eval_lang"
    srun python evaluate.py --model $model --train_langs $train_lang --eval_lang $eval_lang
    echo "$eval_lang complete"
    echo
done
echo "all evaluations complete"