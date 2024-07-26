#! /bin/bash
#SBATCH --job-name=evaluate_register
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time 01:59:00
#SBATCH --output=output/eval.txt

source load_env.sh

models=("mbert" "xlmr" "glot500")
train_langs=("en" "fi" "fr" "sv" "id" "tr")
eval_langs=("ar" "ca" "es" "fa" "hi" "jp" "no" "pt" "ur" "zh")

for model in "${models[@]}"; do
    for train_lang in "${train_langs[@]}"; do
        echo "evaluating $model trained on $train_lang"
        for eval_lang in "${eval_langs[@]}"; do
            echo "evaluating on $eval_lang"
            srun python evaluate.py --model $model --train_langs $train_lang --eval_lang $eval_lang
            echo "$eval_lang complete"
            echo
        done
    done
done
echo "all evaluations complete"