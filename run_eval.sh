#! /bin/bash
#SBATCH --job-name=evaluate_register
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time 00:59:00
#SBATCH --error=output/eval_err.txt
#SBATCH --output=output/eval_out.txt

module load python/python-3.11.3
source /home2020/home/lilpa/harbison/experiences/env/bin/activate

models=("glot500")
train_langs=("en" "fi" "fr" "sv" "id" "tr")
eval_langs=("en" "fi" "fr" "sv" "id" "tr")

for model in "${models[@]}"; do
    for train_lang in "${train_langs[@]}"; do
        echo "evaluating $model trained on $train_lang"
        for eval_lang in "${eval_langs[@]}"; do
            echo "evaluating on $eval_lang"
            mkdir -p "output/$model/$train_lang/eval"
            srun python evaluate.py --model $model --train_langs $train_lang --eval_lang $eval_lang
            echo "$eval_lang complete"
            echo
        done
    done
done
echo "all evaluations complete"