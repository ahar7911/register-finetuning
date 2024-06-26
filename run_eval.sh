#! /bin/bash
#SBATCH --job-name=evaluate_register
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time 00:59:00
#SBATCH --error=output/err_eval.txt
#SBATCH --output=output/out_eval.txt

module load python/python-3.11.3
source /home2020/home/lilpa/harbison/experiences/env/bin/activate

models=("mbert" "xlm-r" "glot500")
train_langs=("fi" "fr" "sv")
eval_langs=("en" "fi" "fr" "sv")

for model in "${models[@]}"; do
    for train_lang in "${train_langs[@]}"; do
        echo "evaluating $model trained on $train_lang"
        for eval_lang in "${eval_langs[@]}"; do
            echo
            echo "evaluating on $eval_lang"
            mkdir -p "output/$model/$train_lang/eval"
            srun python evaluate.py --model $model --train_langs $train_lang --eval_lang $eval_lang
            echo "$eval_lang complete"
        done
    done
done
echo "all evaluations complete"