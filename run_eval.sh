#! /bin/bash
#SBATCH --job-name=evaluate_reg_class
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time 00:59:00
#SBATCH --mail-type=END
#SBATCH --mail-user=harbison@unistra.fr
#SBATCH --error=output/error.txt
#SBATCH --output=output/output.txt

module load python/python-3.11.3
source /home2020/home/lilpa/harbison/experiences/env/bin/activate

model=mbert
train_langs=fr
eval_langs=("fr" "sw")

echo "evaluating $model trained on $train_langs"
for lang in "${eval_langs[@]}"
do
    echo "evaluating on $lang"
    time python evaluate.py --model $model --train_langs $train_langs --eval_lang $lang
    echo "$lang complete"
done
echo "all evaluations complete"

# echo "evaluating untrained $model"
# for lang in "${eval_langs[@]}"
# do
#     echo "evaluating on $lang"
#     time python evaluate.py --model $model --eval_lang $lang
#     echo "$lang complete"
# done
# echo "all evaluations complete"