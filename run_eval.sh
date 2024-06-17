#! /bin/bash
#SBATCH --job-name=finetune_reg_class
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --time 00:59:00
#SBATCH --mail-type=END
#SBATCH --mail-user=harbison@unistra.fr
#SBATCH --error=output/error.txt
#SBATCH --output=output/output.txt

HF_HUB_OFFLINE=1
module load python/python-3.11.3
source /home2020/home/lilpa/harbison/experiences/env/bin/activate

model=mbert
train_langs=fr
eval_langs=("fr" "sw" "fi")

echo "evaluating $model train on $train_lang"
for lang in $langs;
do
    echo "evaluating on $lang"
    python evaluate.py --model $model --train_langs $train_langs --eval_lang $lang
    echo "evaluation complete"
done