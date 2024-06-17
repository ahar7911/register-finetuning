#! /bin/bash
#SBATCH --job-name=finetune_reg_class
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time 00:59:00
#SBATCH --error=output/err_train.txt
#SBATCH --output=output/out_train.txt

module load python/python-3.11.3
source /home2020/home/lilpa/harbison/experiences/env/bin/activate

model=mbert
lang=sw

echo "finetuning $model on $lang"
time python finetune.py --model $model --train_langs $lang
echo "finetuning complete"