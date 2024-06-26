#! /bin/bash
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time 01:59:00

module load python/python-3.11.3
source /home2020/home/lilpa/harbison/experiences/env/bin/activate

model=$1
lang=$2

echo
echo "finetuning $model on $lang"
srun python finetune.py --model $model --train_langs $lang
echo "finetuning $model on $lang complete"