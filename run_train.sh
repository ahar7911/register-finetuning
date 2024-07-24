#! /bin/bash
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time 00:59:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=harbison@unistra.fr

set -e # if python script fails, bash script fails

source load_env.sh

model=$1
lang=$2
[ "$model" == "glot500" ] && batch_size=8 || batch_size=16

echo
echo "finetuning $model on $lang"
srun python finetune.py --model $model --train_langs $lang --balanced --batch_size $batch_size
echo "finetuning $model on $lang complete"