#! /bin/bash
#SBATCH --job-name=finetune_reg_class
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
lang=fr

echo "finetuning $model on $lang"
python finetune.py --model $model --train_langs $lang
echo "finetuning complete"