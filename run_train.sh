#! /bin/bash
#SBATCH --job-name=finetune_reg_class
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --constraint=gputc
#SBATCH --time 00:59:00
#SBATCH --mail-type=END
#SBATCH --mail-user=harbison@unistra.fr
#SBATCH --error=output/error.txt
#SBATCH --output=output/output.txt

TRANSFORMERS_OFFLINE=1
module load python/python-3.11.3
source /home2020/home/lilpa/harbison/experiences/env/bin/activate

# run python script here
echo "start"
python finetune.py --model mbert --train_langs fr
echo "end"