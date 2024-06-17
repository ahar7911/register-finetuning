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

langs=("fr" "sw" "fi")

for lang in $langs
do
    echo "EVALUATING $lang"
    python evaluate.py --model mbert --train_langs fr --eval_lang $lang
    echo "FINETUNING COMPLETE"
done