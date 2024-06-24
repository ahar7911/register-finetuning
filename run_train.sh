#! /bin/bash
#SBATCH --job-name=finetune_reg_class
#SBATCH -A lilpa
#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time 01:59:00
#SBATCH --error=output/err_train.txt
#SBATCH --output=output/out_train.txt

module load python/python-3.11.3
source /home2020/home/lilpa/harbison/experiences/env/bin/activate

models=("mbert" "xlm-r" "glot500")
langs=("tr" "id")

for model in "${models[@]}"; do
    for lang in "${langs[@]}"; do
        echo "finetuning $model on $lang"
        mkdir -p "output/$model/$lang"
        time python finetune.py --model $model --train_langs $lang
        echo "finetuning $model on $lang complete"
    done
done
echo "finetuning complete"