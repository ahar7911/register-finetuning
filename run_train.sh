#! /bin/bash
#SBATCH --job-name=finetune_reg_class
#SBATCH --nodes=1
#SBATCH -p publicgpu
#SBATCH -A lilpa
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
python register_finetune.py --model mbert --eval_lang fr
echo "end"