#! /bin/bash

models=("glot500")
langs=("en" "fi" "fr" "sv" "id" "tr")

for model in "${models[@]}"; do
    for lang in "${langs[@]}"; do
        output_dir="output/$model/$lang"
        mkdir -p $output_dir
        sbatch --job-name="finetune_${model}_${lang}" --error="$output_dir/train_err.txt" --output="$output_dir/train_out.txt" run_train.sh $model $lang
    done
done
echo "all jobs submitted"