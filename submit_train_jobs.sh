#! /bin/bash

models=("mbert" "xlmr" "glot500")
langs=("en" "fr" "fi" "sv" "tr")

for model in "${models[@]}"; do
    for lang in "${langs[@]}"; do
        output_dir="output/$model-$lang"
        mkdir -p $output_dir
        echo "submitting finetuning job for $model on $lang"
        sbatch --job-name="finetune-${model}-${lang}" --output="$output_dir/train.txt" run_train.sh $model $lang
    done
done
echo "all jobs submitted"