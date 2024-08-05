#! /bin/bash

models=("mbert" "xlmr" "glot500")
langs=("distr" "en-fr" "fi-sv" "fr-sv")
subfolder=""

for model in "${models[@]}"; do
    for lang in "${langs[@]}"; do
        if [ -z "${subfolder}" ]; then
            output_dir="output/$model-$lang"
        else
            output_dir="output/$subfolder/$model-$lang"
        fi
        mkdir -p $output_dir
        echo "submitting finetuning job for $model on $lang"
        sbatch --job-name="finetune-${model}-${lang}" --output="$output_dir/train.txt" run_train.sh $model $lang $subfolder
    done
done
echo "all jobs submitted"