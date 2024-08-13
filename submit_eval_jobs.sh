#! /bin/bash

models=("mbert" "xlmr" "glot500")
langs=("distr")
#("en" "fi" "fr" "sv" "en-fi" "en-fr" "en-sv" "fi-fr" "fi-sv" "fr-sv" "id" "ru" "tr" "de")
subfolder="bl"

for model in "${models[@]}"; do
    for lang in "${langs[@]}"; do
        if [ -z "${subfolder}" ]; then
            output_dir="output/$model-$lang"
        else
            output_dir="output/$subfolder/$model-$lang"
        fi
        mkdir -p $output_dir
        echo "submitting evaluation job for $model on $lang"
        sbatch --job-name="evaluate_${model}_${lang}" --output="$output_dir/eval.txt" run_eval.sh $model $lang $subfolder
    done
done
echo "all jobs submitted"