#! /bin/bash

models=("mbert" "xlmr" "glot500")
langs=("distr" "fi-sv")
subfolder=""

for model in "${models[@]}"; do
    for lang in "${langs[@]}"; do
        if [ -z "${subfolder}" ]; then
            output_dir="output/$model-$lang"
        else
            output_dir="output/$subfolder/$model-$lang"
        fi
        mkdir -p $output_dir
        echo "submitting evaluation job for $model on $lang"
        sbatch --job-name="evaluate-${model}-${lang}" --output="$output_dir/eval.txt" run_eval.sh $model $lang $subfolder
    done
done
echo "all jobs submitted"