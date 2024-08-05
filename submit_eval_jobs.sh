#! /bin/bash

models=("mbert" "xlmr" "glot500")
langs=("distr" "fi-sv")

for model in "${models[@]}"; do
    for lang in "${langs[@]}"; do
        output_dir="output/$model-$lang"
        mkdir -p $output_dir
        echo "submitting evaluation job for $model on $lang"
        sbatch --job-name="evaluate-${model}-${lang}" --output="$output_dir/eval.txt" run_eval.sh $model $lang
    done
done
echo "all jobs submitted"