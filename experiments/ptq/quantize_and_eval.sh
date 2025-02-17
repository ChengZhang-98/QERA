function check_rval {
    if [[ $? -ne 0 ]]; then
        echo "❌ $1 failed."
        exit 1
    else
        echo "✅ $1 succeeded."
    fi
}

cd ../../
check_rval "cd to project root"

scaling_mode="diag"
model_name=TinyLlama/TinyLlama_v1.1 # change to other checkpoints like meta-llama/Llama-3.1-8B
model_name_esc=$(echo $model_name | sed 's/\//-/g')

if [[ $scaling_mode == "w-only" ]]; then
    other_args="--disable-qera"
else
    other_args="--qera-scaling-mode $scaling_mode"
fi

python ptq_pipeline.py experiments/configs/w-only-uniform-rank.yaml \
    --model-name $model_name \
    --perplexity-eval-batch-size 2 \
    --disable-perplexity-eval --lm-eval-num-fewshot 0 --lm-eval-batch-size auto \
    --max-position-embeddings 2048 \
    $other_args \
    --output-dir ./checkpoints/ptq/${scaling_mode}/${model_name_esc} \
    --num-calibration-samples 128
