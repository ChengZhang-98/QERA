function check_rval {
    if [[ $? -ne 0 ]]; then
        echo "❌ $1 failed."
        exit 1
    else
        echo "✅ $1 succeeded."
    fi
}

cd ~/Projects/LoQER/
check_rval "cd ~/Projects/LoQER"

scaling_mode="diag"
model_name=meta-llama/Llama-3.1-8B
model_name_esc=$(echo $model_name | sed 's/\//-/g')

if [[ $scaling_mode == "w-only" ]]; then
    other_args="--disable-loqer"
else
    other_args="--loqer-scaling-mode $scaling_mode"
fi

HF_HUB_OFFLINE=1 python ptq_pipeline.py experiments/configs/w-only-uniform-rank.yaml \
    --model-name $model_name \
    --perplexity-eval-batch-size 2 \
    --disable-perplexity-eval --lm-eval-num-fewshot 0 --lm-eval-batch-size auto \
    --max-position-embeddings 2048 \
    $other_args \
    --output-dir ./checkpoints/ptq/${scaling_mode}/${model_name_esc} \
    --num-calibration-samples 128
