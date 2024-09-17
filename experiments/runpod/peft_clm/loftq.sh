workdir=/workspace/LoQER/experiments/runpod/peft_clm
cd $workdir

function check_return_value() {
    if [[ $? -ne 0 ]]; then
        echo "❌ $1 failed."
    fi
}

model_name=meta-llama/Meta-Llama-3.1-8B
# model_name=Cheng98/TinyLlama_v1.1
# task="wikitext2"
task="slim_pajama_100m"
adapter_init="loftq"
lora_rank=8
quant_type="fp"
quant_bits=2
seed=42
mxint_block_size=32

timestamp=$(date +%Y%m%d-%H%M%S)

bash adapt_and_clm_train.sh $model_name \
    $task \
    $adapter_init \
    $lora_rank \
    $quant_type $quant_bits \
    $seed \
    $mxint_block_size 2>&1 | tee loftq_${timestamp}.log
