workdir=~/Projects/LoQER/experiments/jamie/peft_clm
cd $workdir

function check_return_value() {
    if [[ $? -ne 0 ]]; then
        echo "❌ $1 failed."
    fi
}

model_name=meta-llama/Llama-2-7b-hf
# model_name=Cheng98/TinyLlama_v1.1
# task="wikitext2"
task="slim_pajama_100m"
adapter_init="lora"
lora_rank=8
quant_type="fp"
quant_bits=4
seed=42

timestamp=$(date +%Y%m%d-%H%M%S)

bash adapt_and_clm_train.sh $model_name \
    $task \
    $adapter_init \
    $lora_rank \
    $quant_type $quant_bits \
    $seed 2>&1 | tee lora_${timestamp}.log
