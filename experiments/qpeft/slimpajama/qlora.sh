model_name=meta-llama/Llama-2-7b-hf
task="slim_pajama_100m"
adapter_init="qlora"
lora_rank=8
quant_type="fp"
quant_bits=4
seed=42


bash adapt_and_clm_train.sh $model_name \
    $task \
    $adapter_init \
    $lora_rank \
    $quant_type $quant_bits \
    $seed
