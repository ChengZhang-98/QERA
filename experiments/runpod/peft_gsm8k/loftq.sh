model_name=meta-llama/Llama-2-7b-hf
adapt_init=loftq
lora_rank=128
quant_type=nf
quant_bits=4
seed=42
mxint_block_size=32

bash adapt_and_gsm8k_train.sh \
    $model_name \
    $adapt_init \
    $lora_rank \
    $quant_type \
    $quant_bits \
    $seed \
    $mxint_block_size
