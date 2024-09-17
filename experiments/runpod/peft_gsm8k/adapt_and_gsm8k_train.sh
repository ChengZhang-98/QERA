workdir=/workspace/LoQER
ckpt_dir=$workdir/checkpoints/runpod
cd $workdir

function check_return_value() {
    if [[ $? -ne 0 ]]; then
        echo "❌ $1 failed."
        exit 1
    fi
}

# read model_name from $1
if [[ -z $1 ]]; then
    echo "Usage: $1 <model_name>"
    exit 1
else
    model_name=$1
fi
# read adapt_init from $2
if [[ -z $3 ]]; then
    echo "Usage: $2 <adapter_init>"
    exit 1
else
    adapter_init=$2
fi
# read lora_rank from $3
if [[ -z $3 ]]; then
    echo "Usage: $3 <lora_rank>"
    exit 1
else
    lora_rank=$3
fi
# read quant_type from $4
if [[ -z $4 ]]; then
    echo "Usage: $4 <quant_type>"
    exit 1
else
    quant_type=$4
fi
# read quant_bits from $5
if [[ -z $5 ]]; then
    echo "Usage: $5 <quant_bits>"
    exit 1
else
    quant_bits=$5
fi
# read seed from $6
if [[ -z $6 ]]; then
    echo "Usage: $6 <seed>"
    exit 1
else
    seed=$6
fi
# read mxint_block_size from $8
if [[ -z $7 ]]; then
    echo "Defaulting $7 <mxint_block_size> to 64"
    mxint_block_size=64
else
    mxint_block_size=$7
fi

overwrite_adapt_dir=false

max_seq_len=1024
lora_alpha=$((lora_rank * 2))

# loftq
loftq_num_iters=5

# loqer
loqer_num_calibration_samples=256
loqer_calibration_batch_size=2
if [[ $quant_bits == 2 ]]; then
    loqer_scaling_mode=rxx
else
    loqer_scaling_mode=diag
fi

loqer_calibration_seq_len=$max_seq_len
task_name_for_calibration="slim_pajama_100m_peft"

# other_train_flags="--max_train_steps 16"
other_train_flags=""

# training
per_device_train_batch_size=8
per_device_eval_batch_size=8
num_train_epochs=6
gradient_accumulation_steps=4
lr_scheduler_type=cosine
num_warmup_steps=100

model_name_esc=$(echo $model_name | sed 's/\//-/g')
dataset_name_esc="gsm8k"

# lora, qlora, loftq, loqer
if [[ $adapter_init == "qlora" ]]; then
    adapt_output_dir=${ckpt_dir}/qlora_clm/${model_name_esc}_rank-${lora_rank}_${quant_type}_${quant_bits}bit_seed-${seed}
elif [[ $adapter_init == "loftq" ]]; then
    adapt_output_dir=${ckpt_dir}/loftq_clm/${model_name_esc}_rank-${lora_rank}_${loftq_num_iters}iter_${quant_type}_${quant_bits}bit
elif [[ $adapter_init == "lora" ]]; then
    adapt_output_dir=${ckpt_dir}/lora_clm/${model_name_esc}_rank-${lora_rank}_seed-${seed}
elif [[ $adapter_init == "loqer" ]]; then
    adapt_output_dir=${ckpt_dir}/loqer_clm/${model_name_esc}_rank-${lora_rank}_${loqer_scaling_mode}_calibrated-on-${task_name_for_calibration}_${quant_type}_${quant_bits}bit
elif [[ $adapter_init == "full-finetune" ]]; then
    adapt_output_dir=${ckpt_dir}/full-finetune_clm/$dataset_name_esc/${model_name_esc}
else
    echo "Invalid adapter_init: $adapter_init"
    exit 1
fi

if [[ $adapter_init != "full-finetune" && $adapter_init != "lora" && $quant_type == "mxint" ]]; then
    adapt_output_dir=${adapt_output_dir}_blocksize-${mxint_block_size}
fi
# adapt_output_dir=${adapt_output_dir}

if [[ $adapter_init != "full-finetune" ]]; then
    # if output_dir not exists, create adapted model
    if [[ $overwrite_adapt_dir == "true" || ! -d $adapt_output_dir ]]; then
        python adapt_and_save.py \
            clm $model_name $adapter_init $adapt_output_dir \
            --loqer-calibration-set $task_name_for_calibration \
            --loqer-num-calibration-samples $loqer_num_calibration_samples \
            --loqer-calibration-batch-size $loqer_calibration_batch_size \
            --loqer-max-seq-length $loqer_calibration_seq_len \
            --loqer-scaling-mode $loqer_scaling_mode \
            --loftq-num-iters $loftq_num_iters \
            --quant-type $quant_type \
            --quant-bits $quant_bits \
            --device-map auto-balanced \
            --lora-rank $lora_rank \
            --lora-alpha $lora_alpha \
            --lora-dropout 0 \
            --num-workers 16 \
            --seed $seed \
            --mxint-block-size $mxint_block_size \
            -ow # --peek-post-init-metrics # overwrite the output directory if it exists
        check_return_value "Adapt and save"
        sleep 2
    else
        if [[ $overwrite_adapt_dir == "false" && -d $adapt_output_dir ]]; then
            echo "🔍 $adapt_output_dir exists. Skip adapting and saving the model."
            sleep 2
        fi
    fi
fi

if [[ $adapter_init != "full-finetune" ]]; then
    model_name_or_path=$adapt_output_dir/base_model
    lora_adapter_dir=$adapt_output_dir/adapter
else
    # full-finetune
    model_name_or_path=$model_name
    lora_adapter_dir="NA"
fi

learning_rate_list=(6e-5)
# loop over learning rates
for learning_rate in ${learning_rate_list[@]}; do
    timestamp=$(date +%Y%m%d-%H%M%S)
    training_ckpt_dir=${ckpt_dir}/fine-tune-ckpt/${dataset_name_esc}/${model_name_esc}/${adapter_init}/$(basename $adapt_output_dir)_lr-${learning_rate}_${timestamp}
    run_name=${dataset_name_esc}_${adapter_init}_$(basename $adapt_output_dir)_lr-${learning_rate}_seed-${seed}

    if [[ $adapter_init == "full-finetune" || $adapter_init == "lora" ]]; then
        tags="${dataset_name_esc} ${adapter_init} ${model_name_esc} rank-${lora_rank}"
    else
        tags="debug ${dataset_name_esc} ${adapter_init} ${model_name_esc} rank-${lora_rank} ${quant_type} ${quant_bits}-bit"
        if [[ $adapter_init == "loqer" ]]; then
            tags="${tags} ${loqer_scaling_mode} calibrated-on-${loqer_calibration_set_type}"
        fi
        if [[ $quant_type == "mxint" ]]; then
            tags="${tags} mxint-block-size-${mxint_block_size}"
        fi
    fi

    accelerate launch train_gsm8k.py \
        --model_name_or_path $model_name_or_path --tokenizer_name $model_name \
        --lora_adapter_dir $lora_adapter_dir \
        --output_dir $training_ckpt_dir \
        --learning_rate $learning_rate \
        --weight_decay 0.01 \
        --lr_scheduler_type cosine \
        --num_warmup_steps $num_warmup_steps \
        --seed $seed \
        --dataset_name gsm8k \
        --dataset_config main \
        --pad_to_max_length \
        --max_source_length 128 \
        --max_target_length 256 \
        --num_train_epochs $num_train_epochs \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_eval_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --with_tracking \
        --report_to wandb \
        --run_name $run_name \
        --wandb_tags $tags

    # check return value
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to train the model."
        exit 1
    fi
done

# huggingface script example
# python train_gsm8k_llama.py \
#     --model_name_or_path LoftQ/Llama-2-13b-hf-4bit-64rank \
#     --output_dir exp_results/gsm8k/llama-2-13b/bit4-rank64/lr1e-4 \
#     --learning_rate 1e-4  \
#     --weight_decay 0.1 \
#     --lr_scheduler_type cosine \
#     --num_warmup_steps 100 \
#     --seed 202 \
#     --dataset_name gsm8k \
#     --dataset_config main \
#     --pad_to_max_length \
#     --max_source_length 128 \
#     --max_target_length 256 \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --with_tracking \
#     --report_to tensorboard
