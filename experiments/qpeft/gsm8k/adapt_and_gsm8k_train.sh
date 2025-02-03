workdir=~/Projects/QERA
ckpt_dir=$workdir/checkpoints/gsm8k
env_name=qera
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
if [[ -z $8 ]]; then
    if [[ $quant_type == "mxint" ]]; then
        echo "Usage: $8 <mxint_block_size>"
        exit 1
    fi
    mxint_block_size=32
else
    mxint_block_size=$8
fi

overwrite_adapt_dir=false

max_seq_len=1024
lora_alpha=$((lora_rank * 2))

# loftq
loftq_num_iters=5

# qera
qera_num_calibration_samples=512
qera_calibration_batch_size=2
if [[ $quant_bits == 2 ]]; then
    qera_scaling_mode=rxx
else
    qera_scaling_mode=diag
fi

qera_calibration_seq_len=$max_seq_len
task_name_for_calibration="slim_pajama_100m_peft"

other_train_flags=""

# training
per_device_train_batch_size=4
per_device_eval_batch_size=2
num_train_epochs=10
gradient_accumulation_steps=4
lr_scheduler_type=cosine
num_warmup_steps=100

model_name_esc=$(echo $model_name | sed 's/\//-/g')
dataset_name_esc="gsm8k"

# lora, qlora, loftq, qera
if [[ $adapter_init == "qlora" ]]; then
    adapt_output_dir=${ckpt_dir}/qlora_clm/${model_name_esc}_rank-${lora_rank}_${quant_type}_${quant_bits}bit_seed-${seed}
elif [[ $adapter_init == "loftq" ]]; then
    adapt_output_dir=${ckpt_dir}/loftq_clm/${model_name_esc}_rank-${lora_rank}_${loftq_num_iters}iter_${quant_type}_${quant_bits}bit
elif [[ $adapter_init == "lora" ]]; then
    adapt_output_dir=${ckpt_dir}/lora_clm/${model_name_esc}_rank-${lora_rank}_seed-${seed}
elif [[ $adapter_init == "qera" ]]; then
    adapt_output_dir=${ckpt_dir}/qera_clm/${model_name_esc}_rank-${lora_rank}_${qera_scaling_mode}_calibrated-on-${task_name_for_calibration}_${quant_type}_${quant_bits}bit
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
        conda run -n $env_name --no-capture-output python adapt_and_save.py \
            clm $model_name $adapter_init $adapt_output_dir \
            --qera-calibration-set $task_name_for_calibration \
            --qera-num-calibration-samples $qera_num_calibration_samples \
            --qera-calibration-batch-size $qera_calibration_batch_size \
            --qera-max-seq-length $qera_calibration_seq_len \
            --qera-scaling-mode $qera_scaling_mode \
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

learning_rate_list=(3e-5)
# loop over learning rates
for learning_rate in ${learning_rate_list[@]}; do
    timestamp=$(date +%Y%m%d-%H%M%S)
    training_ckpt_dir=${ckpt_dir}/fine-tune-ckpt/${dataset_name_esc}/${model_name_esc}/${adapter_init}/$(basename $adapt_output_dir)_lr-${learning_rate}_${timestamp}
    run_name=${dataset_name_esc}_${adapter_init}_$(basename $adapt_output_dir)_lr-${learning_rate}

    if [[ $adapter_init == "full-finetune" || $adapter_init == "lora" ]]; then
        tags="${dataset_name_esc} ${adapter_init} ${model_name_esc} rank-${lora_rank}"
    else
        tags="debug ${dataset_name_esc} ${adapter_init} ${model_name_esc} rank-${lora_rank} ${quant_type} ${quant_bits}-bit"
        if [[ $adapter_init == "qera" ]]; then
            tags="${tags} ${qera_scaling_mode} calibrated-on-${qera_calibration_set_type}"
        fi
        if [[ $quant_type == "mxint" ]]; then
            tags="${tags} mxint-block-size-${mxint_block_size}"
        fi
    fi

    conda run -n $env_name --no-capture-output accelerate launch train_gsm8k.py \
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

