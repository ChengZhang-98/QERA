workdir=~/Projects/LoQER
ckpt_dir=$workdir/checkpoints/jamie
env_name=loqer
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
# read task_name from $2
if [[ -z $2 ]]; then
    echo "Usage: $2 <task_name>"
    exit 1
else
    task_name=$2
fi
# read adapt_init from $3
if [[ -z $3 ]]; then
    echo "Usage: $3 <adapter_init>"
    exit 1
else
    adapter_init=$3
fi
# read lora_rank from $4
if [[ -z $4 ]]; then
    echo "Usage: $4 <lora_rank>"
    exit 1
else
    lora_rank=$4
fi
# read quant_type from $5
if [[ -z $5 ]]; then
    echo "Usage: $5 <quant_type>"
    exit 1
else
    quant_type=$5
fi
# read quant_bits from $6
if [[ -z $6 ]]; then
    echo "Usage: $6 <quant_bits>"
    exit 1
else
    quant_bits=$6
fi
# read seed from $7
if [[ -z $7 ]]; then
    echo "Usage: $7 <seed>"
    exit 1
else
    seed=$7
fi
# read mxint_block_size from $8
if [[ -z $8 ]]; then
    echo "Defaulting $8 <mxint_block_size> to 64"
    mxint_block_size=64
else
    mxint_block_size=$8
fi

overwrite_adapt_dir=false

max_seq_len=1024
lora_alpha=$((lora_rank * 2))

# loftq
loftq_num_iters=5

# loqer
loqer_num_calibration_samples=512
loqer_calibration_batch_size=2
if [[ $quant_bits == 2 ]]; then
    loqer_scaling_mode=rxx
else
    loqer_scaling_mode=diag
fi

loqer_calibration_seq_len=$max_seq_len

if [[ $task_name == "wikitext2" ]]; then
    task_name_for_calibration="wikitext2_peft"
    dataset_flags_for_train="--dataset_name Salesforce/wikitext --dataset_config_name wikitext-2-raw-v1"
elif [[ $task_name == "slim_pajama_6b" ]]; then
    task_name_for_calibration="slim_pajama_6b_peft"
    dataset_flags_for_train="--dataset_name DKYoon/SlimPajama-6B"
elif [[ $task_name == "slim_pajama_1b" ]]; then
    task_name_for_calibration="slim_pajama_1b_peft"
    dataset_flags_for_train="--dataset_name iankur/SlimPajama-1B"
elif [[ $task_name == "slim_pajama_600m" ]]; then
    task_name_for_calibration="slim_pajama_600m_peft"
    dataset_flags_for_train="--dataset_name iankur/SlimPajama-600M" # iankur/SlimPajama-600M
elif [[ $task_name == "slim_pajama_100m" ]]; then
    task_name_for_calibration="slim_pajama_100m_peft"
    dataset_flags_for_train="--dataset_name iankur/SlimPajama-100M"
else
    echo "Unsupported task_name: $task_name"
    exit 1
fi

# other_train_flags="--max_train_steps 16"
other_train_flags=""

# training
per_device_train_batch_size=1
per_device_eval_batch_size=8
# num_train_epochs=3
max_train_steps=1000
evaluate_every_n_steps=200
gradient_accumulation_steps=8
lr_scheduler_type=cosine

model_name_esc=$(echo $model_name | sed 's/\//-/g')
dataset_name_esc=$(echo $task_name | sed 's/,/-/g')
dataset_name_esc=$(echo $dataset_name_esc | sed 's/\//-/g')

# lora, qlora, loftq, loqer
if [[ $adapter_init == "qlora" ]]; then
    adapt_output_dir=${ckpt_dir}/qlora_clm/${model_name_esc}_rank-${lora_rank}_${quant_type}_${quant_bits}bit
elif [[ $adapter_init == "loftq" ]]; then
    adapt_output_dir=${ckpt_dir}/loftq_clm/${model_name_esc}_rank-${lora_rank}_${loftq_num_iters}iter_${quant_type}_${quant_bits}bit
elif [[ $adapter_init == "lora" ]]; then
    adapt_output_dir=${ckpt_dir}/lora_clm/${model_name_esc}_rank-${lora_rank}
elif [[ $adapter_init == "loqer" ]]; then
    adapt_output_dir=${ckpt_dir}/loqer_clm/${model_name_esc}_rank-${lora_rank}_${loqer_scaling_mode}_calibrated-on-${loqer_calibration_set_type}_${quant_type}_${quant_bits}bit
elif [[ $adapter_init == "full-finetune" ]]; then
    adapt_output_dir=${ckpt_dir}/full-finetune_clm/$dataset_name_esc/${model_name_esc}
else
    echo "Invalid adapter_init: $adapter_init"
    exit 1
fi

if [[ $adapter_init != "full-finetune" && $adapter_init != "lora" && $quant_type == "mxint" ]]; then
    adapt_output_dir=${adapt_output_dir}_blocksize-${mxint_block_size}
fi
adapt_output_dir=${adapt_output_dir}_seed-${seed}

if [[ $adapter_init != "full-finetune" ]]; then
    # if output_dir not exists, create adapted model
    if [[ $overwrite_adapt_dir == "true" || ! -d $adapt_output_dir ]]; then
        conda run -n $env_name --no-capture-output python adapt_and_save.py \
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
            --num-workers 16 \
            --seed $seed \
            --mxint-block-size $mxint_block_size \
            --peek-post-init-metrics -ow # overwrite the output directory if it exists
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
        if [[ $adapter_init == "loqer" ]]; then
            tags="${tags} ${loqer_scaling_mode} calibrated-on-${loqer_calibration_set_type}"
        fi
        if [[ $quant_type == "mxint" ]]; then
            tags="${tags} mxint-block-size-${mxint_block_size}"
        fi
    fi

    conda run -n $env_name --no-capture-output torchrun --nproc_per_node 8 clm_train.py \
        --tokenizer_name $model_name --config_name $model_name --model_name_or_path $model_name_or_path \
        $dataset_flags_for_train \
        --block_size $max_seq_len \
        --per_device_train_batch_size $per_device_train_batch_size --per_device_eval_batch_size $per_device_eval_batch_size \
        --preprocessing_num_workers 16 \
        --learning_rate $learning_rate \
        --lr_scheduler_type $lr_scheduler_type \
        --max_train_steps $max_train_steps \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --seed $seed \
        --output_dir $training_ckpt_dir \
        --lora_adapter_dir $lora_adapter_dir \
        --evaluate_every_n_steps $evaluate_every_n_steps \
        --with_tracking --report_to wandb --run_name $run_name --wandb_tags $tags $other_train_flags

    # --num_train_epochs $num_train_epochs \
    # check return value
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to train the model."
        exit 1
    fi
done
