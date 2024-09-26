# envs
workdir=~/Projects/LoQER
ckpt_dir=$workdir/checkpoints/glue
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
    echo "Usage: $5 <quant_type>. fp, nf, or mxint"
    exit 1
else
    quant_type=$5
fi
# read quant_bits from $6
if [[ -z $6 ]]; then
    echo "Usage: $6 <quant_bits>, 2, 3, or 4."
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
    if [[ $quant_type == "mxint" ]]; then
        echo "Usage: $8 <mxint_block_size>"
        exit 1
    fi
    mxint_block_size=32
else
    mxint_block_size=$8
fi

if [[ $task_name == "mnli" ]]; then
    num_labels=3
elif [[ $task_name == "stsb" ]]; then
    num_labels=1
else
    num_labels=2
fi
overwrite_adapt_dir=false

max_seq_len=256
lora_alpha=$((lora_rank * 2)) # alpha = 2 * rank

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
loqer_calibration_set_type=pretrain
if [[ $loqer_calibration_set_type == "pretrain" ]]; then
    loqer_calibration_seq_len=512
else
    loqer_calibration_seq_len=$max_seq_len
fi

if [[ $loqer_calibration_set_type == "downstream" ]]; then
    task_name_for_calibration=glue,${task_name}_peft
else
    task_name_for_calibration=wikitext2_mlm
fi

# adapt and save

other_flags=""

# training
per_device_train_batch_size=64
per_device_eval_batch_size=64
num_train_epochs=5
gradient_accumulation_steps=1
lr_scheduler_type=cosine

# format names
model_name_esc=$(echo $model_name | sed 's/\//-/g')
dataset_name_esc=$(echo $task_name | sed 's/,/-/g')
dataset_name_esc=$(echo $dataset_name_esc | sed 's/\//-/g')

# lora, qlora, loftq, loqer
if [[ $adapter_init == "qlora" ]]; then
    adapt_output_dir=${ckpt_dir}/qlora_cls/${model_name_esc}_${num_labels}-labels_rank-${lora_rank}_${quant_type}_${quant_bits}bit
elif [[ $adapter_init == "loftq" ]]; then
    adapt_output_dir=${ckpt_dir}/loftq_cls/${model_name_esc}_${num_labels}-labels_rank-${lora_rank}_${loftq_num_iters}iter_${quant_type}_${quant_bits}bit
elif [[ $adapter_init == "lora" ]]; then
    adapt_output_dir=${ckpt_dir}/lora_cls/${model_name_esc}_rank-${lora_rank}
elif [[ $adapter_init == "loqer" ]]; then
    if [[ $loqer_calibration_set_type == "downstream" ]]; then
        adapt_output_dir=${ckpt_dir}/loqer_cls/calibrated-on-${dataset_name_esc}/${model_name_esc}_${num_labels}-labels_rank-${lora_rank}_${loqer_scaling_mode}_calibrated-on-${loqer_calibration_set_type}_${quant_type}_${quant_bits}bit
    else
        adapt_output_dir=${ckpt_dir}/loqer_cls/calibrated-on-${task_name_for_calibration}/${model_name_esc}_${num_labels}-labels_rank-${lora_rank}_${loqer_scaling_mode}_calibrated-on-${loqer_calibration_set_type}_${quant_type}_${quant_bits}bit
    fi
elif [[ $adapter_init == "full-finetune" ]]; then
    adapt_output_dir=${ckpt_dir}/full-finetune/$dataset_name_esc/${model_name_esc}
else
    echo "Invalid adapter_init: $adapter_init"
    exit 1
fi

if [[ $adapter_init != "full-finetune" && $adapter_init != "lora" ]]; then
    if [[ $quant_type == "mxint" ]]; then
        adapt_output_dir=${adapt_output_dir}_blocksize-${mxint_block_size}
    fi
fi
adapt_output_dir=${adapt_output_dir}_seed-${seed}

if [[ $adapter_init != "full-finetune" ]]; then
    # if output_dir not exists, create adapted model
    if [[ $overwrite_adapt_dir == "true" || ! -d $adapt_output_dir ]]; then
        conda run -n $env_name --no-capture-output python adapt_and_save.py \
            cls $model_name $adapter_init $adapt_output_dir \
            --loqer-calibration-set $task_name_for_calibration \
            --loqer-num-calibration-samples $loqer_num_calibration_samples \
            --loqer-calibration-batch-size $loqer_calibration_batch_size \
            --loqer-calibration-set-type $loqer_calibration_set_type \
            --loqer-max-seq-length $loqer_calibration_seq_len \
            --loqer-scaling-mode $loqer_scaling_mode \
            --loftq-num-iters $loftq_num_iters \
            --quant-type $quant_type \
            --quant-bits $quant_bits \
            --device-map cuda \
            --lora-rank $lora_rank \
            --lora-alpha $lora_alpha \
            --num-workers 8 \
            --seed $seed \
            --num-labels $num_labels \
            --mxint-block-size $mxint_block_size \
            --peek-post-init-metrics -ow $other_flags # overwrite the output directory if it exists
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

learning_rate_list=(5e-4 4e-4 3e-4 2e-4 1e-4)
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

    conda run -n $env_name --no-capture-output python glue_train.py \
        --tokenizer_name $model_name --config_name $model_name \
        --task_name $task_name --max_length $max_seq_len --model_name_or_path $model_name_or_path \
        --use_slow_tokenizer \
        --per_device_train_batch_size $per_device_train_batch_size --per_device_eval_batch_size $per_device_eval_batch_size \
        --learning_rate $learning_rate \
        --lr_scheduler_type $lr_scheduler_type \
        --num_train_epochs $num_train_epochs \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --seed $seed \
        --output_dir $training_ckpt_dir \
        --lora_adapter_dir $lora_adapter_dir \
        --with_tracking --report_to wandb --run_name $run_name --wandb-tags $tags

    # check return value
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to train the model."
        exit 1
    fi
done
