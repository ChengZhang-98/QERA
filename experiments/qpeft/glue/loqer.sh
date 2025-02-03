workdir=~/Projects/QERA/experiments/jamie/peft_cls
cd $workdir

function check_return_value() {
    if [[ $? -ne 0 ]]; then
        echo "❌ $1 failed."
        exit 1
    fi
}

task_list=("mrpc" "mnli" "qnli" "rte" "sst2" "cola" "qqp" "stsb")
rank=8
quant_type=fp
quant_bits=4
seed=42
for task in ${task_list[@]}; do
    cd $workdir
    bash ./adapt_and_glue_train.sh Anonymous-Pineapple/roberta-base $task qera $rank $quant_type $quant_bits $seed
    check_return_value "qera $task"
done
