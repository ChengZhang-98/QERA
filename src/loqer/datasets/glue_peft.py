import logging
from itertools import chain

import datasets as hf_datasets
from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def get_raw_data_module_glue_peft(task_name: str) -> hf_datasets.DatasetDict:
    dataset_dict = hf_datasets.load_dataset("nyu-mll/glue", task_name)
    return dataset_dict


def preprocess_data_module_glue_peft(
    task_name: str,
    raw_dataset_dict: hf_datasets.DatasetDict,
    tokenizer,
    model_config,
    pad_to_max_length: bool,
    max_length: int,
    overwrite_cache: bool = False,
):
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = raw_dataset_dict["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config.pad_token_id = tokenizer.pad_token_id

    sentence1_key, sentence2_key = TASK_TO_KEYS[task_name]

    label_to_id = None
    if model_config.label2id != PretrainedConfig(num_labels=num_labels).label2id and not is_regression:
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model_config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model_config.label2id = label_to_id
        model_config.id2label = {id: label for label, id in model_config.label2id.items()}

    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset_dict["train"].column_names,
        desc="Running tokenizer on dataset",
        load_from_cache_file=not overwrite_cache,
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if task_name == "mnli" else "validation"]

    return hf_datasets.DatasetDict(train=train_dataset, validation=eval_dataset)
