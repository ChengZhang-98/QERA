# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import logging
import math
import os
import random
import re
from pathlib import Path
from typing import Callable, Optional

import datasets
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger

from itertools import chain
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    Trainer
)
from dataclasses import dataclass, field
import evaluate
from transformers.testing_utils import CaptureLogger
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import get_peft_model, LoraConfig, PeftModel, TaskType

from peft.import_utils import is_bnb_4bit_available

from peft.utils.loftq_utils import _SafetensorLoader


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.32.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from .utils import replace_lora_weights_loftq

def loftQ_fine_tuning_for_wikitext(args, model, tokenizer, AB_dict):
    train_args = TrainingArguments
    """
    Fine-tune a model using Lora and initialize the lora adapter weights with AB_dict.
    If AB_dict is None, the model will be initialized with LoftQ.

    args.dataset_name: str
        The name of the dataset to use (via the datasets library).
        If the dataset is GSM8K, the fine-tuning will be done on GSM8K dataset and custom accuracy evaluation will be done by following the LoftQ paper.
        Otherwise, this code will return the fine-tuned model but not the evaluation.

    args: argparse.Namespace
        The arguments for the fine-tuning.
    model: transformers models
        The model to fine-tune.
    AB_dict: dict
        The dictionary containing the AB weights for the model.

    Returns (model | None): The fine-tuned model for further evaluation.
    """
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
    # args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            api = HfApi(token=args.hub_token)

            # Create repo (repo_name from args or inferred)
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{args.validation_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[{args.validation_split_percentage}%:]",
        )


    ##########################
    #       Peft Model       #
    ##########################
    # Download weights and configure LoRA
    if args.adapter_name_or_path is None:
        if any(name in args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
            task_type = TaskType.CAUSAL_LM
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

        elif any(name in args.model_name_or_path.lower() for name in ["bart", "t5"]):
            task_type = TaskType.SEQ_2_SEQ_LM
            target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]

        elif any(name in args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]):
            task_type = TaskType.SEQ_CLS
            target_modules = ["query_proj", "key_proj", "value_proj", "dense"]  # embeddings not supported by peft
        else:
            raise NotImplementedError("Other models not supported yet.")

        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=args.rank,
            lora_alpha=16 if task_type is TaskType.CAUSAL_LM else args.rank,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)
        if AB_dict is not None:
            for name, item in AB_dict.items():
                if isinstance(item, torch.Tensor):
                    AB_dict[name] = item.contiguous()
            replace_lora_weights_loftq(model)
            # replace_lora_weights_loftq(model, AB_dict=AB_dict)
        else:
            logger.warning("AB_dict is None. The model will use the LoftQ initialization.")
            replace_lora_weights_loftq(model)
    else:
        model = PeftModel.from_pretrained(model, args.adapter_name_or_path, is_trainable=True)
    model.print_trainable_parameters()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    ##########################
    #      WIKITEXT dataset     #
    ##########################
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )
    block_size = min(1024, tokenizer.model_max_length)
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    # Train!
    logger.info("*** Training ***")
    training_args = TrainingArguments(
        output_dir=args.output_dir,           # Output directory for saving model and logs
        num_train_epochs=args.num_train_epochs,               # Number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.num_warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
    )   
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message="End of training",
                )


PATTERN_NUMBER = re.compile(r"-?\d+\.?\d*")


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = PATTERN_NUMBER.findall(sentence)
    if not pred:
        return float("inf")
    segment = sentence.split("The final answer is ")
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = PATTERN_NUMBER.findall(pred_answer)
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError:
            pred_answer = float("inf")
    return pred_answer


def compute_accuracy(pred: list, gold: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1

    return acc / len(pred)
