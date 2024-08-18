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

import torch
from transformers import (
    SchedulerType,
    MODEL_MAPPING,

)
from peft.import_utils import is_bnb_4bit_available

from peft.utils.loftq_utils import _SafetensorLoader

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
def loftQ_parse_args(parser, use_existing_parser=False):
    if not use_existing_parser:
        parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    ##########################
    #   Generation Config    #
    ##########################
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument("--k", type=int, default=40, help="Choose k candidate words")
    parser.add_argument("--p", type=float, default=0.95, help="The sum of probability of candidate words is 0.9 ")

    ##########################
    #        Exp Args        #
    ##########################
    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default=None,
        help=(
            "The LoRA adapter checkpoint. Set None if you want to fine-tune from LoftQ."
            "Specify a path if you want to evaluate."
        ),
    )
    ##########################
    #        LoRA Args       #
    ##########################
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="The rank of the LoRA adapter.",
    )

    if use_existing_parser is True:
        return
    args = parser.parse_args()

    # Sanity checks
    # if args.dataset_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a dataset name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    # if args.push_to_hub:
    #     assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def _low_rank_decomposition(weight, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {matrix_dimension} dimensions.")

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, "reduced_rank": reduced_rank}


@torch.no_grad()
def _loftq_init_new(qweight, weight, num_bits: int, reduced_rank: int):
    import bitsandbytes as bnb

    if num_bits != 4:
        raise ValueError("Only 4 bit quantization supported at the moment.")
    if not is_bnb_4bit_available():
        raise ValueError("bitsandbytes 4bit quantization is not available.")

    compute_device = "cuda"
    dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, qweight.quant_state)

    weight = weight.to(device=compute_device, dtype=torch.float32)
    residual = weight - dequantized_weight
    torch.cuda.empty_cache()
    # Decompose the residualidual by SVD
    output = _low_rank_decomposition(residual, reduced_rank=reduced_rank)
    L, R, reduced_rank = output["L"], output["R"], output["reduced_rank"]
    return R, L


@torch.no_grad()
def replace_lora_weights_loftq(
    peft_model,
    model_path: Optional[str] = None,
    adapter_name: str = "default",
    callback: Optional[Callable[[torch.nn.Module, str], bool]] = None,
    AB_dict: Optional[dict] = None,
):
    """
    Replace the LoRA weights of a model quantized with bitsandbytes, using the LoftQ technique.

    The replacement is done on the fly by loading in the non-quantized weights from a locally stored safetensors model
    file and initializing the LoRA weights such that the quantization error between the original and quantized weights
    is minimized.

    As lazy loading is not possible with pickle, normal PyTorch checkpoint files cannot be supported.

    Depending on the model size, calling this function may take some time to finish.

    Args:
        peft_model (`PeftModel`):
            The model to replace the weights of. Must be a quantized PEFT model with LoRA layers.
        model_path (`Optional[str]`):
            The path to the model safetensors file. If the model is a Hugging Face model, this will be inferred from
            the model's config. Otherwise, it must be provided.
        adapter_name (`str`):
            The name of the adapter to replace the weights of. The default adapter name is "default".
        callback (`Optional[Callable[[PeftModel, str], bool]]`):
            A callback function that will be called after each module is replaced. The callback function should take
            the model and the name of the current module as input and return a boolean indicating whether the
            replacement should be kept. If the callback returns False, the replacement will be rolled back. This can be
            very useful to confirm that the LoftQ initialization actually decreases the quantization error of the
            model. As an example, this callback could generate logits for given input and compare it with the logits
            from the original, non-quanitzed model with the same input, and only return `True` if there is an
            improvement. As this is a greedy optimization, it's possible that calling this function multiple times
            yields incremental improvements.
    """
    if not is_bnb_4bit_available():
        raise ValueError("bitsandbytes must be installed and the model must be quantized in 4bits.")

    from peft.tuners.lora import Linear4bit

    # model_path = _check_model_path_loftq(model_path, peft_model)
    prefix = "base_model.model."
    any_match = False
    safetensor_loader = _SafetensorLoader(peft_model, model_path)

    # if too slow, consider adding tqdm as an option
    for name, module in peft_model.named_modules():
        if not isinstance(module, Linear4bit):
            continue

        if not name.startswith(prefix):
            raise TypeError("The passed model does not appear to be a valid PeftModel")

        any_match = True
        name = name[len(prefix) :]
        if AB_dict:
            # "model.layers.0.self_attn.o_proj.A"
            lora_A = AB_dict[name + ".A"].T
            lora_B = AB_dict[name + ".B"].T
            device = module.lora_A[adapter_name].weight.device
            dtype = module.lora_A[adapter_name].weight.dtype
            lora_A = lora_A.to(dtype).to(device)
            lora_B = lora_B.to(dtype).to(device)
        else:
            tensor = safetensor_loader.get_tensor(name + ".weight")

            reduced_rank = module.r[adapter_name]
            lora_A, lora_B = _loftq_init_new(module.weight, tensor, num_bits=4, reduced_rank=reduced_rank)
        if not callback:
            module.lora_A[adapter_name].weight.data = lora_A
            module.lora_B[adapter_name].weight.data = lora_B
            continue

        lora_A_before = module.lora_A[adapter_name].weight.data
        lora_B_before = module.lora_B[adapter_name].weight.data

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
        should_replace = callback(peft_model, name)
        if not should_replace:
            # roll back
            module.lora_A[adapter_name].weight.data = lora_A_before
            module.lora_B[adapter_name].weight.data = lora_B_before

        del lora_A_before, lora_B_before

    if not any_match:
        raise ValueError("No bnb LoRA module found on the model")
