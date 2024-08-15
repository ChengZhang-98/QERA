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

from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
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

def _compute_scale_inv_dot_U(scale: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    If not the scale matrix is not invertible, add turbulence to the scale matrix

    Perform S^-1 @ U using `torch.linalg.solve`, which is more numerically stable.
    Refer to https://pytorch.org/docs/stable/generated/torch.linalg.inv.html
    """
    if scale.ndim == 1:
        scale = torch.where(scale <= 0, torch.ones_like(scale) * torch.finfo(scale.dtype).eps, scale)
        return torch.linalg.solve(torch.diag(scale), U)
    elif scale.ndim == 2:
        try:
            return torch.linalg.solve(scale, U)
        except RuntimeError as e:
            logger.warning(f"Matrix inversion failed: {e} Adding turbulence to the scale matrix")
            U_scale, S_scale, V_T_scale = torch.linalg.svd(scale)
            S_scale = torch.where(S_scale <= 0, torch.ones_like(S_scale) * torch.finfo(S_scale.dtype).eps, S_scale)
            scale = U_scale @ torch.diag(S_scale) @ V_T_scale
            return torch.linalg.solve(scale, U)
    else:
        raise ValueError("Scale must be either a vector (diagonal) or a matrix")


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
def _loftq_init_new(qweight, weight, num_bits: int, reduced_rank: int, scale=None):
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
    if scale is None:
        output = _low_rank_decomposition(residual, reduced_rank=reduced_rank)
    else:
        output = _low_rank_loqer_decomposition(residual, scale=scale, reduced_rank=reduced_rank)
    L, R, reduced_rank = output["L"], output["R"], output["reduced_rank"]
    return R, L

    
def _low_rank_loqer_decomposition(weight, scale, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {matrix_dimension} dimensions.")   

    scale = scale.to(weight.dtype).to(weight.device)
    if scale.ndim == 1:
        assert scale.shape[0] == weight.shape[1], "Scale must have the same number of elements as the weight"
        scaled_q_error_T = torch.diag(scale) @ weight.transpose(0, 1)
    elif scale.ndim == 2:
        assert scale.shape[0] == scale.shape[1], "Scale must be a square matrix"
        scaled_q_error_T = scale @ weight.transpose(0, 1)
    else:
        raise ValueError("Scale must be either a vector (diagonal) or a matrix") 

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, V_T = torch.linalg.svd(scaled_q_error_T, full_matrices=False)

    U = U[:, :reduced_rank]
    S = S[:reduced_rank]
    V_T = V_T[:reduced_rank, :]

    if scale.ndim == 1:
        A_T = _compute_scale_inv_dot_U(scale, U)
        B_T = torch.diag(S) @ V_T # ab_quantizer(torch.diag(S) @ V_T) # TODO: since AB is not quantized, does that mean we don't need the ab_quantizer?
    elif scale.ndim == 2:
        A_T = _compute_scale_inv_dot_U(scale, U)
        B_T = torch.diag(S) @ V_T # ab_quantizer(torch.diag(S) @ V_T) # TODO: since AB is not quantized, does that mean we don't need the ab_quantizer?

    return {"L": B_T.T, "R": A_T.T, "U": U, "S": S, "Vh": V_T, "reduced_rank": reduced_rank}


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
    for name, module in tqdm(peft_model.named_modules(), desc="Computing low-rank A and B from scale"):
        if not isinstance(module, Linear4bit):
            continue

        if not name.startswith(prefix):
            raise TypeError("The passed model does not appear to be a valid PeftModel")

        any_match = True
        name = name[len(prefix) :]

        if AB_dict:
            name = name[len(prefix) :]
            tensor = safetensor_loader.get_tensor(name + ".weight")
            reduced_rank = module.r[adapter_name]
            scale = AB_dict[name + ".scale"] # name example: "model.layers.0.self_attn.o_proj.scale"
            lora_A, lora_B = _loftq_init_new(module.weight, tensor, num_bits=4, reduced_rank=reduced_rank, scale=scale)
            ###### Directly replace the weights with Loqer AB_dict ######
            #     lora_A = AB_dict[name + ".A"].T # name example: "model.layers.0.self_attn.o_proj.A"
            #     lora_B = AB_dict[name + ".B"].T
            #     device = module.lora_A[adapter_name].weight.device
            #     dtype = module.lora_A[adapter_name].weight.dtype
            #     lora_A = lora_A.to(dtype).to(device)
            #     lora_B = lora_B.to(dtype).to(device)
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


def loftQ_fine_tuning(args, model, tokenizer, AB_dict):
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
    if args.dataset_name is not None:
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
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )


    ##########################
    #       Peft Model       #
    ##########################
    # Download weights and configure LoRA
    if args.adapter_name_or_path is None:
        # model = PeftModel.from_pretrained(model, args.model_name_or_path, subfolder="loftq_init", is_trainable=True)
        if any(name in args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
            # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            task_type = TaskType.CAUSAL_LM
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

        elif any(name in args.model_name_or_path.lower() for name in ["bart", "t5"]):
            # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
            task_type = TaskType.SEQ_2_SEQ_LM
            target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]

        elif any(name in args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]):
            # model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
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
            model = get_peft_model(model, lora_config)
            replace_lora_weights_loftq(model, AB_dict=AB_dict)
        else:
            logger.warning("AB_dict is None. The model will use the LoftQ initialization.")
            model = PeftModel.from_pretrained(model, args.model_name_or_path, subfolder="loftq_init", is_trainable=True)
            # replace_lora_weights_loftq(model) # QLORA
    else:
        model = PeftModel.from_pretrained(model, args.adapter_name_or_path, is_trainable=True)
    model.print_trainable_parameters()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    ##########################
    #      GSM8K dataset     #
    ##########################

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for source/target.
    source_column, target_column = "question", "answer"

    # Temporarily set max_target_length for training.
    padding = "max_length" if args.pad_to_max_length else False
    task_prompt = "\nAnswer the above question. First think step by step and then answer the final number.\n"

    def prompt_process(sent_1, sent_2, prompt_1="", prompt_2="", prompt_3=""):
        sent_2 = sent_2.replace("####", "The final answer is")
        return prompt_1 + sent_1 + prompt_2 + sent_2 + prompt_3

    def preprocess_function_train(examples):
        sources = examples[source_column]
        targets = examples[target_column]

        inputs = [prompt_process(source, target, prompt_2=task_prompt) for (source, target) in zip(sources, targets)]

        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length + args.max_target_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )

        labels = copy.deepcopy(model_inputs)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            # get the length of the target tokens. -1 to kick out the <BOS> token
            target_tokens = tokenizer(targets, padding=False)
            target_len = [len(label) - 1 for label in target_tokens["input_ids"]]

            # don't calculate the loss from source and padding (left padding)
            for i in range(len(labels["input_ids"])):
                labels["input_ids"][i, : -target_len[i]] = -100

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_function_test(examples):
        sources = examples[source_column]
        labels = examples[target_column]

        inputs = [source + task_prompt for source in sources]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(labels, max_length=args.max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function_train,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on training dataset",
        )

        eval_dataset = raw_datasets["test"].map(
            preprocess_function_test,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )

    # Log a few random samples from the set:
    for index in random.sample(range(len(train_dataset)), 2):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    for index in random.sample(range(len(eval_dataset)), 2):
        logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "lora" in n],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        if type(experiment_config["lr_scheduler_type"]) is str:
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        else:
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                if completed_steps % 50:
                    accelerator.print(f"Epoch: {epoch} | Step: {completed_steps} | Loss: {loss}")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        if args.dataset_name.lower() != "gsm8k":
            return model

        # GSM8K Accuracy Evaluation
        model.eval()
        # TODO: How does it even work? The model need to be still be training in the next iteration
        # TODO: Do I have to revert the type?
        # model.bfloat16() # This is needed for evaluating loftq
        # model.to(getattr(torch, args.eval_dtype))
        gen_kwargs = {
            "max_new_tokens": args.max_target_length,
            "temperature": args.temperature,
            "top_k": args.k,
            "top_p": args.p,
            "do_sample": True,
        }
        ans_pred_list = []
        ans_gold_list = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                gen_kwargs["input_ids"] = batch["input_ids"]
                gen_kwargs["attention_mask"] = batch["attention_mask"]
                generated_tokens = accelerator.unwrap_model(model).generate(**gen_kwargs)

            pred_tokens = generated_tokens[:, args.max_source_length :]
            pred_tokens = accelerator.pad_across_processes(pred_tokens, dim=1, pad_index=tokenizer.pad_token_id)
            gold_tokens = batch["labels"]

            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                gold_tokens = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            pred_tokens, gold_tokens = accelerator.gather_for_metrics((pred_tokens, gold_tokens))
            pred_tokens, gold_tokens = pred_tokens.cpu().numpy(), gold_tokens.cpu().numpy()

            if isinstance(pred_tokens, tuple):
                pred_tokens = pred_tokens[0]
            decoded_pred = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
            decoded_gold = tokenizer.batch_decode(gold_tokens, skip_special_tokens=True)

            # Extract the numbers in sentences
            accelerator.print(decoded_pred)
            ans_pred_list += [extract_answer_number(sentence_pred) for sentence_pred in decoded_pred]
            ans_gold_list += [extract_answer_number(sentence_gold) for sentence_gold in decoded_gold]

        accelerator.print(ans_pred_list)
        accelerator.print(ans_gold_list)
        accuracy = compute_accuracy(ans_gold_list, ans_pred_list)

        logger.info(f"epoch {epoch}: accuracy: {accuracy}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": accuracy,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message=f"Training in progress epoch {epoch}",
                    run_as_future=True,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

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
