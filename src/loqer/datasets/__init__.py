import logging
import datasets as hf_datasets
from argparse import Namespace

from .gsm8k import preprocess_data_module_gsm8k, get_raw_data_module_gsm8k
from .wikitext2 import get_raw_data_module_wikitext2, preprocess_data_module_wikitext2
from .slim_pajama import get_raw_data_module_slim_pajama_6b, preprocess_data_module_slim_pajama_6b
from .wikitext2_peft import get_raw_data_module_wikitext2_peft, preprocess_data_module_wikitext2_peft
from .slim_pajama_6b_peft import get_raw_data_module_slimpajama_6b_peft, preprocess_data_module_slimpajama_6b_peft
from .glue_peft import get_raw_data_module_glue_peft, preprocess_data_module_glue_peft
from .glue_peft import TASK_TO_KEYS as GLUE_TASK_TO_KEYS
from .wikitext2_mlm import get_raw_data_module_wikitext2_mlm, preprocess_data_module_wikitext2_mlm

logger = logging.getLogger(__name__)


def get_raw_data_module(name: str) -> hf_datasets.DatasetDict:
    glue_tasks = [f"glue,{task}_peft" for task in GLUE_TASK_TO_KEYS.keys()]
    if name == "wikitext2":
        return get_raw_data_module_wikitext2()
    elif name == "slim_pajama_6b":
        return get_raw_data_module_slim_pajama_6b()
    elif name == "gsm8k":
        return get_raw_data_module_gsm8k()
    elif name == "wikitext2_peft":
        return get_raw_data_module_wikitext2_peft()
    elif name == "slim_pajama_6b_peft":
        return get_raw_data_module_slimpajama_6b_peft("DKYoon/SlimPajama-6B")
    elif name == "slim_pajama_1b_peft":
        return get_raw_data_module_slimpajama_6b_peft("iankur/SlimPajama-1B")
    elif name == "slim_pajama_600m_peft":
        return get_raw_data_module_slimpajama_6b_peft("iankur/SlimPajama-600M")  # iankur/SlimPajama-600M
    elif name == "slim_pajama_100m_peft":
        return get_raw_data_module_slimpajama_6b_peft("iankur/SlimPajama-100M")
    elif name == "wikitext2_mlm":
        return get_raw_data_module_wikitext2_mlm()
    elif name in glue_tasks:
        # "glue,subtask_peft"
        return get_raw_data_module_glue_peft(name.split(",")[1].removesuffix("_peft"))
    else:
        raise ValueError(f"task {name} not supported")


def preprocess_data_module(
    raw_dataset_dict, name: str, tokenizer, padding, max_length, num_proc: int = 8, args: Namespace = None
) -> hf_datasets.DatasetDict:
    """Preprocess for PTQ evaluation"""
    match name:
        case "wikitext2":
            return preprocess_data_module_wikitext2(
                raw_dataset_dict,
                tokenizer=tokenizer,
                max_length=max_length,
                num_proc=num_proc,
            )
        case "slim_pajama_6b":
            return preprocess_data_module_slim_pajama_6b(
                raw_dataset_dict,
                tokenizer=tokenizer,
                max_length=max_length,
                num_proc=num_proc,
            )
        # TODO
        case "gsm8k":
            return preprocess_data_module_gsm8k(
                raw_dataset_dict,
                tokenizer=tokenizer,
                args=args,
            )
        case _:
            raise ValueError(f"task {name} not supported")


def get_data_module(
    name: str,
    tokenizer,
    padding,
    max_length,
    num_workers: int = 8,
    num_raw_samples: int = None,
    args: Namespace = None,  # TODO: remove this when gsm8k is refactored and moved to get_data_module_for_peft
) -> hf_datasets.DatasetDict:
    """
    A data module refers to a dictionary of datasets with keys "train", "validation", and "test".

    Only `num_samples` examples are preprocessed, which saves time when profiling.
    """
    raw_data_module = get_raw_data_module(name)
    if num_raw_samples is not None:
        raw_data_module = hf_datasets.DatasetDict(
            **{
                split: raw_data_module[split].select(range(min(num_raw_samples, len(raw_data_module[split]))))
                for split in raw_data_module.keys()
            }
        )
    data_module = preprocess_data_module(
        raw_data_module,
        name,
        tokenizer=tokenizer,
        padding=padding,
        max_length=max_length,
        num_proc=num_workers,
        args=args,
    )
    return data_module


def preprocess_data_module_for_peft(
    raw_dataset_dict,
    name: str,
    tokenizer,
    pad_to_max_length: bool,
    max_length: int,
    num_proc: int,
    model_config,
    overwrite_cache: bool = False,
) -> hf_datasets.DatasetDict:
    """Preprocess for PEFT training/evaluation"""
    glue_tasks = [f"glue,{task}_peft" for task in GLUE_TASK_TO_KEYS.keys()]
    if name == "wikitext2_peft":
        return preprocess_data_module_wikitext2_peft(
            raw_dataset_dict,
            tokenizer=tokenizer,
            max_length=max_length,
            num_proc=num_proc,
            overwrite_cache=overwrite_cache,
        )
    elif name in ["slim_pajama_6b_peft", "slim_pajama_1b_peft", "slim_pajama_600m_peft", "slim_pajama_100m_peft"]:
        return preprocess_data_module_slimpajama_6b_peft(
            raw_dataset_dict,
            tokenizer=tokenizer,
            max_length=max_length,
            num_proc=num_proc,
            overwrite_cache=overwrite_cache,
        )
    elif name in glue_tasks:
        # "glue,subtask_peft"
        return preprocess_data_module_glue_peft(
            name.split(",")[1].removesuffix("_peft"),
            raw_dataset_dict,
            tokenizer=tokenizer,
            model_config=model_config,
            pad_to_max_length=pad_to_max_length,
            max_length=max_length,
            overwrite_cache=overwrite_cache,
        )
    elif name == "wikitext2_mlm":
        return preprocess_data_module_wikitext2_mlm(
            raw_dataset_dict,
            tokenizer=tokenizer,
            max_length=max_length,
            num_proc=num_proc,
            overwrite_cache=overwrite_cache,
        )
    else:
        raise ValueError(f"task {name} not supported")


def get_data_module_for_peft(
    name: str,
    tokenizer,
    model_config,
    pad_to_max_length: bool,
    max_length: int,
    num_workers: int,
    overwrite_cache: bool = False,
):
    raw_data_module = get_raw_data_module(name)
    data_module = preprocess_data_module_for_peft(
        raw_data_module,
        name=name,
        tokenizer=tokenizer,
        pad_to_max_length=pad_to_max_length,
        max_length=max_length,
        num_proc=num_workers,
        model_config=model_config,
        overwrite_cache=overwrite_cache,
    )
    return data_module
