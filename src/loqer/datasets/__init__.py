import logging
import datasets as hf_datasets
from argparse import Namespace

from .gsm8k import preprocess_data_module_gsm8k, get_raw_data_module_gsm8k
from .wikitext2 import get_raw_data_module_wikitext2, preprocess_data_module_wikitext2
from .slim_pajama import (
    get_raw_data_module_slim_pajama_6b,
    preprocess_data_module_slim_pajama_6b,
)

logger = logging.getLogger(__name__)


def get_raw_data_module(name: str) -> hf_datasets.DatasetDict:
    match name:
        case "wikitext2":
            return get_raw_data_module_wikitext2()
        case "slim_pajama_6b":
            return get_raw_data_module_slim_pajama_6b()
        case "gsm8k":
            return get_raw_data_module_gsm8k()
        case _:
            raise ValueError(f"task {name} not supported")


def preprocess_data_module(
    raw_dataset_dict, name: str, tokenizer, padding, max_length, num_proc: int = 8, args: Namespace = None
) -> hf_datasets.DatasetDict:
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
    args: Namespace = None,
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
