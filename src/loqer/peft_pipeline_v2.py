from argparse import ArgumentParser
from pathlib import Path
import logging
import yaml
from pprint import pformat
import shutil
import time
import types

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from peft import TaskType
from accelerate import dispatch_model

from loqer.datasets import get_data_module_for_peft
from loqer.models import find_layers_to_register_scale_hook
from loqer.statistic_profiler import register_scale_hooks, share_scales
from loqer.evaluate import evaluate_perplexity
from loqer.fine_tuning import (
    replace_lora_weights_loftq_4bit,
    replace_lora_weights_loqer_4bit,
    replace_lora_weights_loftq_2bit,
    replace_lora_weight_qlora_2bit,
    replace_lora_weight_loqer_2bit,
)
from loqer.utils import create_device_map

logger = logging.getLogger(__name__)


def adapt_and_save_clm_model(
    model_name_or_path: str,
    adapter_init: str,
    output_dir: str,
    loqer_calibration_set: str,
    loqer_num_calibration_samples: int,
    loqer_calibration_batch_size: int,
    loqer_max_seq_length: int,
    loftq_num_iters: int,
    bnb_quant_type: str,
    bnb_n_bits: int,
    lora_rank: int,
    lora_alpha: float,
    lora_target_modules: list[str],
    device_map: str,
    num_workers: int,
    overwrite_output_dir: bool,
    overwrite_dataset_cache: bool,
    loqer_scaling_mode: str,
    peek_post_init_metrics: bool,
):
    """
    Load a pretrained model, quantized it to 4-bit or 2-bit, initialize the lora adapter specified by `adapter_init`, save the base model and the initialized adapter

    - For 4-bit qLoRA, the bnb config and lora config are saved; adapters should be randomly initialized in the training script
    - For 2-bit qLoRA, the quantized base model and lora config are saved; adapters should be randomly initialized in the training script
    - For 2/4-bit LoftQ, both the quantized base model and the adapter are saved
    - For 2/4-bit Loqer, both the quantized base model and the adapter are saved
    """
    assert adapter_init in ["loftq", "loqer", "qlora", "lora"]
    assert bnb_n_bits in [2, 4]
    assert loqer_scaling_mode in ["diag", "rxx"]
    assert bnb_quant_type in ["nf", "fp"]

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite_output_dir:
            raise FileExistsError(f"Output directory {output_dir} already exists")
        else:
            logger.warning(f"⚠️ Output directory {output_dir} already exists and will be overwritten")
            shutil.rmtree(output_dir, ignore_errors=True)

    # LoQER calibration
    scale_dict = None
    calibration_dataloader = None
    if adapter_init == "loqer":
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, _attn_implementation="eager")
        model.eval()
        if "cuda" in device_map:
            model.to(device_map)
        else:
            if hasattr(model, "tie_weights"):
                model.tie_weights()
            device_map = create_device_map(model, device_map)
            model = dispatch_model(model, device_map)

        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        layers_to_register_and_share = find_layers_to_register_scale_hook(model)
        profiler_factory = register_scale_hooks(
            model, layers_to_register_and_share, mode=loqer_scaling_mode, torch_dtype=torch.float32
        )
        calibration_datamodule = get_data_module_for_peft(
            loqer_calibration_set,
            tokenizer=tokenizer,
            model_config=None,
            pad_to_max_length=None,
            max_length=loqer_max_seq_length,
            num_workers=num_workers,
            overwrite_cache=overwrite_dataset_cache,
        )
        calibration_dataloader = DataLoader(
            calibration_datamodule["train"],
            batch_size=loqer_calibration_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )
        start = time.time()
        profile_outputs = evaluate_perplexity(
            model,
            eval_dataloader=calibration_dataloader,
            num_samples=loqer_num_calibration_samples,
            progress_bar=True,
            description="Calibrating scales for LoQER+",
        )
        logger.info(f"Profiling outputs:\n{pformat(profile_outputs, sort_dicts=False)}")
        profiler_factory.remove_all_hooks()
        scale_dict = profiler_factory.get_scale_dict(progress_bar=True)
        calibration_time = time.time() - start
        share_scales(scale_dict, layers_to_register_and_share)
    elif peek_post_init_metrics:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        calibration_datamodule = get_data_module_for_peft(
            loqer_calibration_set,
            tokenizer=tokenizer,
            max_length=loqer_max_seq_length,
            num_workers=num_workers,
            overwrite_cache=overwrite_dataset_cache,
        )
        calibration_dataloader = DataLoader(
            calibration_datamodule["train"],
            batch_size=loqer_calibration_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

    # it seems LoftQ's NFQuantizer does not support double quantization
    bnb_config = None
    if adapter_init in ["qlora", "loftq", "loqer"] and bnb_n_bits == 4:
        bnb_quant_type_4bit = "nf4" if bnb_quant_type == "nf" else "fp4"
        bnb_4bit_use_double_quant = bnb_n_bits == 4
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_quant_type_4bit,
            # bnb_4bit_quant_storage=torch.bfloat16,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=bnb_config)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
        device_map = create_device_map(model, device_map)
        model = dispatch_model(model, device_map)
    model.eval()
    lora_target_modules_ = lora_target_modules
    # fmt: off
    if isinstance(lora_target_modules, (list, tuple)) and len(lora_target_modules) == 1 and lora_target_modules[0] == "all-linear":
        lora_target_modules_ = "all-linear"
    # fmt: on

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=lora_target_modules_,
        init_lora_weights=True,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.eval()

    error_dict = None
    elapsed = None
    bnb_quant_type_2bit = "normal" if bnb_quant_type == "nf" else "uniform"
    if adapter_init == "loftq":
        if bnb_n_bits == 4:
            start = time.time()
            error_dict = replace_lora_weights_loftq_4bit(peft_model, num_iters=loftq_num_iters)
            elapsed = time.time() - start
        elif bnb_n_bits == 2:
            start = time.time()
            error_dict = replace_lora_weights_loftq_2bit(
                peft_model, bnb_quant_type=bnb_quant_type_2bit, num_iters=loftq_num_iters
            )
            elapsed = time.time() - start
        else:
            raise NotImplementedError("LoftQ only supports 2-bit and 4-bit quantization")
    elif adapter_init == "loqer":
        if bnb_n_bits == 4:
            start = time.time()
            error_dict = replace_lora_weights_loqer_4bit(peft_model, scale_dict=scale_dict)
            elapsed = time.time() - start + calibration_time
        elif bnb_n_bits == 2:
            start = time.time()
            error_dict = replace_lora_weight_loqer_2bit(
                peft_model, bnb_quant_type=bnb_quant_type_2bit, scale_dict=scale_dict
            )
            elapsed = time.time() - start + calibration_time
        else:
            raise NotImplementedError("LoQER only supports 2-bit and 4-bit quantization")
    elif adapter_init == "qlora":
        if bnb_n_bits == 4:
            pass
        elif bnb_n_bits == 2:
            start = time.time()
            error_dict = replace_lora_weight_qlora_2bit(peft_model, bnb_quant_type=bnb_quant_type_2bit)
            elapsed = time.time() - start
        else:
            raise NotImplementedError("QLoRA only supports 2-bit and 4-bit quantization")
    elif adapter_init == "lora":
        pass
    else:
        raise ValueError(f"Invalid adapter init: {adapter_init}")

    post_init_ppl = None
    if peek_post_init_metrics:
        peft_model.eval()
        # peft_model.cuda()
        post_init_ppl = evaluate_perplexity(
            peft_model,
            eval_dataloader=calibration_dataloader,
            num_samples=loqer_num_calibration_samples,
            progress_bar=True,
            description=f"Evaluating post initialization ({adapter_init})",
        )
        logger.info(f"Post initialization perplexity ({adapter_init}):\n{pformat(post_init_ppl, sort_dicts=False)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if adapter_init in ["loftq", "loqer"]:
        peft_model.save_pretrained(output_dir / "adapter")
        logger.info(f"Adapter saved to {output_dir / 'adapter'}")
    else:
        # qlora, lora
        lora_config.save_pretrained(output_dir / "adapter")

    if adapter_init in ["loftq", "loqer"] or bnb_n_bits == 2:
        base_model = peft_model.unload()
        # save the bnb model as it is
        base_model.save_pretrained(output_dir / "base_model")
        logger.info(f"Base model saved to {output_dir / 'base_model'}")

    if bnb_config is not None:
        bnb_config_dict = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
            "bnb_4bit_quant_type": bnb_quant_type_4bit,
        }
        with open(output_dir / "bnb_config.yaml", "w") as f:
            yaml.safe_dump(bnb_config_dict, f)

    if elapsed is not None or error_dict is not None or post_init_ppl is not None:
        results = {"initialization_time": elapsed, "error_dict": error_dict, "post_init_ppl": post_init_ppl}
        with open(output_dir / "adapt_and_save_results.yaml", "w") as f:
            yaml.safe_dump(results, f)

    if elapsed is not None:
        logger.info(f"Adapter initialization ({adapter_init}) completed in {elapsed:.2f} seconds")


def adapt_and_save_cls_model(
    model_name_or_path: str,
    adapter_init: str,
    output_dir: str,
    loqer_calibration_set: str,
    loqer_num_calibration_samples: int,
    loqer_calibration_batch_size: int,
    loqer_max_seq_length: int,
    loftq_num_iters: int,
    bnb_quant_type: str,
    bnb_n_bits: int,
    lora_rank: int,
    lora_alpha: float,
    lora_target_modules: list[str],
    device_map: str,
    num_workers: int,
    overwrite_output_dir: bool,
    overwrite_dataset_cache: bool,
    loqer_scaling_mode: str,
    peek_post_init_metrics: bool,
    lora_modules_to_save: list[str],
):
    assert adapter_init in ["loftq", "loqer", "qlora", "lora"]
    assert bnb_n_bits in [2, 4]
    assert loqer_scaling_mode in ["diag", "rxx"]
    assert bnb_quant_type in ["nf", "fp"]
    PAD_TO_MAX_LENGTH = True

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite_output_dir:
            raise FileExistsError(f"Output directory {output_dir} already exists")
        else:
            logger.warning(f"⚠️ Output directory {output_dir} already exists and will be overwritten")
            shutil.rmtree(output_dir, ignore_errors=True)

    # LoQER calibration
    scale_dict = None
    calibration_dataloader = None
    output_ref = []
    if adapter_init == "loqer" or (peek_post_init_metrics and adapter_init != "lora"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, _attn_implementation="eager"
        )
        model.eval()

        if "cuda" in device_map:
            model.to(device_map)
        else:
            if hasattr(model, "tie_weights"):
                model.tie_weights()
            device_map = create_device_map(model, device_map)
            model = dispatch_model(model, device_map)

        data_collator = transformers.default_data_collator
        if adapter_init == "loqer":
            layers_to_register_and_share = find_layers_to_register_scale_hook(model)
            profiler_factory = register_scale_hooks(
                model, layers_to_register_and_share, mode=loqer_scaling_mode, torch_dtype=torch.float32
            )
        calibration_datamodule = get_data_module_for_peft(
            loqer_calibration_set,
            tokenizer=tokenizer,
            model_config=model.config,
            pad_to_max_length=PAD_TO_MAX_LENGTH,
            max_length=loqer_max_seq_length,
            num_workers=num_workers,
            overwrite_cache=overwrite_dataset_cache,
        )
        calibration_dataloader = DataLoader(
            calibration_datamodule["train"],
            batch_size=loqer_calibration_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )
        start = time.time()

        num_samples = 0
        input_device = next(model.parameters()).device
        for i, batch in enumerate(calibration_dataloader):
            with torch.no_grad():
                batch = {k: v.to(input_device) for k, v in batch.items()}
                outputs = model(**batch, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            output_ref.append(last_hidden_states.cpu())
            num_samples += loqer_calibration_batch_size
            if num_samples >= loqer_num_calibration_samples:
                break
        if adapter_init == "loqer":
            profiler_factory.remove_all_hooks()
            scale_dict = profiler_factory.get_scale_dict(progress_bar=True)
            share_scales(scale_dict, layers_to_register_and_share)
        calibration_time = time.time() - start

    bnb_config = None
    if adapter_init in ["qlora", "loftq", "loqer"] and bnb_n_bits == 4:
        bnb_4bit_use_double_quant = bnb_n_bits == 4
        bnb_quant_type_4bit = "nf4" if bnb_quant_type == "nf" else "fp4"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_quant_type_4bit,
            # bnb_4bit_quant_storage=torch.bfloat16,
            llm_int8_skip_modules=lora_modules_to_save,  # https://github.com/huggingface/peft/issues/1720
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, quantization_config=bnb_config
        )
        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        if "cuda" in device_map:
            model.to(device_map)
        else:
            device_map = create_device_map(model, device_map)
            model = dispatch_model(model, device_map)
    model.eval()
    lora_target_modules_ = lora_target_modules
    # fmt: off
    if isinstance(lora_target_modules, (list, tuple)) and len(lora_target_modules) == 1 and lora_target_modules[0] == "all-linear":
        lora_target_modules_ = "all-linear"
    # fmt: on
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=True,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=lora_target_modules_,
        init_lora_weights=True,
        modules_to_save=lora_modules_to_save,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.eval()

    error_dict = None
    elapsed = None
    bnb_quant_type_2bit = "normal" if bnb_quant_type == "nf" else "uniform"
    if adapter_init == "loftq":
        if bnb_n_bits == 4:
            start = time.time()
            error_dict = replace_lora_weights_loftq_4bit(peft_model, num_iters=loftq_num_iters)
            elapsed = time.time() - start
        elif bnb_n_bits == 2:
            start = time.time()
            error_dict = replace_lora_weights_loftq_2bit(
                peft_model, bnb_quant_type=bnb_quant_type_2bit, num_iters=loftq_num_iters
            )
            elapsed = time.time() - start
        else:
            raise NotImplementedError("LoftQ only supports 2-bit and 4-bit quantization")
    elif adapter_init == "loqer":
        if bnb_n_bits == 4:
            start = time.time()
            error_dict = replace_lora_weights_loqer_4bit(peft_model, scale_dict=scale_dict)
            elapsed = time.time() - start + calibration_time
        elif bnb_n_bits == 2:
            start = time.time()
            error_dict = replace_lora_weight_loqer_2bit(
                peft_model, bnb_quant_type=bnb_quant_type_2bit, scale_dict=scale_dict
            )
            elapsed = time.time() - start + calibration_time
        else:
            raise NotImplementedError("LoQER only supports 2-bit and 4-bit quantization")
    elif adapter_init == "qlora":
        if bnb_n_bits == 4:
            pass
        elif bnb_n_bits == 2:
            start = time.time()
            error_dict = replace_lora_weight_qlora_2bit(peft_model, bnb_quant_type=bnb_quant_type_2bit)
            elapsed = time.time() - start
        else:
            raise NotImplementedError("QLoRA only supports 2-bit and 4-bit quantization")
    elif adapter_init == "lora":
        pass
    else:
        raise ValueError(f"Invalid adapter init: {adapter_init}")

    post_init_error = None
    if peek_post_init_metrics and adapter_init != "lora":
        num_samples = 0
        peft_model.eval()
        post_init_outputs = []
        input_device = next(peft_model.parameters()).device
        for i, batch in enumerate(calibration_dataloader):
            batch = {k: v.to(input_device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = peft_model(**batch, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            post_init_outputs.append(last_hidden_states.cpu())
            num_samples += loqer_calibration_batch_size
            if num_samples >= loqer_num_calibration_samples:
                break
        errors = []
        for ref, post in zip(output_ref, post_init_outputs):
            errors.append((ref.cuda() - post.cuda()).abs().mean().cpu().item())
        post_init_error = sum(errors) / len(errors)

    output_dir.mkdir(parents=True, exist_ok=True)
    if adapter_init in ["loftq", "loqer"]:
        peft_model.save_pretrained(output_dir / "adapter")
        logger.info(f"Adapter saved to {output_dir / 'adapter'}")
    else:
        # qlora, lora
        lora_config.save_pretrained(output_dir / "adapter")

    if adapter_init in ["loftq", "loqer"] or (bnb_n_bits == 2 and adapter_init != "lora"):
        base_model = peft_model.unload()
        # save the bnb model as it is
        base_model.save_pretrained(output_dir / "base_model")
        logger.info(f"Base model saved to {output_dir / 'base_model'}")

    if bnb_config is not None:
        bnb_config_dict = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
            "bnb_4bit_quant_type": bnb_quant_type_4bit,
        }
        if lora_modules_to_save is not None:
            bnb_config_dict["llm_int8_skip_modules"] = lora_modules_to_save
        with open(output_dir / "bnb_config.yaml", "w") as f:
            yaml.safe_dump(bnb_config_dict, f)

    if elapsed is not None or error_dict is not None or post_init_error is not None:
        results = {"initialization_time": elapsed, "error_dict": error_dict, "post_init_error": post_init_error}
        with open(output_dir / "adapt_and_save_results.yaml", "w") as f:
            yaml.safe_dump(results, f)

        results.pop("error_dict")
        logger.info(f"Adapter initialization ({adapter_init}) completed:\n{results}")
    else:
        logger.info(f"Adapter initialization ({adapter_init}) completed")


def adapt_and_save_pipeline():
    parser = ArgumentParser()
    parser.add_argument("model_type", type=str, choices=["clm", "cls"], help="Model type: clm or cls")
    parser.add_argument("model_name_or_path", type=str)
    parser.add_argument("adapter_init", type=str, choices=["loftq", "loqer", "qlora", "lora"])
    parser.add_argument("output_dir", type=str)
    parser.add_argument(
        "--loqer-calibration-set", type=str, default=None, help="Default: wikitext2_peft for clm, required for cls"
    )
    parser.add_argument("--loqer-num-calibration-samples", type=int, default=128)
    parser.add_argument("--loqer-calibration-batch-size", type=int, default=2)
    parser.add_argument("--loqer-max-seq-length", type=int, default=2048)
    parser.add_argument("--loqer-scaling-mode", type=str, default="diag", help="Default: diag", choices=["diag", "rxx"])
    parser.add_argument("--loftq-num-iters", type=int, default=1, help="Default: 1")
    parser.add_argument(
        "--bnb-quant-type",
        type=str,
        default="fp",
        choices=["nf", "fp"],
        help="quantization type for the frozen weights. 'nf' means NormalFloat and 'fp' means FloatingPoint",
    )
    parser.add_argument("--bnb-n-bits", type=int, default=4, help="Default: 4", choices=[2, 4])
    parser.add_argument("--lora-rank", type=int, default=64, help="Default: 64")
    parser.add_argument("--lora-alpha", type=float, default=128.0, help="Default: 128.0")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=None,
        help="Default: all linear layers except the output layer",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument("--device-map", type=str, default="cuda", help="Default: cuda")
    parser.add_argument("--num-workers", type=int, default=8, help="Default: 8")
    parser.add_argument("--overwrite-output-dir", "-ow", dest="overwrite_output_dir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-dataset-cache", action="store_true")
    parser.add_argument("--peek-post-init-metrics", action="store_true", default=False)
    args = parser.parse_args()
    logger.info(f"Arguments\n{pformat(vars(args), sort_dicts=True)}")
    transformers.set_seed(args.seed)

    if args.model_type == "clm":
        if args.lora_target_modules is None:
            args.lora_target_modules = "all-linear"
            logger.warning(f" ⚠️Defaulting lora_target_modules to {args.lora_target_modules}")
        adapt_and_save_clm_model(
            args.model_name_or_path,
            adapter_init=args.adapter_init,
            output_dir=args.output_dir,
            loqer_calibration_set=args.loqer_calibration_set,
            loqer_num_calibration_samples=args.loqer_num_calibration_samples,
            loqer_calibration_batch_size=args.loqer_calibration_batch_size,
            loqer_max_seq_length=args.loqer_max_seq_length,
            loftq_num_iters=args.loftq_num_iters,
            bnb_quant_type=args.bnb_quant_type,
            bnb_n_bits=args.bnb_n_bits,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_target_modules,
            device_map=args.device_map,
            num_workers=args.num_workers,
            overwrite_output_dir=args.overwrite_output_dir,
            overwrite_dataset_cache=args.overwrite_dataset_cache,
            loqer_scaling_mode=args.loqer_scaling_mode,
            peek_post_init_metrics=args.peek_post_init_metrics,
        )
    elif args.model_type == "cls":
        if args.lora_target_modules is None:
            args.lora_target_modules = ["key_proj", "query_proj", "value_proj", "dense"]
            logger.warning(f"⚠️ Using default lora_target_modules: {args.lora_target_modules}")
        if args.lora_modules_to_save is None:
            args.lora_modules_to_save = ["pooler.dense", "classifier"]
            logger.warning(f"⚠️ Using default lora_modules_to_save: {args.lora_modules_to_save}")
        adapt_and_save_cls_model(
            args.model_name_or_path,
            adapter_init=args.adapter_init,
            output_dir=args.output_dir,
            loqer_calibration_set=args.loqer_calibration_set,
            loqer_num_calibration_samples=args.loqer_num_calibration_samples,
            loqer_calibration_batch_size=args.loqer_calibration_batch_size,
            loqer_max_seq_length=args.loqer_max_seq_length,
            loftq_num_iters=args.loftq_num_iters,
            bnb_quant_type=args.bnb_quant_type,
            bnb_n_bits=args.bnb_n_bits,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_target_modules,
            lora_modules_to_save=args.lora_modules_to_save,
            device_map=args.device_map,
            num_workers=args.num_workers,
            overwrite_output_dir=args.overwrite_output_dir,
            overwrite_dataset_cache=args.overwrite_dataset_cache,
            loqer_scaling_mode=args.loqer_scaling_mode,
            peek_post_init_metrics=args.peek_post_init_metrics,
        )

    args_dict = vars(args)
    with open(Path(args.output_dir) / "adapt_and_save_args.yaml", "w") as f:
        yaml.safe_dump(args_dict, f)
