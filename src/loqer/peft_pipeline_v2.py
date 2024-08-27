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
    replace_lora_weights_loftq_kbit,
    replace_lora_weight_qlora_kbit,
    replace_lora_weight_loqer_kbit,
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
    quant_type: str,
    quant_bits: int,
    lora_rank: int,
    lora_alpha: float,
    lora_target_modules: list[str] | None,
    device_map: str,
    num_workers: int,
    overwrite_output_dir: bool,
    overwrite_dataset_cache: bool,
    loqer_scaling_mode: str,
    peek_post_init_metrics: bool,
    lora_modules_to_save: list[str] | None,  # lm_head will not be quantized
    mxint_block_size: int,
):
    """
    Apply Lora or qLoRA to a causal language model and save the base model & adapted model to disk.
    """
    assert adapter_init in ["loftq", "loqer", "qlora", "lora"]
    assert loqer_scaling_mode in ["diag", "rxx"]
    assert quant_type in ["nf", "fp", "mxint"]
    if quant_type in ["nf", "fp"]:
        assert quant_bits in [2, 4]

    if lora_target_modules is None:
        lora_target_modules = "all-linear"
        logger.warning(
            f" ⚠️ Defaulting lora_target_modules to {lora_target_modules}, which automatically selects all linear layers except for lm_head"
        )
    if lora_modules_to_save is None:
        logger.warning(f" ⚠️ Defaulting lora_modules_to_save to 'None'. LM head will not be quantized.")

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
    if adapter_init == "loqer" or (peek_post_init_metrics and adapter_init != "lora"):
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

        # data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        data_collator = transformers.default_data_collator
        if adapter_init == "loqer":
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
            description="Pretrained model profiling",
        )
        logger.info(f"Profiling outputs:\n{pformat(profile_outputs, sort_dicts=False)}")
        if adapter_init == "loqer":
            profiler_factory.remove_all_hooks()
            scale_dict = profiler_factory.get_scale_dict(progress_bar=True)
            share_scales(scale_dict, layers_to_register_and_share)
        calibration_time = time.time() - start

    # it seems LoftQ's NFQuantizer does not support double quantization
    bnb_config = None
    if adapter_init in ["qlora", "loftq", "loqer"] and (quant_type in ["nf", "fp"] and quant_bits == 4):
        bnb_4bit_use_double_quant = quant_bits == 4
        bnb_quant_type_4bit = "nf4" if quant_type == "nf" else "fp4"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_quant_type_4bit,
            bnb_4bit_quant_storage=torch.bfloat16,  # !: uint8 will not work with qLoRA + FSDP
            # bnb_4bit_quant_storage=torch.uint8,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=bnb_config)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
        if "cuda" in device_map:
            model = model.to(device_map)
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
        task_type=TaskType.CAUSAL_LM,
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
    if adapter_init == "loftq":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            start = time.time()
            error_dict = replace_lora_weights_loftq_4bit(peft_model, num_iters=loftq_num_iters)
            elapsed = time.time() - start
        else:
            start = time.time()
            error_dict = replace_lora_weights_loftq_kbit(
                peft_model,
                quant_type=quant_type,
                num_bits=quant_bits,
                num_iters=loftq_num_iters,
                mxint_block_size=mxint_block_size,
            )
            elapsed = time.time() - start
    elif adapter_init == "loqer":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            start = time.time()
            error_dict = replace_lora_weights_loqer_4bit(peft_model, scale_dict=scale_dict)
            elapsed = time.time() - start + calibration_time
        else:
            start = time.time()
            error_dict = replace_lora_weight_loqer_kbit(
                peft_model,
                scale_dict=scale_dict,
                quant_type=quant_type,
                num_bits=quant_bits,
                mxint_block_size=mxint_block_size,
            )
            elapsed = time.time() - start + calibration_time
    elif adapter_init == "qlora":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            pass
        else:
            start = time.time()
            error_dict = replace_lora_weight_qlora_kbit(
                peft_model, quant_type=quant_type, num_bits=quant_bits, mxint_block_size=mxint_block_size
            )
            elapsed = time.time() - start
    elif adapter_init == "lora":
        pass
    else:
        raise ValueError(f"Invalid adapter init: {adapter_init}")

    post_init_ppl = None
    if peek_post_init_metrics and adapter_init != "lora":
        peft_model.eval()
        post_init_ppl = evaluate_perplexity(
            peft_model,
            eval_dataloader=calibration_dataloader,
            num_samples=loqer_num_calibration_samples,
            progress_bar=True,
            description=f"Evaluating post initialization ({adapter_init})",
        )
        logger.info(f"Post initialization perplexity ({adapter_init}):\n{pformat(post_init_ppl, sort_dicts=False)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(output_dir / "adapter")
    logger.info(f"Adapter saved to {output_dir / 'adapter'}")

    base_model = peft_model.unload()
    base_model.save_pretrained(output_dir / "base_model")
    logger.info(f"Base model saved to {output_dir / 'base_model'}")

    if elapsed is not None or error_dict is not None or post_init_ppl is not None:
        results = {"initialization_time": elapsed, "error_dict": error_dict, "post_init_ppl": post_init_ppl}
        with open(output_dir / "adapt_and_save_results.yaml", "w") as f:
            yaml.safe_dump(results, f)
        results.pop("error_dict")
        logger.info(f"Adapter initialization ({adapter_init}) completed:\n{results}")
    else:
        logger.info(f"Adapter initialization ({adapter_init}) completed")


def adapt_and_save_cls_model(
    model_name_or_path: str,
    adapter_init: str,
    output_dir: str,
    loqer_calibration_set: str,
    loqer_calibration_set_type: str,
    loqer_num_calibration_samples: int,
    loqer_calibration_batch_size: int,
    loqer_max_seq_length: int,
    loftq_num_iters: int,
    quant_type: str,
    quant_bits: int,
    lora_rank: int,
    lora_alpha: float,
    lora_target_modules: list[str] | None,
    device_map: str,
    num_workers: int,
    overwrite_output_dir: bool,
    overwrite_dataset_cache: bool,
    loqer_scaling_mode: str,
    peek_post_init_metrics: bool,
    lora_modules_to_save: list[str] | None,
    mxint_block_size: int,
    num_labels: int,
):
    assert adapter_init in ["loftq", "loqer", "qlora", "lora"]
    assert loqer_scaling_mode in ["diag", "rxx"]
    assert quant_type in ["nf", "fp", "mxint"]
    if quant_type in ["nf", "fp"]:
        assert quant_bits in [2, 4]
    assert loqer_calibration_set_type in ["downstream", "pretrain"]
    PAD_TO_MAX_LENGTH = True
    MLM_PROBABILITY = 0.15

    if lora_target_modules is None:
        if "deberta" in model_name_or_path.lower():
            lora_target_modules = ["key_proj", "query_proj", "value_proj", "dense"]
        elif "roberta" in model_name_or_path.lower():
            lora_target_modules = r"roberta\.encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|(attention\.output\.dense)|(intermediate\.dense)|(output\.dense))"
        else:
            raise ValueError(f"Cannot determine default modules to save for {model_name_or_path}")

        logger.info(f"🔍 Using default lora_target_modules: {lora_target_modules}")

    if lora_modules_to_save is None:
        if "deberta" in model_name_or_path.lower():
            lora_modules_to_save = ["pooler.dense", "classifier"]
        elif "roberta" in model_name_or_path.lower():
            lora_modules_to_save = ["classifier"]
        else:
            raise ValueError(f"Cannot determine default modules to save for {model_name_or_path}")

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

        if loqer_calibration_set_type == "downstream":
            data_collator = transformers.default_data_collator
        else:
            data_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
            )
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
                batch = {k: v.to(input_device) for k, v in batch.items() if k != "labels"}
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
    if adapter_init in ["qlora", "loftq", "loqer"] and (quant_type in ["nf", "fp"] and quant_bits == 4):
        bnb_4bit_use_double_quant = quant_bits == 4
        bnb_quant_type_4bit = "nf4" if quant_type == "nf" else "fp4"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_quant_type_4bit,
            bnb_4bit_quant_storage=torch.bfloat16,
            llm_int8_skip_modules=lora_modules_to_save,  # https://github.com/huggingface/peft/issues/1720
        )
        # this num_labels is just a placeholder, it will be overwritten by the actual number of labels in the dataset
        # !: don't set num_labels to 2, or the floating-point classifier will be mis-recognized as bnb weight by PeftModel.from_pretrained(..., ignore_mismatched_sizes=True)
        # !: then ignore_mismatched_sizes=True will not work
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, quantization_config=bnb_config, num_labels=num_labels
        )
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
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
    if adapter_init == "loftq":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            start = time.time()
            error_dict = replace_lora_weights_loftq_4bit(peft_model, num_iters=loftq_num_iters)
            elapsed = time.time() - start
        else:
            start = time.time()
            error_dict = replace_lora_weights_loftq_kbit(
                peft_model,
                quant_type=quant_type,
                num_bits=quant_bits,
                num_iters=loftq_num_iters,
                mxint_block_size=mxint_block_size,
            )
            elapsed = time.time() - start
    elif adapter_init == "loqer":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            start = time.time()
            error_dict = replace_lora_weights_loqer_4bit(peft_model, scale_dict=scale_dict)
            elapsed = time.time() - start + calibration_time
        else:
            start = time.time()
            error_dict = replace_lora_weight_loqer_kbit(
                peft_model,
                scale_dict=scale_dict,
                quant_type=quant_type,
                num_bits=quant_bits,
                mxint_block_size=mxint_block_size,
            )
            elapsed = time.time() - start + calibration_time
    elif adapter_init == "qlora":
        if quant_bits == 4 and quant_type in ["nf", "fp"]:
            pass
        else:
            start = time.time()
            error_dict = replace_lora_weight_qlora_kbit(
                peft_model, quant_type=quant_type, num_bits=quant_bits, mxint_block_size=mxint_block_size
            )
            elapsed = time.time() - start
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
            batch = {k: v.to(input_device) for k, v in batch.items() if k != "labels"}
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
    peft_model.save_pretrained(output_dir / "adapter")
    logger.info(f"Adapter saved to {output_dir / 'adapter'}")

    base_model = peft_model.unload()
    base_model.save_pretrained(output_dir / "base_model")
    logger.info(f"Base model saved to {output_dir / 'base_model'}")

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
    parser.add_argument(
        "--loqer-calibration-set-type",
        type=str,
        default="downstream",
        help="Default: downstream, required for cls",
        choices=["downstream", "pretrain"],
    )
    parser.add_argument("--loqer-num-calibration-samples", type=int, default=128)
    parser.add_argument("--loqer-calibration-batch-size", type=int, default=2)
    parser.add_argument("--loqer-max-seq-length", type=int, default=2048)
    parser.add_argument("--loqer-scaling-mode", type=str, default="diag", help="Default: diag", choices=["diag", "rxx"])
    parser.add_argument("--loftq-num-iters", type=int, default=1, help="Default: 1")
    parser.add_argument(
        "--quant-type",
        type=str,
        default="fp",
        choices=["nf", "fp", "mxint"],
        help="quantization type for the frozen weights. 'nf' means NormalFloat and 'fp' means FloatingPoint",
    )
    parser.add_argument("--quant-bits", type=int, default=4, help="Default: 4", choices=[2, 3, 4])
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
    parser.add_argument("--mxint-block-size", type=int, default=32)
    parser.add_argument("--num-labels", type=int, default=2)
    args = parser.parse_args()
    logger.info(f"Arguments\n{pformat(vars(args), sort_dicts=True)}")
    transformers.set_seed(args.seed)

    if args.model_type == "clm":
        adapt_and_save_clm_model(
            args.model_name_or_path,
            adapter_init=args.adapter_init,
            output_dir=args.output_dir,
            loqer_calibration_set=args.loqer_calibration_set,
            loqer_num_calibration_samples=args.loqer_num_calibration_samples,
            loqer_calibration_batch_size=args.loqer_calibration_batch_size,
            loqer_max_seq_length=args.loqer_max_seq_length,
            loftq_num_iters=args.loftq_num_iters,
            quant_type=args.quant_type,
            quant_bits=args.quant_bits,
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
            mxint_block_size=args.mxint_block_size,
        )
    elif args.model_type == "cls":
        adapt_and_save_cls_model(
            args.model_name_or_path,
            adapter_init=args.adapter_init,
            output_dir=args.output_dir,
            loqer_calibration_set=args.loqer_calibration_set,
            loqer_calibration_set_type=args.loqer_calibration_set_type,
            loqer_num_calibration_samples=args.loqer_num_calibration_samples,
            loqer_calibration_batch_size=args.loqer_calibration_batch_size,
            loqer_max_seq_length=args.loqer_max_seq_length,
            loftq_num_iters=args.loftq_num_iters,
            quant_type=args.quant_type,
            quant_bits=args.quant_bits,
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
            mxint_block_size=args.mxint_block_size,
            num_labels=args.num_labels,
        )

    args_dict = vars(args)
    with open(Path(args.output_dir) / "adapt_and_save_args.yaml", "w") as f:
        yaml.safe_dump(args_dict, f)
