from argparse import ArgumentParser
from pathlib import Path
import logging
import yaml
from pprint import pformat
import shutil
import time

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft import TaskType
from accelerate import dispatch_model

from loqer.datasets import get_data_module_for_peft
from loqer.models import find_layers_to_register_scale_hook
from loqer.statistic_profiler import register_scale_hooks, share_scales
from loqer.evaluate import evaluate_perplexity
from loqer.fine_tuning import (
    replace_lora_weights_loftq_4bit,
    replace_lora_weights_loqer_4bit,
    # replace_lora_weight_qlora_2bit,
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
):
    """
    We measured the perplexity after initialization before fine-tuning as sanity check.

    *: 4-bit:
        loftq, 4-bit
            1 iter:  ppl = 8.3574939161191
            2 iters: ppl = 8.352068320988458
            3 iters: ppl = 8.349460638357831
            4 iters: ppl = 8.348115802592645
            5 iters: ppl = 8.348127931271831

        loqer, 4-bit ppl = 8.310657814102111

    *: 2-bit:
        loftq, 2-bit
            1 iter:  ppl = 80.41665261056761
            2 iters: ppl = 137745.25699722476
            3 iters: ppl = 141260.56222336058
            4 iters: ppl = 143592.7901237662
            5 iters: ppl = 145648.63265452045

            Interestingly, for some layers, the approximation error of loftq decreases with more iterations, but the rest of the layers decrease first and then increase.
            For example, model.layers.7.mlp.gate_proj,    f-norm of approximation error over 5 iterations [35.5290412902832, 33.894344329833984, 33.31197738647461, 33.06462478637695, 32.95921325683594]
                         model.layers.3.self_attn.v_proj, f-norm of approximation error over 5 iterations [4.079835414886475, 3.8148927688598633, 3.803752899169922, 3.8422060012817383, 3.8932950496673584]

        loqer, 2-bit ppl = 8.886483023661881
    """
    assert adapter_init in ["loftq", "loqer", "qlora"]
    assert bnb_n_bits in [2, 4]
    assert loqer_scaling_mode in ["diag", "rxx"]

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite_output_dir:
            raise FileExistsError(f"Output directory {output_dir} already exists")
        else:
            logger.warning(f"⚠️ Output directory {output_dir} already exists and will be overwritten")
            shutil.rmtree(output_dir, ignore_errors=True)

    # LoQER calibration
    scale_dict = None
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
        calibration_time = time.time() - start
        logger.info(f"Profiling outputs:\n{pformat(profile_outputs, sort_dicts=False)}")
        profiler_factory.remove_all_hooks()
        scale_dict = profiler_factory.get_scale_dict(progress_bar=True)
        share_scales(scale_dict, layers_to_register_and_share)
    else:
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
    bnb_4bit_use_double_quant = bnb_n_bits == 4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_quant_type,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=bnb_config)
    model.eval()
    lora_target_modules_ = lora_target_modules
    # fmt: off
    if isinstance(lora_target_modules, (list, tuple)) and len(lora_target_modules) == 1 and lora_target_modules[0] == "all-linear":
        lora_target_modules_ = "all-linear"
    # fmt: on

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
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
    if adapter_init == "loftq":
        if bnb_n_bits == 4:
            start = time.time()
            error_dict = replace_lora_weights_loftq_4bit(peft_model, num_iters=loftq_num_iters)
            elapsed = time.time() - start
            # evaluate ppl
            post_init_ppl = evaluate_perplexity(
                peft_model,
                eval_dataloader=calibration_dataloader,
                num_samples=loqer_num_calibration_samples,
                progress_bar=True,
                description="Evaluating post initialization Loftq+",
            )
            logger.info(f"Post initialization perplexity (LoftQ):\n{pformat(post_init_ppl, sort_dicts=False)}")
        else:
            raise NotImplementedError("LoftQ only supports 4-bit quantization")
    elif adapter_init == "loqer":
        if bnb_n_bits == 4:
            start = time.time()
            error_dict = replace_lora_weights_loqer_4bit(peft_model, scale_dict=scale_dict)
            elapsed = time.time() - start + calibration_time
            # # evaluate ppl
            post_init_ppl = evaluate_perplexity(
                peft_model,
                eval_dataloader=calibration_dataloader,
                num_samples=loqer_num_calibration_samples,
                progress_bar=True,
                description="Evaluating post initialization LoQER+",
            )
            logger.info(f"Post initialization perplexity (LoQER+):\n{pformat(post_init_ppl, sort_dicts=False)}")
        else:
            raise NotImplementedError("LoQER only supports 4-bit quantization")
    elif adapter_init == "qlora":
        if bnb_n_bits == 4:
            post_init_ppl = evaluate_perplexity(
                peft_model,
                eval_dataloader=calibration_dataloader,
                num_samples=loqer_num_calibration_samples,
                progress_bar=True,
                description="Evaluating post initialization QLoRA",
            )
            logger.info(f"Post initialization perplexity (qLoRA):\n{pformat(post_init_ppl, sort_dicts=False)}")
            pass
        else:
            raise NotImplementedError("QLoRA only supports 4-bit quantization")
        # elif bnb_n_bits == 2:
        #     error_dict = replace_lora_weight_qlora_2bit(peft_model)
    else:
        raise ValueError(f"Invalid adapter init: {adapter_init}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if adapter_init in ["loftq", "loqer"]:
        peft_model.save_pretrained(output_dir / "adapter")
        logger.info(f"Adapter saved to {output_dir / 'adapter'}")
    else:
        # qlora
        lora_config.save_pretrained(output_dir / "adapter")

    if adapter_init in ["loftq", "loqer"] and bnb_n_bits == 4:
        base_model = peft_model.unload()
        # save the bnb model as it is
        base_model.save_pretrained(output_dir / "base_model")
        logger.info(f"BnB base model saved to {output_dir / 'base_model'}")

    bnb_config_dict = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
        "bnb_4bit_quant_type": bnb_quant_type,
    }
    with open(output_dir / "bnb_config.yaml", "w") as f:
        yaml.safe_dump(bnb_config_dict, f)

    if elapsed is not None or error_dict is not None:
        results = {"initialization_time": elapsed, "error_dict": error_dict}
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
):
    LOQER_SCALING_MODE = "diag"

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite_output_dir:
            raise FileExistsError(f"Output directory {output_dir} already exists")
        else:
            logger.warning(f"⚠️ Output directory {output_dir} already exists and will be overwritten")
            shutil.rmtree(output_dir, ignore_errors=True)

    raise NotImplementedError("adapt_and_save_cls_model is not implemented yet")
    # TODO


def adapt_and_save_pipeline():
    parser = ArgumentParser()
    parser.add_argument("model_type", type=str, choices=["clm", "cls"], help="Model type: clm or cls")
    parser.add_argument("model_name_or_path", type=str)
    parser.add_argument("adapter_init", type=str, choices=["loftq", "loqer", "qlora"])
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--loqer-calibration-set", type=str, default="wikitext2_peft", help="Default: wikitext2_peft")
    parser.add_argument("--loqer-num-calibration-samples", type=int, default=128)
    parser.add_argument("--loqer-calibration-batch-size", type=int, default=2)
    parser.add_argument("--loqer-max-seq-length", type=int, default=2048)
    parser.add_argument("--loqer-scaling-mode", type=str, default="diag", help="Default: diag", choices=["diag", "rxx"])
    parser.add_argument("--loftq-num-iters", type=int, default=1, help="Default: 1")
    parser.add_argument("--bnb-quant-type", type=str, default="fp4", help="Default: fp4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-n-bits", type=int, default=4, help="Default: 4", choices=[2, 4])
    parser.add_argument("--lora-rank", type=int, default=64, help="Default: 64")
    parser.add_argument("--lora-alpha", type=float, default=128.0, help="Default: 128.0")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default="all-linear",
        help="Default: all linear layers except the output layer",
    )
    parser.add_argument("--device-map", type=str, default="cuda", help="Default: cuda")
    parser.add_argument("--num-workers", type=int, default=8, help="Default: 8")
    parser.add_argument("--overwrite-output-dir", "-ow", dest="overwrite_output_dir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-dataset-cache", action="store_true")
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
        )
    elif args.model_type == "cls":
        adapt_and_save_cls_model(
            args.model_name_or_path,
            args.output_dir,
            args.adapter_init,
            args.loqer_calibration_set,
            args.loqer_num_calibration_samples,
            args.loqer_calibration_batch_size,
            args.loqer_max_seq_length,
            args.loftq_num_iters,
            args.bnb_quant_type,
            args.bnb_n_bits,
            args.lora_rank,
            args.lora_alpha,
            args.lora_target_modules,
            args.device_map,
            args.num_workers,
            args.overwrite_output_dir,
        )

    args_dict = vars(args)
    with open(Path(args.output_dir) / "adapt_and_save_args.yaml", "w") as f:
        yaml.safe_dump(args_dict, f)
