import logging
import re
import yaml
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pformat
import math
import datetime

from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import transformers
from accelerate import dispatch_model, init_empty_weights
import pandas as pd

from .fine_tuning import loftQ_parse_args, loftQ_fine_tuning
from .statistic_profiler import register_scale_hooks, share_scales
from .datasets import get_data_module
from .evaluate import evaluate_perplexity, evaluate_harness_downstream
from .models import find_layers_to_approximate, quantize_model, find_layers_to_register_scale_hook
from .approximate import compute_AB_and_approximation_error, attach_AB
from .utils import create_device_map, get_all_device_mem_info

logger = logging.getLogger(__name__)


def _mse_threshold_emoji(mse: float) -> str:
    warning_threshold = 1e-4
    error_threshold = 0.1

    if mse < warning_threshold:
        return "✅"
    elif mse < error_threshold:
        return "⚠️"
    else:
        return "❌"


def pipeline_loqer():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("--model-name", dest="model_name", type=str, help="Model name", default=None)
    parser.add_argument("--loqer-dtype", dest="loqer_dtype", type=str, help="Loqer data type", default=None)
    parser.add_argument("--eval-dtype", dest="eval_dtype", type=str, help="Evaluation data type", default=None)
    parser.add_argument("--device-map", dest="device_map", type=str, help="Device map", default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers", default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, help="Output directory", default=None)
    parser.add_argument("--AB-dict", dest="AB_dict", type=str, help="AB dict", default=None)
    parser.add_argument("--calibration-set", dest="calibration_set", type=str, help="Calibration set", default=None)
    parser.add_argument(
        "--num-calibration-samples",
        dest="num_calibration_samples",
        type=int,
        help="Number of calibration samples",
        default=None,
    )
    parser.add_argument(
        "--perplexity-eval-batch-size",
        dest="perplexity_eval_batch_size",
        type=int,
        help="Perplexity evaluation batch size",
        default=None,
    )
    parser.add_argument(
        "--perplexity-eval-set",
        dest="perplexity_eval_set",
        type=str,
        help="Perplexity evaluation set",
        default=None,
    )
    parser.add_argument(
        "--perplexity-max-seq-length",
        dest="perplexity_max_seq_length",
        type=int,
        help="Perplexity max sequence length",
        default=None,
    )
    parser.add_argument(
        "--lm-eval-tasks", dest="lm_eval_tasks", type=str, nargs="+", help="LM eval tasks", default=None
    )
    parser.add_argument(
        "--lm-eval-num-fewshot", dest="lm_eval_num_fewshot", type=int, help="LM eval num fewshot", default=None
    )
    parser.add_argument(
        "--lm-eval-batch-size", dest="lm_eval_batch_size", type=int, help="LM eval batch size", default=None
    )
    parser.add_argument(
        "--disable-loqer", dest="disable_loqer", action="store_true", help="Disable Loqer", default=None
    )
    parser.add_argument(
        "--loqer-scaling-mode",
        dest="loqer_scaling_mode",
        type=str,
        help="Loqer scaling mode, one of ['diagonal', 'diag', 'rxx', 'identity', 'mixed'].",
        default=None,
        choices=["diagonal", "diag", "rxx", "identity", "lqer", "mixed"],  # "diag" is alias of "diagonal"
    )
    parser.add_argument(
        "--loqer-sqrtm-implementation",
        dest="loqer_sqrtm_implementation",
        type=str,
        help="Loqer sqrtm implementation, one of ['blocked', 'iterative'].",
        default=None,
        choices=["blocked", "iterative"],
    )
    parser.add_argument(
        "--loqer-sqrtm-num-iters",
        dest="loqer_sqrtm_num_iters",
        type=int,
        help="Number of iterations for iterative sqrtm",
        default=None,
    )
    parser.add_argument("--disable-perplexity-eval", dest="disable_perplexity_eval", action="store_true", default=None)
    parser.add_argument("--disable-lm-eval", dest="disable_lm_eval", action="store_true", default=None)
    parser.add_argument("--overwrite-output-dir", "-ow", dest="overwrite_output_dir", action="store_true", default=None)

    args = parser.parse_args()
    args = vars(args)

    with open(args["config"], "r") as f:
        config = yaml.safe_load(f)

    override_args = {}
    args.pop("config")
    for entry, value in args.items():
        if value is not None:
            config[entry] = value
            override_args[entry] = value

    logger.info(f"Configuration: \n{pformat(config, indent=4)}")
    logger.info(f"Override arguments: \n{pformat(override_args, indent=4)}")

    model_name = config["model_name"]
    loqer_dtype = getattr(torch, config["loqer_dtype"])
    eval_dtype = getattr(torch, config["eval_dtype"])
    device_map = config["device_map"]
    num_workers = config["num_workers"]
    output_dir = Path(config["output_dir"]) if config["output_dir"] is not None else None
    AB_dict = config["AB_dict"]
    calibration_set = config["calibration_set"]
    num_calibration_samples = config["num_calibration_samples"]
    perplexity_evaluation_set = config["perplexity_eval_set"]
    perplexity_eval_batch_size = config["perplexity_eval_batch_size"]
    perplexity_max_seq_length = config["perplexity_max_seq_length"]
    lm_eval_tasks = config["lm_eval_tasks"]
    lm_eval_num_fewshot = config["lm_eval_num_fewshot"]
    lm_eval_batch_size = config["lm_eval_batch_size"]

    disable_loqer = config["disable_loqer"]
    loqer_scaling_mode = config["loqer_scaling_mode"]
    loqer_sqrtm_implementation = config["loqer_sqrtm_implementation"]
    loqer_sqrtm_num_iters = config["loqer_sqrtm_num_iters"]
    loqer_config = config["loqer_config"]
    disable_perplexity_eval = config["disable_perplexity_eval"]
    disable_lm_eval = config["disable_lm_eval"]
    loqer_scaling_mode_map = config["loqer_scaling_mode_map"]
    overwrite_output_dir = config["overwrite_output_dir"]

    # check output directory
    if overwrite_output_dir:
        logger.warning("⚠️ Overwriting output directory")
        AB_dict_in_output_dir = output_dir / "AB_dict.pt"
        if AB_dict_in_output_dir.is_file():
            if AB_dict is not None:
                if Path(AB_dict) != AB_dict_in_output_dir:
                    logger.warning(
                        f"⚠️ AB_dict is specified but not the same as the one in the output directory: {AB_dict_in_output_dir}. Use the specified {AB_dict}"
                    )
            else:
                AB_dict = AB_dict_in_output_dir
                logger.warning(f"🔊 Using AB_dict in the output directory: {AB_dict}")
    else:
        if output_dir is not None and output_dir.is_dir() and len(list(output_dir.iterdir())) > 0:
            raise ValueError(f"Output directory {output_dir} is not empty")

    # sqrtm_implementation
    if loqer_scaling_mode in ["rxx", "mixed"]:
        if loqer_sqrtm_implementation == "blocked":
            # refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
            logger.info("🔊 Using blocked sqrtm implementation. Only CPU + Scipy is supported")
        elif loqer_sqrtm_implementation == "iterative":
            # refer to https://link.springer.com/article/10.1023/A:1019150005407
            logger.info(f"🔊 Using iterative sqrtm implementation (number of iterations={loqer_sqrtm_num_iters})")
        else:
            raise ValueError(f"Unknown sqrtm_implementation: {loqer_sqrtm_implementation}")

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=loqer_dtype, _attn_implementation="eager"
    )
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map = create_device_map(model, device_map=device_map)
    logger.info(f"Device map: {device_map}")
    model = dispatch_model(model, device_map)
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if not disable_loqer and AB_dict is None:
        if loqer_scaling_mode == "identity":
            logger.info("🔊 Using identity scale (torch.eye)")
        logger.info("🚀 Running data calibration...")
        layers_to_register_and_share = find_layers_to_register_scale_hook(model)
        profiler_factory = register_scale_hooks(
            model,
            layers_to_register_and_share=layers_to_register_and_share,
            mode=loqer_scaling_mode,
            torch_dtype=loqer_dtype,
            mode_map=loqer_scaling_mode_map,
        )

        calibration_datamodule = get_data_module(
            name=calibration_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=20 * num_calibration_samples,
            num_workers=num_workers,
        )

        calibration_dataloader = DataLoader(
            calibration_datamodule["train"],
            batch_size=perplexity_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        mem_info = get_all_device_mem_info()
        logger.info(f"Device memory before profiling starts: \n{pformat(mem_info)}")
        profile_outputs = evaluate_perplexity(
            model=model,
            eval_dataloader=calibration_dataloader,
            num_samples=num_calibration_samples if loqer_scaling_mode != "identity" else perplexity_eval_batch_size,
            progress_bar=True,
            input_device=None,
            description="Calibrating",
        )

        profiler_factory.remove_all_hooks()
        if loqer_scaling_mode == "rxx":
            scale_dict = profiler_factory.get_scale_dict(
                progress_bar=True,
                sqrtm_implementation=loqer_sqrtm_implementation,
                sqrtm_num_iters=loqer_sqrtm_num_iters,
            )
        else:
            scale_dict = profiler_factory.get_scale_dict(progress_bar=True)

        share_scales(scale_dict, layers_to_register_and_share)
        logger.info(f"Perplexity after profiling: {profile_outputs['perplexity']:.4f}")

    logger.info("🚀 Quantizing model...")
    quantize_model(model, loqer_config)

    if not disable_loqer:
        layers_to_approximate = find_layers_to_approximate(model)
        if AB_dict is None:
            logger.info("🚀 Loqer is enabled. Computing A & B...")
            AB_dict, mse_df = compute_AB_and_approximation_error(model, layers_to_approximate, scale_dict, loqer_config)
            del scale_dict
            attach_AB(model, layers_to_approximate, AB_dict)
            mse_df_emoji = mse_df.copy()
            mse_df_emoji.loc[:, "mse?"] = mse_df["mse"].apply(_mse_threshold_emoji)
            logger.info(f"Approximation error (mean squared error): \n{mse_df_emoji.to_markdown()}")
        else:
            logger.info("🚀 Loqer is enabled and AB_dict is specified. Attaching A & B...")
            AB_dict = torch.load(AB_dict)
            mse_df = None
            attach_AB(model, layers_to_approximate, AB_dict)
    else:
        logger.warning("⚠️ Loqer is disabled, skipping layer approximation")
    # logger.info(f"Model after approximation: \n{model}")

    if not disable_perplexity_eval:
        logger.info("🚀 Evaluating perplexity...")
        eval_datamodule = get_data_module(
            name=perplexity_evaluation_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=None,
            num_workers=num_workers,
        )
        eval_dataloader = DataLoader(
            eval_datamodule["test"],
            batch_size=perplexity_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )
        model = model.to(eval_dtype)
        model = dispatch_model(model, device_map)
        ppl_results = evaluate_perplexity(
            model=model,
            eval_dataloader=eval_dataloader,
            num_samples=None,
            progress_bar=True,
            input_device=None,
            description="Evaluating",
        )

        if disable_loqer:
            logger.info(f"Perplexity after quantization (no LoQER): {ppl_results['perplexity']:.4f}")
        else:
            logger.info(f"Perplexity after approximation: {ppl_results['perplexity']:.4f}")

    if not disable_lm_eval:
        logger.info("🚀 Evaluating lm-eval downstream tasks...")
        model = model.to(eval_dtype)
        model = dispatch_model(model, device_map)
        lm_eval_results = evaluate_harness_downstream(
            model,
            tasks=lm_eval_tasks,
            num_fewshot=lm_eval_num_fewshot,
            no_cache=True,
            batch_size=lm_eval_batch_size,
        )
        logger.info(f"Downstream task results: \n{pformat(lm_eval_results)}")

    if output_dir is not None:
        logger.info(f"🚀 Saving results to {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        if not disable_loqer and mse_df is not None:
            # save approximation results
            mse_df.to_csv(output_dir / "approximation_error.csv", index=False)
            # save AB_dict
            AB_dict_path = output_dir / "AB_dict.pt"
            torch.save(AB_dict, AB_dict_path)
            config["AB_dict"] = AB_dict_path.resolve().as_posix()

        # save perplexity results
        if not disable_perplexity_eval:
            with open(output_dir / "perplexity_results.yaml", "w") as f:
                yaml.dump(ppl_results, f)

        # save lm-eval results
        if not disable_lm_eval:
            with open(output_dir / "lm_eval_results.yaml", "w") as f:
                yaml.dump(lm_eval_results, f)

        # save config
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)


def pipeline_fp16():
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("--dtype", dest="dtype", type=str, help="Evaluation data type", default="float16")
    parser.add_argument("--device-map", dest="device_map", type=str, help="Device map", default="auto-balanced")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers", default=8)
    parser.add_argument("--output-dir", dest="output_dir", type=str, help="Output directory", default=None)
    parser.add_argument(
        "--perplexity-eval-set",
        dest="perplexity_eval_set",
        type=str,
        help="Perplexity evaluation set",
        default="wikitext2",
    )
    parser.add_argument(
        "--perplexity-eval-batch-size",
        dest="perplexity_eval_batch_size",
        type=int,
        help="Perplexity evaluation batch size",
        default=4,
    )
    parser.add_argument(
        "--perplexity-max-seq-length",
        dest="perplexity_max_seq_length",
        type=int,
        help="Perplexity max sequence length",
        default=2048,
    )
    parser.add_argument(
        "--lm-eval-tasks",
        dest="lm_eval_tasks",
        type=str,
        nargs="+",
        help="LM eval tasks",
        default=[
            "arc_easy",
            "lambada_openai",
            "piqa",
            "winogrande",
            "arc_challenge",
            "boolq",
            "openbookqa",
        ],
    )
    parser.add_argument(
        "--lm-eval-num-fewshot", dest="lm_eval_num_fewshot", type=int, help="LM eval num fewshot", default=0
    )
    parser.add_argument(
        "--lm-eval-batch-size", dest="lm_eval_batch_size", type=int, help="LM eval batch size", default=16
    )
    parser.add_argument("--disable-perplexity-eval", dest="disable_perplexity_eval", action="store_true")
    parser.add_argument("--disable-lm-eval", dest="disable_lm_eval", action="store_true")

    args = parser.parse_args()

    logger.info(f"Arguments: \n{pformat(vars(args), indent=4)}")

    model_name = args.model_name
    dtype = getattr(torch, args.dtype)
    device_map = args.device_map
    num_workers = args.num_workers
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    perplexity_evaluation_set = args.perplexity_eval_set
    perplexity_eval_batch_size = args.perplexity_eval_batch_size
    perplexity_max_seq_length = args.perplexity_max_seq_length
    lm_eval_tasks = args.lm_eval_tasks
    lm_eval_num_fewshot = args.lm_eval_num_fewshot
    lm_eval_batch_size = args.lm_eval_batch_size
    disable_perplexity_eval = args.disable_perplexity_eval
    disable_lm_eval = args.disable_lm_eval

    # check output directory
    if output_dir is not None and output_dir.is_dir() and len(list(output_dir.iterdir())) > 0:
        raise ValueError(f"Output directory {output_dir} is not empty")

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if not disable_perplexity_eval:
        logger.info("🚀 Evaluating perplexity...")
        perplexity_datamodule = get_data_module(
            name=perplexity_evaluation_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=None,
            num_workers=num_workers,
        )
        perplexity_dataloader = DataLoader(
            perplexity_datamodule["test"],
            batch_size=perplexity_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        model = dispatch_model(model, device_map=create_device_map(model, device_map=device_map))
        perplexity_results = evaluate_perplexity(
            model=model,
            eval_dataloader=perplexity_dataloader,
            num_samples=None,
            progress_bar=True,
            input_device=None,
            description="Evaluating perplexity",
        )

        logger.info(f"Perplexity: {perplexity_results['perplexity']:.4f}")

    if not disable_lm_eval:
        logger.info("🚀 Evaluating lm-eval downstream tasks...")
        lm_eval_results = evaluate_harness_downstream(
            model,
            tasks=lm_eval_tasks,
            num_fewshot=lm_eval_num_fewshot,
            no_cache=True,
            batch_size=lm_eval_batch_size,
        )
        logger.info(f"Downstream task results: \n{pformat(lm_eval_results)}")

    if output_dir is not None:
        logger.info(f"🚀 Saving results to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # save perplexity results
        if not disable_perplexity_eval:
            with open(output_dir / "perplexity_results.yaml", "w") as f:
                yaml.dump(perplexity_results, f)

        # save lm-eval results
        if not disable_lm_eval:
            with open(output_dir / "lm_eval_results.yaml", "w") as f:
                yaml.dump(lm_eval_results, f)

        # save args
        with open(output_dir / "args.yaml", "w") as f:
            yaml.dump(vars(args), f)


def pipeline_q_baseline():
    from transformers import BitsAndBytesConfig, AwqConfig, GPTQConfig
    from auto_gptq import exllama_set_max_input_length
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", help="Load in 4-bit model")
    parser.add_argument("--dtype", dest="dtype", type=str, help="Evaluation data type", default="float16")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers", default=8)
    parser.add_argument("--output-dir", dest="output_dir", type=str, help="Output directory", default=None)
    parser.add_argument(
        "--perplexity-eval-set",
        dest="perplexity_eval_set",
        type=str,
        help="Perplexity evaluation set",
        default="wikitext2",
    )
    parser.add_argument(
        "--perplexity-eval-batch-size",
        dest="perplexity_eval_batch_size",
        type=int,
        help="Perplexity evaluation batch size",
        default=4,
    )
    parser.add_argument(
        "--perplexity-max-seq-length",
        dest="perplexity_max_seq_length",
        type=int,
        help="Perplexity max sequence length",
        default=2048,
    )
    parser.add_argument(
        "--lm-eval-tasks",
        dest="lm_eval_tasks",
        type=str,
        nargs="+",
        help="LM eval tasks",
        default=[
            "arc_easy",
            "lambada_openai",
            "piqa",
            "winogrande",
            "arc_challenge",
            "boolq",
            "openbookqa",
        ],
    )
    parser.add_argument(
        "--lm-eval-num-fewshot", dest="lm_eval_num_fewshot", type=int, help="LM eval num fewshot", default=0
    )
    parser.add_argument(
        "--lm-eval-batch-size", dest="lm_eval_batch_size", type=int, help="LM eval batch size", default=16
    )
    parser.add_argument("--disable-perplexity-eval", dest="disable_perplexity_eval", action="store_true")
    parser.add_argument("--disable-lm-eval", dest="disable_lm_eval", action="store_true")

    args = parser.parse_args()

    logger.info(f"Arguments: \n{pformat(vars(args), indent=4)}")

    model_name = args.model_name
    load_in_4bit = args.load_in_4bit
    dtype = getattr(torch, args.dtype)
    num_workers = args.num_workers
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    perplexity_evaluation_set = args.perplexity_eval_set
    perplexity_eval_batch_size = args.perplexity_eval_batch_size
    perplexity_max_seq_length = args.perplexity_max_seq_length
    lm_eval_tasks = args.lm_eval_tasks
    lm_eval_num_fewshot = args.lm_eval_num_fewshot
    lm_eval_batch_size = args.lm_eval_batch_size
    disable_perplexity_eval = args.disable_perplexity_eval
    disable_lm_eval = args.disable_lm_eval

    # check output directory
    if output_dir is not None and output_dir.is_dir() and len(list(output_dir.iterdir())) > 0:
        raise ValueError(f"Output directory {output_dir} is not empty")

    # Load model and tokenizer
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, quantization_config=quantization_config
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map={"": 0})
    model.eval()

    if load_in_4bit:
        q_method = "bnb-4bit"
    else:
        if isinstance(model.config.quantization_config, GPTQConfig):
            q_method = "gptq"
            model = exllama_set_max_input_length(model, 8192)
        elif isinstance(model.config.quantization_config, AwqConfig):
            q_method = "awq"
        elif isinstance(model.config.quantization_config, dict):
            q_method = model.config.quantization_config["quant_method"]
        else:
            raise ValueError(f"Unknown quantization method: {model.config.quantization_config}")

    logger.info(f"🔊 Quantization method: {q_method}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if not disable_perplexity_eval:
        logger.info("🚀 Evaluating perplexity...")
        perplexity_datamodule = get_data_module(
            name=perplexity_evaluation_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=None,
        )
        perplexity_dataloader = DataLoader(
            perplexity_datamodule["test"],
            batch_size=perplexity_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        perplexity_results = evaluate_perplexity(
            model=model,
            eval_dataloader=perplexity_dataloader,
            num_samples=None,
            progress_bar=True,
            input_device=None,
            description="Evaluating perplexity",
        )

        logger.info(f"Perplexity: {perplexity_results['perplexity']:.4f}")

    if not disable_lm_eval:
        logger.info("🚀 Evaluating lm-eval downstream tasks...")
        lm_eval_results = evaluate_harness_downstream(
            model,
            tasks=lm_eval_tasks,
            num_fewshot=lm_eval_num_fewshot,
            no_cache=True,
            batch_size=lm_eval_batch_size,
        )
        logger.info(f"Downstream task results: \n{pformat(lm_eval_results)}")

    if output_dir is not None:
        logger.info(f"🚀 Saving results to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # save perplexity results
        if not disable_perplexity_eval:
            with open(output_dir / "perplexity_results.yaml", "w") as f:
                yaml.dump(perplexity_results, f)

        # save lm-eval results
        if not disable_lm_eval:
            with open(output_dir / "lm_eval_results.yaml", "w") as f:
                yaml.dump(lm_eval_results, f)

        # save args
        with open(output_dir / "args.yaml", "w") as f:
            yaml.dump(vars(args), f)


def _check_chunk_id(model_name, layers_per_chunk, chunk_id=None):
    """
    Check if the chunk_id is valid for the given model and layers_per_chunk.
    """
    with init_empty_weights():
        config = transformers.AutoConfig.from_pretrained(model_name, _attn_implementation="eager")
        model = transformers.AutoModelForCausalLM.from_config(config)
        model_cls = model.__class__
        model = model_cls(config)
    layers_to_register_and_share = find_layers_to_register_scale_hook(model)

    num_chunks = math.ceil(len(layers_to_register_and_share) / layers_per_chunk)

    if chunk_id is not None:
        if chunk_id > num_chunks:
            logger.error(f"❌ chunk_id (={chunk_id}) must be smaller than the number of chunks ({num_chunks})")
            raise RuntimeError(f"chunk_id (={chunk_id}) must be smaller than the number of chunks ({num_chunks})")
    else:
        logger.info(f"Model name: {model_name}")
        logger.info(f"Layers per chunk: {layers_per_chunk}")
        logger.info(f"Allowed chunk IDs: [0, {num_chunks - 1}]")

    return num_chunks


def _verify_AB_dict_chunks(AB_dict_dir: Path, num_chunks: int, current_chunk_tag=None) -> set[str]:
    chunks_to_check = [f"{i}-of-{num_chunks-1}.pt" for i in range(num_chunks)]
    if current_chunk_tag is not None:
        chunks_to_check.remove(current_chunk_tag + ".pt")

    if AB_dict_dir.is_dir():
        existing_chunks = [f.name for f in AB_dict_dir.iterdir() if f.is_file()]
    else:
        existing_chunks = []
    missing_chunks = set(chunks_to_check) - set(existing_chunks)
    return missing_chunks


def pipeline_loqer_chunked():
    parser = ArgumentParser()
    fine_tuning_group = parser.add_argument_group("Fine_Tuning Configuration")
    loftQ_parse_args(fine_tuning_group, use_existing_parser=True)
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("--model-name", dest="model_name", type=str, help="Model name", default=None)
    parser.add_argument("--loqer-dtype", dest="loqer_dtype", type=str, help="Loqer data type", default=None)
    parser.add_argument("--eval-dtype", dest="eval_dtype", type=str, help="Evaluation data type", default=None)
    parser.add_argument("--device-map", dest="device_map", type=str, help="Device map", default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers", default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, help="Output directory", default=None)
    # parser.add_argument("--AB-dict", dest="AB_dict", type=str, help="AB dict", default=None)
    parser.add_argument("--calibration-set", dest="calibration_set", type=str, help="Calibration set", default=None)
    parser.add_argument(
        "--num-calibration-samples",
        dest="num_calibration_samples",
        type=int,
        help="Number of calibration samples",
        default=None,
    )
    parser.add_argument(
        "--perplexity-eval-batch-size",
        dest="perplexity_eval_batch_size",
        type=int,
        help="Perplexity evaluation batch size",
        default=None,
    )
    parser.add_argument(
        "--perplexity-eval-set",
        dest="perplexity_eval_set",
        type=str,
        help="Perplexity evaluation set",
        default=None,
    )
    parser.add_argument(
        "--perplexity-max-seq-length",
        dest="perplexity_max_seq_length",
        type=int,
        help="Perplexity max sequence length",
        default=None,
    )
    parser.add_argument(
        "--lm-eval-tasks", dest="lm_eval_tasks", type=str, nargs="+", help="LM eval tasks", default=None
    )
    parser.add_argument(
        "--lm-eval-num-fewshot", dest="lm_eval_num_fewshot", type=int, help="LM eval num fewshot", default=None
    )
    parser.add_argument(
        "--lm-eval-batch-size", dest="lm_eval_batch_size", type=int, help="LM eval batch size", default=None
    )
    parser.add_argument(
        "--disable-loqer", dest="disable_loqer", action="store_true", help="Disable Loqer", default=None
    )
    parser.add_argument(
        "--loqer-scaling-mode",
        dest="loqer_scaling_mode",
        type=str,
        help="Loqer scaling mode, one of ['diagonal', 'diag', 'rxx', 'identity', 'mixed', 'lqer'].",
        default=None,
        choices=["diagonal", "diag", "rxx", "identity", "mixed", "lqer"],  # "diag" is alias of "diagonal"
    )
    parser.add_argument(
        "--loqer-sqrtm-implementation",
        dest="loqer_sqrtm_implementation",
        type=str,
        help="Loqer sqrtm implementation, one of ['blocked', 'iterative'].",
        default=None,
        choices=["blocked", "iterative"],
    )
    parser.add_argument(
        "--loqer-sqrtm-num-iters",
        dest="loqer_sqrtm_num_iters",
        type=int,
        help="Number of iterations for iterative sqrtm",
        default=None,
    )
    parser.add_argument("--disable-perplexity-eval", dest="disable_perplexity_eval", action="store_true", default=None)
    parser.add_argument("--disable-lm-eval", dest="disable_lm_eval", action="store_true", default=None)
    parser.add_argument("--layers-per-chunk", dest="layers_per_chunk", type=int, help="Layers per chunk", default=None)
    parser.add_argument("--chunk-id", dest="chunk_id", type=int, help="Chunk ID", default=None)

    args = parser.parse_args()
    args = vars(args)

    with open(args["config"], "r") as f:
        config = yaml.safe_load(f)

    override_args = {}
    args.pop("config")
    for entry, value in args.items():
        if value is not None:
            config[entry] = value
            override_args[entry] = value

    # NOTE: Unlike above, fine-tuning yaml config file has higher priority than command line arguments... 
    # (because there are many default values in the command line arguments)
    if "fine_tuning_config" in config:
        fine_tuning_config = config.pop("fine_tuning_config")
        config.update(fine_tuning_config)
        args.update(fine_tuning_config)
        for entry, value in fine_tuning_config.items():
            override_args.pop(entry, None)

    logger.info(f"Configuration: \n{pformat(config, indent=4)}")
    logger.info(f"Override arguments: \n{pformat(override_args, indent=4)}")

    model_name = config["model_name"]
    loqer_dtype = getattr(torch, config["loqer_dtype"])
    eval_dtype = getattr(torch, config["eval_dtype"])
    device_map = config["device_map"]
    num_workers = config["num_workers"]
    output_dir = Path(config["output_dir"]) if config["output_dir"] is not None else None
    # AB_dict = config["AB_dict"]
    calibration_set = config["calibration_set"]
    num_calibration_samples = config["num_calibration_samples"]
    perplexity_evaluation_set = config["perplexity_eval_set"]
    perplexity_eval_batch_size = config["perplexity_eval_batch_size"]
    perplexity_max_seq_length = config["perplexity_max_seq_length"]
    lm_eval_tasks = config["lm_eval_tasks"]
    lm_eval_num_fewshot = config["lm_eval_num_fewshot"]
    lm_eval_batch_size = config["lm_eval_batch_size"]

    disable_loqer = config["disable_loqer"]
    loqer_scaling_mode = config["loqer_scaling_mode"]
    loqer_sqrtm_implementation = config["loqer_sqrtm_implementation"]
    loqer_sqrtm_num_iters = config["loqer_sqrtm_num_iters"]
    loqer_config = config["loqer_config"]
    disable_perplexity_eval = config["disable_perplexity_eval"]
    disable_lm_eval = config["disable_lm_eval"]
    loqer_scaling_mode_map = config["loqer_scaling_mode_map"]

    layers_per_chunk = config["layers_per_chunk"]
    chunk_id = config["chunk_id"]
    # Lora fine-tuning
    fine_tuning = config.get("fine_tuning", False)


    # assert chunk_id is not None
    assert output_dir is not None

    num_chunks = _check_chunk_id(model_name, layers_per_chunk, chunk_id)
    chunk_tag = f"{chunk_id}-of-{num_chunks-1}"

    # check output directory
    AB_dict_dir = output_dir.joinpath("AB_dict")
    missing_chunks = _verify_AB_dict_chunks(AB_dict_dir=AB_dict_dir, num_chunks=num_chunks, current_chunk_tag=None)
    assert not (len(missing_chunks) > 0 and chunk_id is None), f"Missing chunks: {missing_chunks}"

    if len(missing_chunks) > 0:
        # only allows disable_loqer=False and loqer_scaling_mode in ["diag", "diagonal", "rxx", "mixed", "identity"]
        if disable_loqer:
            raise ValueError("disable_loqer=True is not supported for chunked pipeline.")
        else:
            if loqer_scaling_mode not in ["diag", "diagonal", "rxx", "mixed", "identity", "lqer"]:
                raise ValueError("loqer_scaling_mode should be one of ['diagonal', 'diag', 'rxx', 'mixed', 'identity', 'lqer']")

        # sqrtm_implementation
        if loqer_scaling_mode in ["rxx", "mixed"]:
            if loqer_sqrtm_implementation == "blocked":
                # refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
                logger.info("🔊 Using blocked sqrtm implementation. Only CPU + Scipy is supported")
            elif loqer_sqrtm_implementation == "iterative":
                # refer to https://link.springer.com/article/10.1023/A:1019150005407
                logger.info(f"🔊 Using iterative sqrtm implementation (number of iterations={loqer_sqrtm_num_iters})")
            else:
                raise ValueError(f"Unknown sqrtm_implementation: {loqer_sqrtm_implementation}")

        # Load model and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=loqer_dtype, _attn_implementation="eager"
        )
        model.eval()
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        device_map = create_device_map(model, device_map=device_map)
        logger.info(f"Device map: {device_map}")
        model = dispatch_model(model, device_map)
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # solve chunk_id
        layers_to_register_and_share = find_layers_to_register_scale_hook(model)
        layers_to_register_and_share = layers_to_register_and_share[chunk_id::num_chunks]
        logger.info(
            f"🔊 Chunk id = {chunk_id}, total number of chunks = {num_chunks}, layers included in this chunk:\n{pformat(list(map(lambda x: x['target_layer'], layers_to_register_and_share)))}"
        )

        profiler_factory = register_scale_hooks(
            model,
            layers_to_register_and_share=layers_to_register_and_share,
            mode=loqer_scaling_mode,
            torch_dtype=loqer_dtype,
            mode_map=loqer_scaling_mode_map,
        )

        calibration_datamodule = get_data_module(
            name=calibration_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=20 * num_calibration_samples,
            num_workers=num_workers,
        )

        calibration_dataloader = DataLoader(
            calibration_datamodule["train"],
            batch_size=perplexity_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        mem_info = get_all_device_mem_info()
        logger.info(f"Device memory before profiling starts: \n{pformat(mem_info)}")
        profile_outputs = evaluate_perplexity(
            model=model,
            eval_dataloader=calibration_dataloader,
            num_samples=num_calibration_samples if loqer_scaling_mode != "identity" else perplexity_eval_batch_size,
            progress_bar=True,
            input_device=None,
            description="Calibrating",
        )

        profiler_factory.remove_all_hooks()
        if loqer_scaling_mode in ["rxx", "mixed"]:
            scale_dict = profiler_factory.get_scale_dict(
                progress_bar=True,
                sqrtm_implementation=loqer_sqrtm_implementation,
                sqrtm_num_iters=loqer_sqrtm_num_iters,
            )
        else:
            scale_dict = profiler_factory.get_scale_dict(progress_bar=True)

        share_scales(scale_dict, layers_to_register_and_share)
        logger.info(f"Perplexity after profiling: {profile_outputs['perplexity']:.4f}")

        logger.info("🚀 Quantizing model...")
        quantize_model(model, loqer_config)

        logger.info("🚀 Loqer is enabled. Computing A & B...")
        layers_to_approximate = find_layers_to_approximate(model)
        layers_to_approximate = list(filter(lambda x: x in scale_dict, layers_to_approximate))
        AB_dict, mse_df = compute_AB_and_approximation_error(
            model, layers_to_approximate, scale_dict, loqer_config, move_model_back=False
        )
        del scale_dict
        mse_df_emoji = mse_df.copy()
        mse_df_emoji.loc[:, "mse?"] = mse_df["mse"].apply(_mse_threshold_emoji)
        logger.info(f"Approximation error (mean squared error): \n{mse_df_emoji.to_markdown()}")

        missing_chunks = _verify_AB_dict_chunks(
            AB_dict_dir=AB_dict_dir, num_chunks=num_chunks, current_chunk_tag=chunk_tag
        )

        # save this chunk
        mse_df_dir = output_dir.joinpath("approximation_error")
        config_dir = output_dir.joinpath("config")
        AB_dict_path = AB_dict_dir.joinpath(f"{chunk_tag}.pt")
        logger.info(f"Current missing chunks: {missing_chunks}")
        AB_dict_dir.mkdir(parents=True, exist_ok=True)
        mse_df_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)

        mse_df.to_csv(mse_df_dir.joinpath(f"{chunk_tag}.csv"), index=False)
        torch.save(AB_dict, AB_dict_path)
        with open(config_dir.joinpath(f"{chunk_tag}.yaml"), "w") as f:
            yaml.dump(config, f)
    else:
        logger.info(f"🔊 All chunks of AB_dict are ready. Quantize model, attach AB_dict and run evaluation.")
        # Load model and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=loqer_dtype, _attn_implementation="eager"
        )
        model.eval()
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        device_map = create_device_map(model, device_map=device_map)
        logger.info(f"Device map: {device_map}")
        model = dispatch_model(model, device_map)
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        quantize_model(model, loqer_config)

    if len(missing_chunks) == 0:
        # merge all chunks
        AB_dict = {}
        AB_dict_chunks = list(filter(lambda x: x.is_file() and x.name.endswith(".pt"), AB_dict_dir.iterdir()))
        for chunk in tqdm(AB_dict_chunks, desc="Loading chunks"):
            AB_dict.update(torch.load(chunk))
                
        # We don't need to use the loqer_quantized model here. The training script will load the model from the model_name_or_path
        if fine_tuning:
            # fine-tuning the model
            if loqer_config['default-1']['w_quantizer']['name'] != 'normalfloat':
                raise ValueError("Fine-tuning only supports normalfloat quantizer.")
            logger.info("🚀 Fine-tuning...")
            fine_tuning_args = Namespace(**args)
            model=loftQ_fine_tuning(fine_tuning_args, AB_dict=AB_dict)
        else:
            # attach A & B
            layers_to_approximate = find_layers_to_approximate(model)
            attach_AB(model, layers_to_approximate, AB_dict)

        # evaluate
        if not disable_perplexity_eval:
            logger.info("🚀 Evaluating perplexity...")
            eval_datamodule = get_data_module(
                name=perplexity_evaluation_set,
                tokenizer=tokenizer,
                padding="max_length",
                max_length=perplexity_max_seq_length,
                num_raw_samples=None,
                num_workers=num_workers,
            )
            eval_dataloader = DataLoader(
                eval_datamodule["test"],
                batch_size=perplexity_eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=data_collator,
            )
            model = model.to(eval_dtype)
            model = dispatch_model(model, device_map)
            mem_info = get_all_device_mem_info()
            logger.info(f"Device memory before perplexity evaluation starts: \n{pformat(mem_info)}")
            ppl_results = evaluate_perplexity(
                model=model,
                eval_dataloader=eval_dataloader,
                num_samples=None,
                progress_bar=True,
                input_device=None,
                description="Evaluating",
            )

            if disable_loqer:
                logger.info(f"Perplexity after quantization (no LoQER): {ppl_results['perplexity']:.4f}")
            else:
                logger.info(f"Perplexity after approximation: {ppl_results['perplexity']:.4f}")

        if not disable_lm_eval:
            logger.info("🚀 Evaluating lm-eval downstream tasks...")
            model = model.to(eval_dtype)
            model = dispatch_model(model, device_map)
            lm_eval_results = evaluate_harness_downstream(
                model,
                tasks=lm_eval_tasks,
                num_fewshot=lm_eval_num_fewshot,
                no_cache=True,
                batch_size=lm_eval_batch_size,
            )
            logger.info(f"Downstream task results: \n{pformat(lm_eval_results)}")

        # save perplexity results
        if not disable_perplexity_eval:
            with open(output_dir / "perplexity_results.yaml", "w") as f:
                yaml.dump(ppl_results, f)

        # save lm-eval results
        if not disable_lm_eval:
            with open(output_dir / "lm_eval_results.yaml", "w") as f:
                yaml.dump(lm_eval_results, f)
    else:
        logger.info(f"Chunk {chunk_tag} is saved. Please run the pipeline for the rest chunks.")
        logger.info(f"Missing chunks: \n{pformat(missing_chunks)}")


def chunk_checker():
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("layers_per_chunk", type=int, help="Layers per chunk")
    parser.add_argument("--output-dir", "-o", dest="output_dir", type=str, help="Output directory", default=None)
    args = parser.parse_args()

    model_name = args.model_name
    layers_per_chunk = args.layers_per_chunk
    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    num_chunks = _check_chunk_id(model_name, layers_per_chunk, None)

    if output_dir is not None:
        AB_dict_dir = output_dir.joinpath("AB_dict")
        if not AB_dict_dir.is_dir():
            logger.warning(f"Output directory {output_dir} does not exist.")
            return
        else:
            missing_chunks = _verify_AB_dict_chunks(
                AB_dict_dir=AB_dict_dir, num_chunks=num_chunks, current_chunk_tag=None
            )
            if len(missing_chunks) == 0:
                logger.info("All chunks are ready.")
            else:
                logger.info(f"Missing chunks: \n{pformat(missing_chunks, sort_dicts=False)}")


def _merge_chunked_approximation_error(approx_error_dir: Path):
    if isinstance(approx_error_dir, str):
        approx_error_dir = Path(approx_error_dir)
    df = None
    for file in approx_error_dir.iterdir():
        if not file.is_file():
            continue

        if not re.match(r"\d+-of-\d+.csv", file.name):
            continue

        chunk_df = pd.read_csv(file)
        if df is None:
            df = chunk_df
        else:
            df = pd.concat([df, chunk_df], ignore_index=True)

    return df


def merge_chunked_results():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--quick-save",
        "-s",
        dest="quick_save",
        action="store_true",
        help="Save merged results to $output_dir/approximation_error/quick-save-$timestamp.csv",
        default=False,
    )
    parser.add_argument("--output-file", "-o", dest="output_file", type=str, help="Output file", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    approx_error_dir = output_dir.joinpath("approximation_error")
    assert approx_error_dir.is_dir(), f"Directory {approx_error_dir} does not exist."

    df = _merge_chunked_approximation_error(approx_error_dir)

    logger.info(f"Merged approximation error: \n{df.to_markdown()}")

    if args.quick_save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        df.to_csv(approx_error_dir.joinpath(f"quick-save-{timestamp}.csv"), index=False)
        logger.info(f"Quick save to {approx_error_dir.joinpath(f'quick-save-{timestamp}.csv')}")

    if args.output_file is not None:
        df.to_csv(args.output_file, index=False)
        logger.info(f"Saved to {args.output_file}")
