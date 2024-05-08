import logging
import yaml
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

import torch
from torch.utils.data import DataLoader
import transformers
from accelerate import dispatch_model
from transformers import BitsAndBytesConfig, AwqConfig, GPTQConfig
from auto_gptq import exllama_set_max_input_length

from .statistic_profiler import register_scale_hooks, share_scales
from .datasets import get_data_module
from .evaluate import evaluate_perplexity, evaluate_harness_downstream
from .models import find_layers_to_approximate, quantize_model, find_layers_to_register_scale_hook
from .approximate import compute_AB_and_approximation_error, attach_AB
from .utils import create_device_map

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
        help="Loqer scaling mode, one of ['diagonal', 'diag', 'rxx', 'dummy'].",
        default=None,
        choices=["diagonal", "diag", "rxx", "dummy"],  # "diag" is alias of "diagonal"
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

    # check output directory
    if output_dir is not None and output_dir.is_dir() and len(list(output_dir.iterdir())) > 0:
        raise ValueError(f"Output directory {output_dir} is not empty")

    # sqrtm_implementation
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
        if loqer_scaling_mode == "dummy":
            logger.info("🔊 Using dummy scale (torch.ones)")
        logger.info("🚀 Running data calibration...")
        layers_to_register_and_share = find_layers_to_register_scale_hook(model)
        profiler_factory = register_scale_hooks(
            model,
            layers_to_register_and_share=layers_to_register_and_share,
            mode=loqer_scaling_mode,
            torch_dtype=loqer_dtype,
        )

        calibration_datamodule = get_data_module(
            name=calibration_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=20 * num_calibration_samples,
        )

        calibration_dataloader = DataLoader(
            calibration_datamodule["train"],
            batch_size=perplexity_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        profile_outputs = evaluate_perplexity(
            model=model,
            eval_dataloader=calibration_dataloader,
            num_samples=num_calibration_samples if loqer_scaling_mode != "dummy" else perplexity_eval_batch_size,
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
        if AB_dict is None:
            logger.info("🚀 Loqer is enabled. Computing A & B...")
            layers_to_approximate = find_layers_to_approximate(model)
            AB_dict, mse_df = compute_AB_and_approximation_error(model, layers_to_approximate, scale_dict, loqer_config)
            attach_AB(model, layers_to_approximate, AB_dict)
            mse_df_emoji = mse_df.copy()
            mse_df_emoji.loc[:, "mse?"] = mse_df["mse"].apply(_mse_threshold_emoji)
            logger.info(f"Approximation error (mean squared error): \n{mse_df_emoji.to_markdown()}")
        else:
            logger.info("🚀 Loqer is enabled and AB_dict is specified. Attaching A & B...")
            AB_dict = torch.load(AB_dict)
            attach_AB(model, list(AB_dict.keys()), AB_dict)
    else:
        logger.warning("⚠️ Loqer is disabled, skipping layer approximation")
    logger.info(f"Model after approximation: \n{model}")

    if not disable_perplexity_eval:
        logger.info("🚀 Evaluating perplexity...")
        eval_datamodule = get_data_module(
            name=perplexity_evaluation_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=None,
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

        if not disable_loqer:
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
    parser.add_argument("--device-map", dest="device_map", type=str, help="Device map", default="auto-balance")
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
