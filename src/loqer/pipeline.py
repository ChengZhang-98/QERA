import logging
import yaml
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

import torch
from torch.utils.data import DataLoader
import transformers
from accelerate import dispatch_model

from .statistic_profiler import register_scale_hooks
from .datasets import get_data_module
from .evaluate import evaluate_perplexity, evaluate_harness_downstream
from .models import find_layers_to_approximate, quantize_model
from .approximate import compute_AB_and_approximation_error, attach_AB
from .utils import create_device_map

logger = logging.getLogger(__name__)


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
        help="Loqer scaling mode",
        default=None,
        choices=["diagonal", "rxx"],
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
    loqer_config = config["loqer_config"]
    disable_perplexity_eval = config["disable_perplexity_eval"]
    disable_lm_eval = config["disable_lm_eval"]

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=loqer_dtype, _attn_implementation="eager"
    )
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map = create_device_map(model, device_map=device_map)
    model = dispatch_model(model, device_map)
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if not disable_loqer and AB_dict is None:
        logger.info("🚀 Running data calibration...")
        profiler_factory = register_scale_hooks(model, mode=loqer_scaling_mode)

        calibration_datamodule = get_data_module(
            name=calibration_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=perplexity_max_seq_length,
            num_raw_samples=num_calibration_samples * 10,
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
            num_samples=None,
            progress_bar=True,
            input_device=None,
            description="Calibrating",
        )

        profiler_factory.remove_all_hooks()
        scale_dict = profiler_factory.get_scale_dict()
        logger.info(f"Perplexity after profiling: {profile_outputs['perplexity']:.4f}")

    logger.info("🚀 Quantizing model...")
    quantize_model(model, loqer_config)

    if not disable_loqer:
        if AB_dict is None:
            logger.info("🚀 Loqer is enabled. Computing A & B...")
            layers_to_approximate = find_layers_to_approximate(model)
            AB_dict, mse_df = compute_AB_and_approximation_error(model, layers_to_approximate, scale_dict, loqer_config)
            attach_AB(model, layers_to_approximate, AB_dict)
            logger.info(f"Approximation error (mean squared error): \n{mse_df.to_markdown()}")
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
        if output_dir.is_dir() and len(list(output_dir.iterdir())) > 0:
            raise ValueError(f"Output directory {output_dir} is not empty")

        output_dir.mkdir(parents=True, exist_ok=True)

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
