import logging
import re
import time
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

from loqer.fine_tuning import loftQ_parse_args, loftQ_fine_tuning
from loqer.statistic_profiler import register_scale_hooks, share_scales
from loqer.datasets import get_data_module
from loqer.evaluate import evaluate_perplexity, evaluate_harness_downstream
from loqer.models import find_layers_to_approximate, quantize_model, find_layers_to_register_scale_hook
from loqer.approximate import compute_AB_and_approximation_error, attach_AB
from loqer.utils import create_device_map, get_all_device_mem_info

logger = logging.getLogger(__name__)


def collect_runtime(model_name, loqer_scaling_mode):
    loqer_config = {
        "model\\.layers\\.[0-9]+\\.self_attn\\.(k|q|v|o)_proj": "default-1",
        "model\\.layers\\.[0-9]+\\.mlp\\.(gate|down|up)_proj": "default-1",
        "model\\.layers\\.[0-9]+\\.self_attn\\.(matmul_0|matmul_1)": "default-matmul",
        "model\\.decoder\\.layers\\.[0-9]+\\.self_attn\\.(k|q|v|out)_proj": "default-1",
        "model\\.decoder\\.layers\\.[0-9]+\\.(fc1|fc2)": "default-1",
        "model\\.decoder\\.layers\\.[0-9]+\\.self_attn\\.(bmm_0|bmm_1)": "default-matmul",
        "default-1": {
            "rank": 32,
            "name": "loqer",
            "is_ptq": True,
            "x_quantizer": {"name": "bypass"},
            "w_quantizer": {"name": "mxint", "width": 4, "block_size": 32, "block_axis": -1},
            "b_quantizer": {"name": "bypass"},
        },
        "default-matmul": {"name": "flexible", "x_quantizer": {"name": "bypass"}, "w_quantizer": {"name": "bypass"}},
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, _attn_implementation="eager"
    )
    model.eval()

    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map = create_device_map(model, device_map="auto-balanced")
    logger.info(f"Device map: {device_map}")
    model = dispatch_model(model, device_map)
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    calibration_datamodule = get_data_module(
        name="slim_pajama_6b",
        tokenizer=tokenizer,
        padding="max_length",
        max_length=2048,
        num_raw_samples=20 * 128,
        num_workers=256,
    )
    calibration_dataloader = DataLoader(
        calibration_datamodule["train"],
        batch_size=4,
        shuffle=False,
        num_workers=8,
        collate_fn=data_collator,
    )
    inference_start = time.time()
    profile_outputs = evaluate_perplexity(
        model=model,
        eval_dataloader=calibration_dataloader,
        num_samples=256 if loqer_scaling_mode != "identity" else 4,
        progress_bar=True,
        input_device=None,
        description="Calibrating",
    )
    inference_elapsed = time.time() - inference_start

    layers_to_register_and_share = find_layers_to_register_scale_hook(model)
    patterns = [
        "layers.0.self_attn.k_proj",
        "layers.6.self_attn.o_proj",
        "layers.14.mlp.gate_proj",
        "layers.21.mlp.down_proj",
    ]
    layers_to_register_and_share_filtered = []
    for layer in layers_to_register_and_share:
        for pattern in patterns:
            if pattern in layer["target_layer"]:
                layers_to_register_and_share_filtered.append(layer)
                break
    print(f"layers_to_register_and_share_filtered: {layers_to_register_and_share_filtered}")
    profiler_factory = register_scale_hooks(
        model,
        layers_to_register_and_share=layers_to_register_and_share_filtered,
        mode=loqer_scaling_mode,
        torch_dtype=torch.float32,
        mode_map=None,
    )

    inference_and_calibrate_one_layer_start = time.time()
    profile_outputs = evaluate_perplexity(
        model=model,
        eval_dataloader=calibration_dataloader,
        num_samples=256 if loqer_scaling_mode != "identity" else 4,
        progress_bar=True,
        input_device=None,
        description="Calibrating",
    )
    inference_and_calibrate_one_layer_elapsed = time.time() - inference_and_calibrate_one_layer_start

    pure_inference_elapsed = inference_elapsed
    rxx_clibration_elapsed = inference_and_calibrate_one_layer_elapsed - pure_inference_elapsed
    rxx_clibration_elapsed = rxx_clibration_elapsed * model.config.num_hidden_layers + pure_inference_elapsed

    mem_info = get_all_device_mem_info()
    logger.info(f"Device memory before profiling starts: \n{pformat(mem_info)}")

    profiler_factory.remove_all_hooks()
    sqrtm_start = time.time()
    if loqer_scaling_mode == "rxx":
        scale_dict = profiler_factory.get_scale_dict(
            progress_bar=True,
            sqrtm_implementation="blocked",
            sqrtm_num_iters=None,
        )
    else:
        scale_dict = profiler_factory.get_scale_dict(progress_bar=True)
    sqrtm_elapsed = time.time() - sqrtm_start
    sqrtm_elapsed = sqrtm_elapsed * model.config.num_hidden_layers

    share_scales(scale_dict, layers_to_register_and_share_filtered)
    logger.info(f"Perplexity after profiling: {profile_outputs['perplexity']:.4f}")

    layers_to_approximate = find_layers_to_approximate(model)
    layers_to_approximate_filtered = []
    for layer in layers_to_approximate:
        for pattern in patterns:
            if pattern in layer:
                layers_to_approximate_filtered.append(layer)
                break
    print(f"layers_to_approximate_filtered: {layers_to_approximate_filtered}")
    scale_calculation_start = time.time()
    AB_dict, mse_df = compute_AB_and_approximation_error(
        model, layers_to_approximate_filtered, scale_dict, loqer_config, move_model_back=False
    )
    scale_calculation_elapsed = time.time() - scale_calculation_start
    scale_calculation_elapsed = scale_calculation_elapsed * model.config.num_hidden_layers

    return {
        "pure_inference (part of calibration)": pure_inference_elapsed,
        "calibration": rxx_clibration_elapsed,
        "sqrtm": sqrtm_elapsed,
        "svd": scale_calculation_elapsed,
    }


if __name__ == "__main__":
    model_names = [
        # "Cheng98/TinyLlama_v1.1",
        # "google/gemma-2-2b",
        # "meta-llama/Llama-2-7b-hf",
        # "meta-llama/Llama-2-13b-hf",
        "huggyllama/llama-30b",
        # "meta-llama/Llama-2-70b-hf",
    ]

    loqer_scaling_modes = ["diag", "rxx"]
    results = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"./runtime_{timestamp}.yaml"

    for loqer_scaling_mode in loqer_scaling_modes:
        for model_name in model_names:
            results[f"{model_name}_{loqer_scaling_mode}"] = collect_runtime(model_name, loqer_scaling_mode)
            with open(save_path, "w") as f:
                yaml.dump(results, f)
