import logging

import torch
from torch.utils.data import DataLoader
import transformers
from accelerate import dispatch_model

from .statistic_profiler import register_scale_hooks
from .datasets import get_data_module
from .evaluate import evaluate_perplexity
from .models import find_layers_to_approximate, quantize_model
from .approximate import compute_AB_and_approximation_error, attach_AB
from .utils import create_device_map

logger = logging.getLogger(__name__)


def pipeline():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name = "facebook/opt-1.3b"
    model_name = "Open-Orca/Mistral-7B-OpenOrca"
    dtype = torch.float32
    eval_dtype = torch.float16
    device_map = "auto-balance"
    scaling_mode = "diagonal"
    calibration_set = "slim_pajama_6b"
    num_calibration_samples = 16
    batch_size_ppl = 4
    max_seq_length = 2048

    loqer_config = {
        # llama
        r"model\.layers\.[0-9]+\.self_attn\.(k|q|v|o)_proj": "default-1",
        r"model\.layers\.[0-9]+\.mlp\.(gate|down|up)_proj": "default-1",
        r"model\.layers\.[0-9]+\.self_attn\.(matmul_0|matmul_1)": "default-matmul",
        # opt
        r"model\.decoder\.layers\.[0-9]+\.self_attn\.(k|q|v|out)_proj": "default-1",
        r"model\.decoder\.layers\.[0-9]+\.(fc1|fc2)": "default-1",
        r"model\.decoder\.layers\.[0-9]+\.self_attn\.(bmm_0|bmm_1)": "default-matmul",
        "default-1": {
            "rank": 64,
            "name": "loqer",
            "x_quantizer": {"name": "bypass"},
            "w_quantizer": {
                "name": "block_fp",
                "width": 4,
                "exponent_width": 8,
                "exponent_bias": None,
                "block_size": [4, 16],
            },
            "b_quantizer": {"name": "bypass"},
        },
        "default-matmul": {
            "name": "flexible",
            "x_quantizer": {"name": "bypass"},
            "w_quantizer": {"name": "bypass"},
        },
    }

    eval_set = "wikitext2"

    num_workers = 8
    disable_loqer = False

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, _attn_implementation="eager"
    )
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map = create_device_map(model, device_map=device_map)
    model = dispatch_model(model, device_map)
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if not disable_loqer:
        profiler_factory = register_scale_hooks(model, mode=scaling_mode)

        calibration_datamodule = get_data_module(
            name=calibration_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=max_seq_length,
            num_raw_samples=num_calibration_samples * 10,
        )

        calibration_dataloader = DataLoader(
            calibration_datamodule["train"],
            batch_size=batch_size_ppl,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        results = evaluate_perplexity(
            model=model,
            eval_dataloader=calibration_dataloader,
            num_samples=None,
            progress_bar=True,
            input_device=None,
            description="Calibrating",
        )

        profiler_factory.remove_all_hooks()
        scale_dict = profiler_factory.get_scale_dict()

        logger.info(f"Perplexity after profiling: {results['perplexity']:.4f}")

    quantize_model(model, loqer_config)

    if not disable_loqer:
        logger.info("🔊 Loqer is enabled")
        layers_to_approximate = find_layers_to_approximate(model)
        AB_dict, mse_df = compute_AB_and_approximation_error(model, layers_to_approximate, scale_dict, loqer_config)
        attach_AB(model, layers_to_approximate, AB_dict)
        logger.info(f"Approximation error (mean squared error): \n{mse_df.to_markdown()}")
    else:
        logger.warning("⚠️ Loqer is disabled, skipping layer approximation")
    logger.info(f"Model after approximation: \n{model}")

    eval_datamodule = get_data_module(
        name=eval_set,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_seq_length,
        num_raw_samples=None,
    )
    eval_dataloader = DataLoader(
        eval_datamodule["test"],
        batch_size=batch_size_ppl,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
    )
    model = model.to(eval_dtype)
    model = dispatch_model(model, device_map)
    results = evaluate_perplexity(
        model=model,
        eval_dataloader=eval_dataloader,
        num_samples=None,
        progress_bar=True,
        input_device=None,
        description="Evaluating",
    )

    if disable_loqer:
        logger.info(f"Perplexity after quantization (no LoQER): {results['perplexity']:.4f}")
    else:
        logger.info(f"Perplexity after approximation: {results['perplexity']:.4f}")
