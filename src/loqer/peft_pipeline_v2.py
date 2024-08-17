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

from loqer.datasets import get_data_module
from loqer.models import find_layers_to_register_scale_hook
from loqer.statistic_profiler import register_scale_hooks, share_scales
from loqer.evaluate import evaluate_perplexity
from loqer.fine_tuning import replace_lora_weights_loftq, replace_lora_weights_loqer
from loqer.utils import create_device_map

logger = logging.getLogger(__name__)


def adapt_and_save_clm_model():
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
    parser = ArgumentParser()
    parser.add_argument("model_name_or_path", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("adapter_init", type=str, choices=["loftq", "loqer"])
    parser.add_argument("--loqer-calibration-set", type=str, default="wikitext2", help="Default: wikitext2")
    parser.add_argument("--loqer-num-calibration-samples", type=int, default=128)
    parser.add_argument("--loqer-calibration-batch-size", type=int, default=2)
    parser.add_argument("--loqer-max-seq-length", type=int, default=2048)
    parser.add_argument("--loftq-num-iters", type=int, default=1, help="Default: 1")
    parser.add_argument("--bnb-quant-type", type=str, default="nf4", help="Default: nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-n-bits", type=int, default=4, help="Default: 4", choices=[2, 4])
    parser.add_argument("--lora-rank", type=int, default=64, help="Default: 64")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="Default: 16.0")
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
    args = parser.parse_args()

    transformers.set_seed(42)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite_output_dir:
            raise FileExistsError(f"Output directory {output_dir} already exists")
        else:
            logger.warning(f"⚠️ Output directory {output_dir} already exists and will be overwritten")
            shutil.rmtree(output_dir, ignore_errors=True)

    # LoQER calibration
    scale_dict = None
    if args.adapter_init == "loqer":
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, _attn_implementation="eager")
        model.eval()
        if "cuda" in args.device_map:
            model.to(args.device_map)
        else:
            if hasattr(model, "tie_weights"):
                model.tie_weights()
            device_map = create_device_map(model, args.device_map)
            model = dispatch_model(model, device_map)

        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        layers_to_register_and_share = find_layers_to_register_scale_hook(model)
        profiler_factory = register_scale_hooks(
            model, layers_to_register_and_share, mode="diag", torch_dtype=torch.float32
        )
        calibration_datamodule = get_data_module(
            args.loqer_calibration_set,
            tokenizer=tokenizer,
            padding="max_length",
            max_length=args.loqer_max_seq_length,
            num_raw_samples=args.loqer_num_calibration_samples * 30,
            num_workers=args.num_workers,
        )
        calibration_dataloader = DataLoader(
            calibration_datamodule["train"],
            batch_size=args.loqer_calibration_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=data_collator,
        )
        start = time.time()
        profile_outputs = evaluate_perplexity(
            model,
            eval_dataloader=calibration_dataloader,
            num_samples=args.loqer_num_calibration_samples,
            progress_bar=True,
            description="Calibrating scales for LoQER+",
        )
        calibration_time = time.time() - start
        logger.info(f"Profiling outputs:\n{pformat(profile_outputs, sort_dicts=False)}")
        profiler_factory.remove_all_hooks()
        scale_dict = profiler_factory.get_scale_dict(progress_bar=True)
        share_scales(scale_dict, layers_to_register_and_share)

    # it seems LoftQ's NFQuantizer does not support double quantization
    bnb_4bit_use_double_quant = args.bnb_n_bits == 4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=args.bnb_quant_type,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, quantization_config=bnb_config)
    model.eval()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=args.lora_target_modules,
        init_lora_weights=True,
    )
    peft_model = get_peft_model(model, lora_config)

    error_dict = None
    elapsed = None
    if args.adapter_init == "loftq":
        start = time.time()
        error_dict = replace_lora_weights_loftq(peft_model, num_iters=args.loftq_num_iters, num_bits=args.bnb_n_bits)
        elapsed = time.time() - start
    elif args.adapter_init == "loqer":
        start = time.time()
        error_dict = replace_lora_weights_loqer(peft_model, scale_dict=scale_dict, num_bits=args.bnb_n_bits)
        elapsed = time.time() - start + calibration_time
    else:
        raise ValueError(f"Invalid adapter init: {args.adapter_init}")

    output_dir.mkdir(parents=True, exist_ok=True)

    peft_model.save_pretrained(output_dir)
    bnb_config_dict = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
        "bnb_4bit_quant_type": args.bnb_quant_type,
    }
    with open(output_dir / "bnb_config.yaml", "w") as f:
        yaml.safe_dump(bnb_config_dict, f)
    logger.info(f"Model saved to {output_dir}")

    args = vars(args)
    args["results"] = {"initialization_time": elapsed, "error_dict": error_dict}
    with open(output_dir / "adapt_and_save_args.yaml", "w") as f:
        yaml.safe_dump(args, f)
