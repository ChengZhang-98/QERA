import re
import datetime

import torch
from torch.utils.data import DataLoader
import transformers
from accelerate import dispatch_model

from qera.datasets import get_data_module
from qera.models import find_layers_to_register_scale_hook
from qera.statistic_profiler import register_scale_hooks
from qera.evaluate import evaluate_perplexity

from qera.utils import create_device_map


@torch.no_grad()
def collect_rxx(model_name, device_map, target_layers: list[str]):
    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, _attn_implementation="eager", torch_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model.eval()

    if hasattr(model, "tie_weights"):
        model.tie_weights()

    if device_map:
        device_map = create_device_map(model, device_map)
        model = dispatch_model(model, device_map)
    else:
        model.cuda()

    layers_to_register_and_share = find_layers_to_register_scale_hook(model)
    layers_to_register_and_share_filtered = []

    for layer in layers_to_register_and_share:
        if any(re.match(pattern, layer["target_layer"]) for pattern in target_layers):
            layers_to_register_and_share_filtered.append(layer)

    print(f"Layers to register and share: {layers_to_register_and_share_filtered}")

    profiler_factory = register_scale_hooks(
        model,
        layers_to_register_and_share_filtered,
        mode="rxx",
        torch_dtype=torch.bfloat16,
    )

    datamodule = get_data_module(
        name="wikitext2",
        tokenizer=tokenizer,
        max_length=1024,
        padding="max_length",
        num_raw_samples=20 * 128,
        num_workers=8,
    )
    dataloader = DataLoader(
        datamodule["train"],
        batch_size=4,
        shuffle=False,
        num_workers=8,
        collate_fn=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    _ = evaluate_perplexity(model, dataloader, num_samples=64, progress_bar=True)

    profiler_factory.remove_all_hooks()

    scales = profiler_factory.scales
    for name in scales:
        scale = scales[name].cuda() / profiler_factory.n_samples[name]
        scales[name] = scale.cpu()

    return scales


if __name__ == "__main__":
    from safetensors.torch import save_file

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # model_names = ["meta-llama/Meta-Llama-3-8B"]
    model_names = ["meta-llama/Llama-2-7b-hf"]
    target_layers = [
        r"model\.layers\.(0|3|7|11|15|19|23|27|31)\.self_attn\.k_proj",
        r"model\.layers\.(0|3|7|11|15|19|23|27|31)\.self_attn\.o_proj",
        r"model\.layers\.(0|3|7|11|15|19|23|27|31)\.mlp\.(gate_proj|down_proj)",
    ]

    for model_name in model_names:
        scales = collect_rxx(
            model_name, device_map="auto-balanced", target_layers=target_layers
        )
        model_name_escape = model_name.replace("/", "_")
        scales_path = f"scales_{model_name_escape}_{timestamp}.safetensors"
        save_file(scales, scales_path)
