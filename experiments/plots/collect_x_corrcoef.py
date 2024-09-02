import re
import datetime

import torch
from torch.utils.data import DataLoader
import transformers
from accelerate import dispatch_model
from tqdm import tqdm

from loqer.datasets import get_data_module
from loqer.models import find_layers_to_approximate, find_layers_to_register_scale_hook
from loqer.statistic_profiler import register_scale_hooks
from loqer.evaluate import evaluate_perplexity

from loqer.utils import create_device_map


class ScaleHookFactoryCorrCoef:
    """
    scale = diag( sqrt( E[ x_1^2]), sqrt( E[ x_2^2]), ..., sqrt( E[ x_n^2] ) )
    """

    def __init__(self) -> None:
        self.input_samples = {}
        self.handles = []

    def get_scale_hook(self, name: str) -> callable:
        self.input_samples[name] = None

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.view(-1, x.shape[-1]).cpu()
            if self.input_samples[name] is None:
                self.input_samples[name] = x
            else:
                self.input_samples[name] = torch.cat([self.input_samples[name], x], dim=0)

        return scale_hook

    @torch.no_grad()
    def get_corrcoef(self):
        corrcoef = {}
        for name in tqdm(self.input_samples, desc="Computing corrcoef"):
            x = self.input_samples[name].cuda().T
            corrcoef[name] = torch.corrcoef(x).cpu().half()

        return corrcoef

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


@torch.no_grad()
def collect_rxx(model_name, device_map, target_layers: list[str]):
    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, _attn_implementation="eager", torch_dtype=torch.float16
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

    registered_layers = []
    hook_factory = ScaleHookFactoryCorrCoef()
    for name, layer in model.named_modules():
        if any(re.match(pattern, name) for pattern in target_layers):
            hook_factory.handles.append(layer.register_forward_hook(hook_factory.get_scale_hook(name)))
            registered_layers.append(name)

    print(f"Registered layers: {registered_layers}")

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

    _ = evaluate_perplexity(model, dataloader, num_samples=32, progress_bar=True)
    del model

    hook_factory.remove_all_hooks()

    corr_coefs = hook_factory.get_corrcoef()
    return corr_coefs


if __name__ == "__main__":
    from safetensors.torch import save_file

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_names = ["meta-llama/Meta-Llama-3-8B"]  # ["Cheng98/TinyLlama_v1.1"]
    target_layers = [
        r"model\.layers\.(0|3|7|11|15|19|23|27|31)\.self_attn\.k_proj",
        r"model\.layers\.(0|3|7|11|15|19|23|27|31)\.self_attn\.o_proj",
        r"model\.layers\.(0|3|7|11|15|19|23|27|31)\.mlp\.(gate_proj|down_proj)",
    ]

    for model_name in model_names:
        corrcoef = collect_rxx(model_name, device_map="auto-balanced", target_layers=target_layers)
        model_name_escape = model_name.replace("/", "_")
        save_path = f"corrcoef_{model_name_escape}_{timestamp}.safetensors"
        save_file(corrcoef, save_path)
