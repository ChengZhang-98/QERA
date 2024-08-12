# %%
import sys
from pathlib import Path

src_path = Path(__file__).parents[2].joinpath("src")
assert src_path.exists(), f"Path does not exist: {src_path}"
sys.path.append(src_path.as_posix())

import logging
import math
import multiprocessing
from pprint import pprint
import re

import torch
import numpy as np
from scipy import linalg as spla
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from torch.utils.data import DataLoader
from loqer.datasets import get_data_module
from loqer.evaluate import evaluate_perplexity
from loqer.utils import create_device_map, get_layer_by_name

logger = logging.getLogger("loqer." + __name__)


def sqrtm_scipy(A: np.ndarray, blocksize=64):
    if not isinstance(A, np.ndarray):
        raise RuntimeError("input matrix must be a numpy array")
    A_sqrt, errest = spla.sqrtm(A, disp=False, blocksize=blocksize)
    # return A_sqrt, errest
    return dict(A_sqrt=A_sqrt, errest=errest)


def is_pos_def(x):
    if isinstance(x, np.ndarray):
        return np.all(np.linalg.eigvals(x) > 0).item()
    elif isinstance(x, torch.Tensor):
        return torch.all(torch.linalg.eigvals(x) > 0).cpu().item()
    else:
        raise RuntimeError("input must be a numpy array or torch tensor")


def create_my_device_map(model, num_hidden_layers):
    num_devices = torch.cuda.device_count()
    max_memory = {i: torch.cuda.mem_get_info(i)[0] // 5 for i in range(num_devices)}
    device_map = infer_auto_device_map(model, no_split_module_classes=model._no_split_modules, max_memory=max_memory)
    n_decoder_layers = num_hidden_layers
    n_layers_per_device = math.floor(1 + n_decoder_layers / num_devices)
    if n_layers_per_device == 0:
        n_layers_per_device = 1
    balanced_device_map = {}
    current_device = 0
    current_decoder_idx = 0

    for layer_name in device_map:
        if ".layers." in layer_name:
            if (current_decoder_idx + 1) % n_layers_per_device == 0:
                current_device += 1
            current_decoder_idx += 1
        balanced_device_map[layer_name] = min(current_device, num_devices - 1)
    device_map = balanced_device_map
    return device_map


class ScaleHookFactoryRxx:
    """
    For row vector x,
    scale = E[ x^T x ] ^ 0.5, where Rxx = E[ x^T x ] is the auto-correlation matrix
    """

    def __init__(self, rxx_dtype) -> None:
        self.scales = {}
        self.errests = {}  # estimated error of sqrtm
        self.is_pos_def = {}
        self.n_samples = {}
        self.compute_devices = {}
        self.sizes = {}
        self.rxx_dtype = rxx_dtype
        self.scale_dtype = None
        self.handles = []

    @torch.no_grad()
    def get_scale_hook(self, name: str) -> callable:
        """ """

        self.scales[name] = None
        self.n_samples[name] = 0

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.reshape(-1, x.shape[-1])
            n_samples, in_features = x.shape
            if self.scales[name] is None:
                if self.scale_dtype is None:
                    self.scale_dtype = x.dtype
                self.compute_devices[name] = x.device
                if self.compute_devices[name].type == "cpu":
                    logger.warning("Using CPU for computing Rxx, this may be slow")
                self.scales[name] = torch.zeros(in_features, in_features, dtype=self.rxx_dtype)
                self.sizes[name] = (in_features, in_features)

            compute_device = self.compute_devices[name]
            scales = self.scales[name].to(compute_device, self.rxx_dtype)
            # batched outer product
            delta = torch.einsum("bi,bj->ij", x, x).to(self.rxx_dtype)
            scales += delta
            self.scales[name] = scales
            self.n_samples[name] += n_samples

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self, progress_bar=False) -> dict[str, torch.Tensor]:

        # convert to numpy
        for name in self.scales:
            self.scales[name] = self.scales[name].cpu().numpy()
        num_cores = multiprocessing.cpu_count()
        num_processes = max(1, num_cores // 64)

        with multiprocessing.Pool(num_processes) as pool:
            with tqdm(total=len(self.scales), desc="Computing scale", disable=not progress_bar) as pbar:
                for name, scale_and_err in zip(self.scales.keys(), pool.imap(sqrtm_scipy, self.scales.values())):
                    scale = scale_and_err["A_sqrt"]
                    errest = scale_and_err["errest"]
                    self.scales[name] = scale
                    self.errests[name] = errest.item()
                    pbar.update()

        # convert to torch tensor
        for name in self.scales:
            scale = self.scales[name]
            n_samples = self.n_samples[name]
            scale = torch.from_numpy(scale).to(self.scale_dtype).to(self.compute_devices[name])
            scale = scale * (1 / math.sqrt(n_samples))
            self.scales[name] = scale

        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def sqrtm_estimated_error_vs_hidden_size(model_name, layers_to_profile: list[str], batch_size: int):
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model_dtype = torch.float32
    num_calibration_samples = 256
    rxx_dtype = torch.float64

    hook_factory = ScaleHookFactoryRxx(rxx_dtype=rxx_dtype)
    # only supports llama model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=model_dtype, _attn_implementation="eager"
    )
    assert isinstance(model, LlamaForCausalLM)
    model.eval()
    max_layer_idx = max([int(re.search(r"\d+", layer_name).group()) for layer_name in layers_to_profile])
    # remove the layers after the last layer to profile
    model.model.layers = model.model.layers[: max_layer_idx + 1]
    print(f"Removed layers after {max_layer_idx}. Pruned model: \n", model.model.layers)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map = create_my_device_map(model, num_hidden_layers=max_layer_idx + 1)
    print("Device map: ", device_map)
    model = dispatch_model(model, device_map)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    for layer_name in layers_to_profile:
        try:
            target_layer = get_layer_by_name(model, layer_name)
        except ValueError:
            continue
        target_layer.register_forward_hook(hook_factory.get_scale_hook(layer_name))

    calibration_datamodule = get_data_module(
        name="slim_pajama_6b",
        tokenizer=tokenizer,
        padding="max_length",
        max_length=2048,
        num_raw_samples=20 * num_calibration_samples,
        num_workers=8,
    )

    calibration_dataloader = DataLoader(
        calibration_datamodule["train"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=data_collator,
    )

    _ = evaluate_perplexity(
        model=model,
        eval_dataloader=calibration_dataloader,
        num_samples=num_calibration_samples,
        progress_bar=True,
        description="Calibrating",
    )
    del model

    _ = hook_factory.get_scale_dict(progress_bar=True)
    hook_factory.remove_all_hooks()
    sqrtm_result_profile = dict(
        model_name=model_name,
        profiled_layers=layers_to_profile,
        errests=list(hook_factory.errests.values()),
        is_pos_def=list(hook_factory.is_pos_def.values()),
        rxx_sizes=list(hook_factory.sizes.values()),
    )
    return sqrtm_result_profile


# %%

if __name__ == "__main__":
    from argparse import ArgumentParser
    import yaml
    from pathlib import Path
    import datetime

    parser = ArgumentParser()
    parser.add_argument(
        "--model_names",
        "-m",
        dest="model_names",
        nargs="+",
        default=[
            "Cheng98/TinyLlama_v1.1",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
        ],
    )
    parser.add_argument(
        "--layers_to_profile",
        "-l",
        dest="layers_to_profile",
        nargs="+",
        default=[
            # "model.layers.3.self_attn.q_proj",
            # "model.layers.6.self_attn.o_proj",
            # "model.layers.9.mlp.gate_proj",
            "model.layers.11.mlp.down_proj",
        ],
    ),
    parser.add_argument("--batch_size", "-b", dest="batch_size", type=int, default=8)

    args = parser.parse_args()

    model_names = args.model_names
    layers_to_profile = args.layers_to_profile
    batch_size = args.batch_size
    timestamp = datetime.datetime.now().strftime("%Y%-m-%d_%H-%M-%S")
    yaml_path = Path(__file__).parent.joinpath(f"sqrtm_error_vs_rxx_size_{timestamp}.yaml")
    results = []

    for model_name in model_names:
        result = sqrtm_estimated_error_vs_hidden_size(
            model_name,
            layers_to_profile=layers_to_profile,
            batch_size=batch_size,
        )
        results.append(result)
        with open(yaml_path, "w") as f:
            yaml.safe_dump(results, f)
        pprint(result, sort_dicts=False)


# %% [markdown]
# | Model | Rxx accumulation | Sqrtm success? |  Runtime (s) | Memory (GB) |  Notes |
# |---|---|---|---|---|---|---|
