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
from functools import partial

import torch
import numpy as np
from scipy import linalg as spla
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import transformers
from accelerate import dispatch_model
from torch.utils.data import DataLoader
from loqer.datasets import get_data_module
from loqer.evaluate import evaluate_perplexity
from loqer.utils import create_device_map, get_layer_by_name

logger = logging.getLogger("loqer." + __name__)


def is_pos_def(x):
    if isinstance(x, np.ndarray):
        return np.all(np.linalg.eigvals(x) > 0).item()
    elif isinstance(x, torch.Tensor):
        return torch.all(torch.linalg.eigvals(x) > 0).cpu().item()
    else:
        raise RuntimeError("input must be a numpy array or torch tensor")


def sqrtm_scipy(A: np.ndarray, blocksize):
    if not isinstance(A, np.ndarray):
        raise RuntimeError("input matrix must be a numpy array")
    A_sqrt, errest = spla.sqrtm(A, disp=False, blocksize=blocksize)
    # return A_sqrt, errest
    return dict(A_sqrt=A_sqrt, errest=errest)


class ScaleHookFactoryRxx:
    """
    For row vector x,
    scale = E[ x^T x ] ^ 0.5, where Rxx = E[ x^T x ] is the auto-correlation matrix
    """

    def __init__(self, rxx_dtype, blocksize) -> None:
        self.scales = {}
        self.errests = {}  # estimated error of sqrtm
        self.n_samples = {}
        self.compute_devices = {}
        self.sizes = {}
        self.rxx_dtype = rxx_dtype
        self.blocksize = blocksize
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
                for name, scale_and_err in zip(
                    self.scales.keys(),
                    pool.starmap(sqrtm_scipy, zip(self.scales.values(), [self.blocksize] * len(self.scales))),
                ):
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


def sqrtm_estimated_error_vs_hidden_size(model_name, layers_to_profile: list[str], block_size: int):
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model_dtype = torch.float32
    num_calibration_samples = 64
    rxx_dtype = torch.float64

    hook_factory = ScaleHookFactoryRxx(rxx_dtype=rxx_dtype, blocksize=block_size)
    # only supports llama model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=model_dtype, _attn_implementation="eager"
    )
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map = create_device_map(model, device_map="auto-balanced")
    model = dispatch_model(model, device_map)

    assert isinstance(model, LlamaForCausalLM)
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
        batch_size=1,
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
        sqrtm_block_size=block_size,
        errests=list(hook_factory.errests.values()),
        sizes=list(hook_factory.sizes.values()),
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
        "--model_name",
        "-m",
        dest="model_name",
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--layers_to_profile",
        "-l",
        dest="layers_to_profile",
        nargs="+",
        default=[
            "model.layers.3.self_attn.q_proj",
            "model.layers.9.mlp.gate_proj",
            "model.layers.18.mlp.down_proj",
        ],
    )
    parser.add_argument(
        "--blocksizes",
        "-b",
        dest="blocksizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512],
    )

    args = parser.parse_args()

    model_name = args.model_name
    layers_to_profile = args.layers_to_profile
    block_sizes = args.blocksizes
    timestamp = datetime.datetime.now().strftime("%Y%-m-%d_%H-%M-%S")
    yaml_path = Path(__file__).parent.joinpath(f"sqrtm_error_vs_rxx_size_{timestamp}.yaml")
    results = []

    for block_size in block_sizes:
        result = sqrtm_estimated_error_vs_hidden_size(
            model_name, layers_to_profile=layers_to_profile, block_size=block_size
        )
        results.append(result)
        with open(yaml_path, "w") as f:
            yaml.dump(results, f)
        pprint(result, sort_dicts=False)
