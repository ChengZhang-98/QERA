from copy import deepcopy
import logging
from functools import partial

import torch
from tqdm import tqdm
import pandas as pd

from .utils import (
    find_matched_pattern,
    get_layer_by_name,
    get_full_device_map,
    move_module_to_device,
)
from .quantize import get_quantizer

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_AB_and_approximation_error(
    model,
    layers_to_approximate: list[str],
    scale_dict: dict[str, torch.Tensor],
    qera_config: dict,
    move_model_back: bool = True,
):
    AB_dict = {}
    df = pd.DataFrame(columns=["layer_name", "mse", "rank"])

    full_device_map = get_full_device_map(model)
    model = model.to("cpu")

    for layer_name in tqdm(layers_to_approximate, desc="Computing low-rank A and B"):
        torch.cuda.empty_cache()
        # device
        layer = get_layer_by_name(model, layer_name)
        layer.to(full_device_map[layer_name])
        # scale
        scale = scale_dict[layer_name].clone()
        # qera config
        matched_entry = find_matched_pattern(layer_name, qera_config.keys())
        if isinstance(matched_entry, str):
            matched_entry = qera_config[matched_entry]
        layer_qera_config = deepcopy(qera_config[matched_entry])
        layer_AB_dict, mse = _compute_scales_and_error_for_fc(
            layer_name, layer, scale, layer_qera_config
        )
        layer_AB_dict = {k: v.to("cpu") for k, v in layer_AB_dict.items()}
        AB_dict.update(layer_AB_dict)
        df.loc[len(df)] = [layer_name, mse, layer_qera_config["rank"]]

        layer.to("cpu")

    if move_model_back:
        move_module_to_device(model, full_device_map)

    return AB_dict, df


def _compute_scale_inv_dot_U(scale: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    If not the scale matrix is not invertible, add turbulence to the scale matrix

    Perform S^-1 @ U using `torch.linalg.solve`, which is more numerically stable.
    Refer to https://pytorch.org/docs/stable/generated/torch.linalg.inv.html
    """
    if scale.ndim == 1:
        scale = torch.where(
            scale <= 0, torch.ones_like(scale) * torch.finfo(scale.dtype).eps, scale
        )
        return torch.linalg.solve(torch.diag(scale), U)
    elif scale.ndim == 2:
        try:
            return torch.linalg.solve(scale, U)
        except RuntimeError as e:
            logger.warning(
                f"Matrix inversion failed: {e} Adding turbulence to the scale matrix"
            )
            U_scale, S_scale, V_T_scale = torch.linalg.svd(scale)
            S_scale = torch.where(
                S_scale <= 0,
                torch.ones_like(S_scale) * torch.finfo(S_scale.dtype).eps,
                S_scale,
            )
            scale = U_scale @ torch.diag(S_scale) @ V_T_scale
            return torch.linalg.solve(scale, U)
    else:
        raise ValueError("Scale must be either a vector (diagonal) or a matrix")


def _compute_scales_and_error_for_fc(
    layer_name: str,
    layer: torch.nn.Linear,
    scale: torch.Tensor,
    layer_qera_config: dict,
) -> tuple[dict[str, torch.Tensor], float]:
    """

    q_error_T = W^T - W_q^T
    SVD(S @ q_error_T) = U @ S @ V^T

    A = S^-1 @ U_k
    B = S_k @ V_k^T

    y_hat = x @ (W_q^T + AB)
          = x @ (W_q^T + S^-1 @ U_k @ S_k @ V_k^T)
          = x @ (W_q^T + S^-1 @ (S @ q_error_T)_k)
          ~ x @ (W_q^T + W^T - W_q^T)
          = x @ W^T

    """
    rank = layer_qera_config["rank"]

    w_quantizer = partial(
        get_quantizer(layer_qera_config["w_quantizer"].pop("name")),
        **layer_qera_config["w_quantizer"],
    )

    ab_q_config = deepcopy(layer_qera_config["x_quantizer"])

    ab_quantizer = partial(get_quantizer(ab_q_config.pop("name")), **ab_q_config)

    weight = layer.weight

    weight_q = w_quantizer(weight).to(weight.device)
    scale = scale.to(weight.dtype).to(weight.device)
    if scale.ndim == 1:
        assert (
            scale.shape[0] == weight.shape[1]
        ), "Scale must have the same number of elements as the weight"
        scaled_q_error_T = torch.diag(scale) @ (weight - weight_q).transpose(0, 1)
    elif scale.ndim == 2:
        assert scale.shape[0] == scale.shape[1], "Scale must be a square matrix"
        scaled_q_error_T = scale @ (weight - weight_q).transpose(0, 1)
    else:
        raise ValueError("Scale must be either a vector (diagonal) or a matrix")

    U, S, V_T = torch.linalg.svd(scaled_q_error_T)

    U = U[:, :rank]
    S = S[:rank]
    V_T = V_T[:rank, :]

    if scale.ndim == 1:
        A = _compute_scale_inv_dot_U(scale, U)
        B = ab_quantizer(torch.diag(S) @ V_T)
    elif scale.ndim == 2:
        A = _compute_scale_inv_dot_U(scale, U)
        B = ab_quantizer(torch.diag(S) @ V_T)
    else:
        raise ValueError("Scale must be either a vector (diagonal) or a matrix")

    A_name = layer_name + ".A"
    B_name = layer_name + ".B"

    mean_squared_error = (
        torch.nn.functional.mse_loss(
            weight.transpose(0, 1), weight_q.transpose(0, 1) + A @ B
        )
        .cpu()
        .item()
    )
    if mean_squared_error > 1e-3:
        logger.warning(f"Mean squared error for {layer_name}: {mean_squared_error}")
    return {A_name: A, B_name: B}, mean_squared_error


def attach_AB(model, layers_to_approximate, AB_dict: dict[str, torch.Tensor]):
    for layer_name in layers_to_approximate:
        A = AB_dict[layer_name + ".A"]
        B = AB_dict[layer_name + ".B"]

        layer: torch.nn.Linear = get_layer_by_name(model, layer_name)
        device = layer.weight.device
        dtype = layer.weight.dtype
        A = A.to(dtype).to(device)
        B = B.to(dtype).to(device)
        layer.A = torch.nn.Parameter(A)
        layer.B = torch.nn.Parameter(B)

    return model
