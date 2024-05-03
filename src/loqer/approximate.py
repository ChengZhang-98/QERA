from copy import deepcopy
from functools import partial

import torch
from tqdm import tqdm
import pandas as pd

from .utils import find_matched_pattern, get_layer_by_name, get_layer_name, set_layer_by_name, setattr_recursive
from .quantize import get_quantizer


@torch.no_grad()
def compute_AB_and_approximation_error(
    model, layers_to_approximate: list[str], scale_dict: dict[str, torch.Tensor], loqer_config: dict
):
    AB_dict = {}
    df = pd.DataFrame(columns=["layer_name", "mse", "rank"])

    for layer_name in tqdm(layers_to_approximate, desc="Computing low-rank A and B"):
        # scale
        scale = scale_dict[layer_name]
        # loqer config
        matched_entry = find_matched_pattern(layer_name, loqer_config.keys())
        if isinstance(matched_entry, str):
            matched_entry = loqer_config[matched_entry]
        layer_loqer_config = deepcopy(loqer_config[matched_entry])
        layer_AB_dict, mse = _compute_scales_and_error_for_fc(model, layer_name, scale, layer_loqer_config)
        AB_dict.update(layer_AB_dict)
        df.loc[len(df)] = [layer_name, mse.item(), layer_loqer_config["rank"]]

    return AB_dict, df


def _compute_scales_and_error_for_fc(
    model: torch.nn.Module, layer_name: str, scale: torch.Tensor, layer_loqer_config: dict
) -> dict[str, torch.Tensor]:
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
    rank = layer_loqer_config["rank"]

    w_quantizer = partial(
        get_quantizer(layer_loqer_config["w_quantizer"].pop("name")), **layer_loqer_config["w_quantizer"]
    )

    ab_q_config = deepcopy(layer_loqer_config["x_quantizer"])

    ab_quantizer = partial(get_quantizer(ab_q_config.pop("name")), **ab_q_config)

    layer: torch.nn.Linear = get_layer_by_name(model, layer_name)
    weight = layer.weight

    weight_q = w_quantizer(weight)
    scale = scale.to(weight.device)
    if scale.ndim == 1:
        assert scale.shape[0] == weight.shape[1], "Scale must have the same number of elements as the weight"
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
        A = ab_quantizer(torch.diag(scale).inverse() @ U)
        B = ab_quantizer(torch.diag(S) @ V_T)
    elif scale.ndim == 2:
        A = ab_quantizer(scale.inverse() @ U)
        B = ab_quantizer(torch.diag(S) @ V_T)
    else:
        raise ValueError("Scale must be either a vector (diagonal) or a matrix")

    A_name = layer_name + ".A"
    B_name = layer_name + ".B"

    mean_squared_error = torch.nn.functional.mse_loss(weight.transpose(0, 1), weight_q.transpose(0, 1) + A @ B)

    return {A_name: A, B_name: B}, mean_squared_error


def attach_AB(model, layers_to_approximate, AB_dict: dict[str, torch.Tensor]):
    for layer_name in layers_to_approximate:
        A = AB_dict[layer_name + ".A"]
        B = AB_dict[layer_name + ".B"]

        layer: torch.nn.Linear = get_layer_by_name(model, layer_name)
        device = layer.weight.device
        layer.A = torch.nn.Parameter(A.to(device))
        layer.B = torch.nn.Parameter(B.to(device))

    return model
