import tqdm
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig
from peft.utils.loftq_utils import is_bnb_4bit_available, _SafetensorLoader
from peft.tuners.lora import Linear4bit


def _low_rank_decomposition_loftq(weight, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {matrix_dimension} dimensions.")

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    error_f_norm = S[reduced_rank:].pow(2).sum().sqrt().cpu().item()

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, "reduced_rank": reduced_rank, "error_f_norm": error_f_norm}


@torch.no_grad()
def init_lora_loftq(qweight, weight: torch.Tensor, num_iters: int, num_bits: int, reduced_rank: int, compute_device):
    assert num_bits == 4, "Only 4-bit is supported for now."
    compute_device = "cuda" if compute_device is None else compute_device

    quant_state = qweight.quant_state
    dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)

    residual = weight.clone()

    error_f_norm = []

    for i in range(num_iters):
        if compute_device is None:
            compute_device_i = qweight.data.device
        else:
            compute_device_i = compute_device
        torch.cuda.empty_cache()
        if num_bits == 4:
            qweight = bnb.nn.Params4bit(
                residual.cpu(),
                requires_grad=False,
                compress_statistics=quant_state.nested,
                quant_type=quant_state.quant_type,
                blocksize=quant_state.blocksize,
            ).to(compute_device_i)
            dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)
        else:
            raise NotImplementedError

        weight = weight.to(device=compute_device_i, dtype=torch.float32)
        dequantized_weight = dequantized_weight.to(device=compute_device_i, dtype=torch.float32)
        residual = weight - dequantized_weight

        output = _low_rank_decomposition_loftq(residual, reduced_rank)
        L, R = output["L"], output["R"]
        residual = weight - torch.mm(L, R)

        error_f_norm.append(output["error_f_norm"])

    lora_A, lora_B = R, L

    return lora_A, lora_B, qweight, error_f_norm


def replace_lora_weights_loftq(
    peft_model, model_path=None, adapter_name="default", num_iters: int = 4, num_bits: int = 4, compute_device=None
):
    """
    LoftQ (A replicate of Official LoftQ)

    q_weight, weight # (out_dim, in_dim)

    residual = weight - q_weight
    U, S, V_T = svd(residual)
    L = U[:, :rank] @ sqrt(S[:rank]) # L.shape = (out_dim, rank)
    R = sqrt(S[:rank]) @ V_T[:rank, :] # R.shape = (rank, in_dim)

    lora_A = R # lora_A.shape = (rank, in_dim)
    lora_B = L # lora_B.shape = (out_dim, rank)

    forward: x @ (q_weight.T + lora_A.T @ lora_B.T) = x @ (q_weight + lora_B @ lora_A).T = x @ (q_weight + L @ R).T
            = x @ (q_weight + U[:, :rank] @ sqrt(S[:rank]) @ sqrt(S[:rank]) @ V_T[:rank, :]).T
            = x @ (q_weight + U[:, :rank] @ S[:rank] @ V_T[:rank, :]).T = x @ (q_weight + residual).T
            ~ x @ weight.T
    """
    prefix = "base_model.model."
    safetensor_loader = _SafetensorLoader(peft_model, model_path)

    named_modules = {name: module for name, module in peft_model.named_modules()}
    error_dict = {}

    for name, module in tqdm.tqdm(named_modules.items(), desc="Replacing LoRA adapters (LoftQ)"):
        if not isinstance(module, Linear4bit):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        name = name[len(prefix) :]
        ori_weight = safetensor_loader.get_tensor(name + ".weight")

        reduced_rank = module.r[adapter_name]

        lora_A, lora_B, new_qweight, error_f_norm = init_lora_loftq(
            qweight=module.weight,
            weight=ori_weight,
            num_iters=num_iters,
            num_bits=num_bits,
            reduced_rank=reduced_rank,
            compute_device=compute_device,
        )
        error_dict[name] = error_f_norm

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
        module.weight.data = new_qweight

    return error_dict


@torch.no_grad()
def _low_rank_decomposition_loqer(x: torch.Tensor, reduced_rank: int):
    assert x.ndim == 2
    # U, S, Vh = torch.linalg.svd(x, full_matrices=True)
    # L = U[:, :reduced_rank] @ torch.diag(S[:reduced_rank])
    # R = Vh[:reduced_rank, :]

    # it seems full_matrices=False is more accurate
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    sqrt_S = torch.sqrt(S)
    L = U @ torch.diag(sqrt_S)[:, :reduced_rank]
    R = torch.diag(sqrt_S)[:reduced_rank, :] @ Vh

    return L, R


@torch.no_grad()
def init_lora_loqer(qweight, weight: torch.Tensor, scale: torch.Tensor, num_bits: int, reduced_rank, compute_device):
    """
    LoQER+ for PEFT

    q_weight, weight # (out_dim, in_dim)
    scale # (in_dim, in_dim)

    residual = weight - q_weight
    residual_scaled = residual @ scale

    U, S, Vh = svd(residual_scaled)
    L = U @ sqrt(S)[:, :rank]  # L.shape = (out_dim, rank)
    R = sqrt(S)[:rank, :] @ Vh @ scale^-1 # R.shape = (rank, in_dim)

    lora_A = R # lora_A.shape = (rank, in_dim)
    lora_B = L # lora_B.shape = (out_dim, rank)

    forward: x @ (q_weight.T + lora_A.T @ lora_B.T) = x @ (q_weight + lora_B @ lora_A).T = x @ (q_weight + L @ R).T
            = x @ (q_weight + U @ sqrt(S)[:, :rank] @ sqrt(S)[:rank, :] @ Vh @ scale^-1).T
            ~ x @ (q_weight + residual).T
            = x @ weight.T
    """
    assert num_bits == 4
    assert scale.ndim in [1, 2]
    assert scale.shape[0] == weight.shape[1]
    # compute_device = "cuda" if compute_device is None else compute_device
    compute_device = qweight.data.device if compute_device is None else compute_device

    if scale.ndim == 1:
        scale = torch.diag(scale)

    quant_state = qweight.quant_state
    dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)

    weight = weight.to(device=compute_device, dtype=torch.float32)
    dequantized_weight = dequantized_weight.to(device=compute_device, dtype=torch.float32)
    scale = scale.to(device=compute_device, dtype=torch.float32)

    residual = weight - dequantized_weight
    residual_scaled = residual @ scale

    torch.cuda.empty_cache()
    L, R_ = _low_rank_decomposition_loqer(residual_scaled, reduced_rank)

    R = torch.linalg.solve(scale, R_.T).T

    lora_A, lora_B = R, L
    return lora_A, lora_B


@torch.no_grad()
def replace_lora_weights_loqer(
    peft_model,
    scale_dict: dict[str, torch.Tensor],
    model_path=None,
    adapter_name="default",
    num_bits: int = 4,
    compute_device=None,
):
    prefix = "base_model.model."
    safetensor_loader = _SafetensorLoader(peft_model, model_path)

    named_modules = {name: module for name, module in peft_model.named_modules()}

    for name, module in tqdm.tqdm(named_modules.items(), desc="Replacing LoRA adapter (LoQER+)"):
        if not isinstance(module, Linear4bit):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        name = name[len(prefix) :]
        ori_weight = safetensor_loader.get_tensor(name + ".weight")
        scale = scale_dict[name]
        reduced_rank = module.r[adapter_name]

        lora_A, lora_B = init_lora_loqer(
            qweight=module.weight,
            weight=ori_weight,
            scale=scale,
            num_bits=num_bits,
            reduced_rank=reduced_rank,
            compute_device=compute_device,
        )

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
