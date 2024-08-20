import tqdm
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig
from peft.utils.loftq_utils import is_bnb_4bit_available, _SafetensorLoader, NFQuantizer
from peft.tuners.lora import Linear4bit
from peft.tuners.lora.layer import Linear as LoraLinear


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
def init_lora_loftq_4bit(qweight, weight: torch.Tensor, num_iters: int, reduced_rank: int, compute_device):
    compute_device = "cuda" if compute_device is None else compute_device
    assert compute_device != "cpu"

    quant_state = qweight.quant_state
    dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)

    residual = weight.clone()
    error_f_norm = []

    for i in range(num_iters):
        qweight = bnb.nn.Params4bit(
            residual.cpu(),
            requires_grad=False,
            compress_statistics=quant_state.nested,
            quant_type=quant_state.quant_type,
            quant_state=None,
            bnb_quantized=False,
            blocksize=quant_state.blocksize,
        ).to(compute_device)
        dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)
        # else:
        #     quantizer = NFQuantizer(
        #         num_bits=num_bits,
        #         device=compute_device,
        #         method="normal" if quant_state.quant_type == "nf4" else "uniform",
        #         block_size=quant_state.blocksize,
        #     )
        #     residual = residual.to(device=compute_device, dtype=torch.float32)
        #     qweight, max_abs, shape = quantizer.quantize_block(residual)
        #     dequantized_weight = quantizer.dequantize_block(qweight, max_abs, shape)

        weight = weight.to(device=compute_device, dtype=torch.float32)
        dequantized_weight = dequantized_weight.to(device=compute_device, dtype=torch.float32)
        residual = weight - dequantized_weight

        torch.cuda.empty_cache()
        output = _low_rank_decomposition_loftq(residual, reduced_rank)
        L, R = output["L"], output["R"]
        residual = weight - torch.mm(L, R)

        error_f_norm.append(output["error_f_norm"])

    lora_A, lora_B = R, L

    return lora_A, lora_B, qweight, error_f_norm


def replace_lora_weights_loftq_4bit(
    peft_model, model_path=None, adapter_name="default", num_iters: int = 1, compute_device=None
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

        lora_A, lora_B, new_qweight, error_f_norm = init_lora_loftq_4bit(
            qweight=module.weight,
            weight=ori_weight,
            num_iters=num_iters,
            reduced_rank=reduced_rank,
            compute_device=compute_device,
        )
        error_dict[name] = error_f_norm

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
        module.weight.data = new_qweight

    return error_dict


@torch.no_grad()
def init_lora_loftq_2bit(weight: torch.Tensor, bnb_quant_type: str, num_iters, reduced_rank: int, compute_device):
    compute_device = "cuda" if compute_device is None else compute_device
    assert compute_device != "cpu"
    ori_weight_dtype = weight.dtype

    residual = weight.clone().to(device=compute_device, dtype=torch.float32)
    weight = weight.to(device=compute_device, dtype=torch.float32)
    error_f_norm = []

    quantizer = NFQuantizer(num_bits=2, device=compute_device, method=bnb_quant_type, block_size=64)

    for i in range(num_iters):
        qweight, max_abs, shape = quantizer.quantize_block(residual)
        dequantized_weight = quantizer.dequantize_block(qweight, max_abs, shape)

        dequantized_weight = dequantized_weight.to(device=compute_device, dtype=torch.float32)
        residual = weight - dequantized_weight

        torch.cuda.empty_cache()
        output = _low_rank_decomposition_loftq(residual, reduced_rank)
        L, R = output["L"], output["R"]
        residual = weight - torch.mm(L, R)
        error_f_norm.append(output["error_f_norm"])

    lora_A, lora_B = R, L
    return lora_A, lora_B, dequantized_weight.to(ori_weight_dtype), error_f_norm


def replace_lora_weights_loftq_2bit(
    peft_model, bnb_quant_type, adapter_name="default", num_iters: int = 1, compute_device=None
):
    """
    Emulated 2-bit loftq
    """
    assert bnb_quant_type in ["normal", "uniform"]
    prefix = "base_model.model."

    named_modules = {name: module for name, module in peft_model.named_modules()}
    error_dict = {}

    for name, module in tqdm.tqdm(named_modules.items(), desc="Replacing LoRA adapters (Emulated 2-bit LoftQ)"):
        if not isinstance(module, LoraLinear):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        reduced_rank = module.r[adapter_name]

        lora_A, lora_B, new_weight, error_f_norm = init_lora_loftq_2bit(
            weight=module.weight,
            bnb_quant_type=bnb_quant_type,
            num_iters=num_iters,
            reduced_rank=reduced_rank,
            compute_device=compute_device,
        )
        error_dict[name] = error_f_norm

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
        module.weight.data = new_weight

    return error_dict


@torch.no_grad()
def _low_rank_decomposition_loqer(x: torch.Tensor, reduced_rank: int):
    assert x.ndim == 2
    # U, S, Vh = torch.linalg.svd(x, full_matrices=True)
    # L = U[:, :reduced_rank] @ torch.diag(S[:reduced_rank])
    # R = Vh[:reduced_rank, :]

    # it seems full_matrices=False is more accurate
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    L = U @ torch.diag(S)[:, :reduced_rank]
    R = Vh[:reduced_rank, :]

    error_f_norm = S[reduced_rank:].pow(2).sum().sqrt().cpu().item()

    return L, R, error_f_norm


@torch.no_grad()
def init_lora_loqer_4bit(qweight, weight: torch.Tensor, scale: torch.Tensor, reduced_rank, compute_device):
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
            = x @ (q_weight + Uk @ Sk @ Vhk @ scale^-1).T
            ~ x @ (q_weight + residual).T
            = x @ weight.T
    """
    assert scale.ndim in [1, 2]
    assert scale.shape[0] == weight.shape[1]
    compute_device = qweight.data.device if compute_device is None else compute_device

    if scale.ndim == 1:
        scale = torch.diag(scale)

    quant_state = qweight.quant_state
    dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)

    # if num_bits == 2:
    #     # quantize the value to 2-bit, though the format is still 4-bit
    #     quantizer = NFQuantizer(
    #         num_bits=num_bits,
    #         device=compute_device,
    #         method="normal" if quant_state.quant_type == "nf4" else "uniform",
    #         block_size=quant_state.blocksize,
    #     )
    #     qweight, max_abs, shape = quantizer.quantize_block(dequantized_weight)
    #     dequantized_weight = quantizer.dequantize_block(qweight, max_abs, shape)

    dequantized_weight = dequantized_weight.to(device=compute_device, dtype=torch.float32)
    weight = weight.to(device=compute_device, dtype=torch.float32)
    scale = scale.to(device=compute_device, dtype=torch.float32)

    residual = weight - dequantized_weight
    residual_scaled = residual @ scale

    torch.cuda.empty_cache()
    L, R_, error_f_norm = _low_rank_decomposition_loqer(residual_scaled, reduced_rank)

    R = torch.linalg.solve(scale, R_.T).T

    lora_A, lora_B = R, L
    return lora_A, lora_B, qweight, error_f_norm


@torch.no_grad()
def replace_lora_weights_loqer_4bit(
    peft_model,
    scale_dict: dict[str, torch.Tensor],
    model_path=None,
    adapter_name="default",
    compute_device=None,
):
    prefix = "base_model.model."
    safetensor_loader = _SafetensorLoader(peft_model, model_path)

    named_modules = {name: module for name, module in peft_model.named_modules()}

    error_dict = {}

    for name, module in tqdm.tqdm(named_modules.items(), desc="Replacing LoRA adapter (LoQER+)"):
        if not isinstance(module, Linear4bit):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        name = name[len(prefix) :]
        ori_weight = safetensor_loader.get_tensor(name + ".weight")
        scale = scale_dict[name]
        reduced_rank = module.r[adapter_name]

        lora_A, lora_B, qweight, error_f_norm = init_lora_loqer_4bit(
            qweight=module.weight,
            weight=ori_weight,
            scale=scale,
            reduced_rank=reduced_rank,
            compute_device=module.weight.device if compute_device is None else compute_device,
        )
        error_dict[name] = [error_f_norm]

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
        module.weight.data = qweight

    return error_dict


@torch.no_grad()
def init_lora_loqer_2bit(
    weight: torch.Tensor, scale: torch.Tensor, bnb_quant_type: str, reduced_rank: int, compute_deivce
):
    assert scale.ndim in [1, 2]
    assert scale.shape[0] == weight.shape[1]
    assert bnb_quant_type in ["normal", "uniform"]
    compute_deivce = "cuda" if compute_deivce is None else compute_deivce

    orig_weight_dtype = weight.dtype
    orig_weight = weight.clone().to(device=compute_deivce, dtype=torch.float32)

    if scale.ndim == 1:
        scale = torch.diag(scale)

    quantizer = NFQuantizer(num_bits=2, device=compute_deivce, method=bnb_quant_type, block_size=64)

    qweight, max_abs, shape = quantizer.quantize_block(weight)
    dequantized_weight = quantizer.dequantize_block(qweight, max_abs, shape)
    dequantized_weight = dequantized_weight.to(device=compute_deivce, dtype=torch.float32)
    scale = scale.to(device=compute_deivce, dtype=torch.float32)

    residual = orig_weight - dequantized_weight
    residual_scaled = residual @ scale

    torch.cuda.empty_cache()
    L, R_, error_f_norm = _low_rank_decomposition_loqer(residual_scaled, reduced_rank)
    R = torch.linalg.solve(scale, R_.T).T

    lora_A, lora_B = R, L
    return lora_A, lora_B, dequantized_weight.to(orig_weight_dtype), error_f_norm


def replace_lora_weight_loqer_2bit(peft_model, bnb_quant_type, scale_dict, adapter_name="default", compute_device=None):
    """
    Emulated 2-bit loqer
    """

    assert bnb_quant_type in ["normal", "uniform"]
    prefix = "base_model.model."

    named_modules = {name: module for name, module in peft_model.named_modules()}
    error_dict = {}

    for name, module in tqdm.tqdm(named_modules.items(), desc="Replacing LoRA adapters (Emulated 2-bit LoQER+)"):
        if not isinstance(module, LoraLinear):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        reduced_rank = module.r[adapter_name]
        name = name[len(prefix) :]
        scale = scale_dict[name]

        lora_A, lora_B, new_weight, error_f_norm = init_lora_loqer_2bit(
            weight=module.weight,
            scale=scale,
            bnb_quant_type=bnb_quant_type,
            reduced_rank=reduced_rank,
            compute_deivce=module.weight.device if compute_device is None else compute_device,
        )
        error_dict[name] = [error_f_norm]

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
        module.weight.data = new_weight

    return error_dict


@torch.no_grad()
def init_lora_qlora_2bit(weight: torch.Tensor, bnb_quant_type: str, compute_device):
    compute_device = "cuda" if compute_device is None else compute_device
    assert compute_device != "cpu"
    ori_weight_dtype = weight.dtype
    ori_weight = weight.clone().to(device=compute_device, dtype=torch.float32)
    weight = weight.to(device=compute_device, dtype=torch.float32)

    quantizer = NFQuantizer(num_bits=2, device=compute_device, method=bnb_quant_type, block_size=64)

    qweight, max_abs, shape = quantizer.quantize_block(weight)
    dequantized_weight = quantizer.dequantize_block(qweight, max_abs, shape)

    error_f_norm = [torch.linalg.norm(ori_weight - dequantized_weight, ord="fro").cpu().item()]

    dequantized_weight = dequantized_weight.to(ori_weight_dtype)
    return dequantized_weight, error_f_norm


@torch.no_grad()
def replace_lora_weight_qlora_2bit(peft_model, bnb_quant_type, compute_device=None):
    assert bnb_quant_type in ["normal", "uniform"]
    prefix = "base_model.model."

    named_modules = {name: module for name, module in peft_model.named_modules()}
    error_dict = {}

    for name, module in tqdm.tqdm(named_modules.items(), desc="Replacing LoRA adapters (Emulatd 2-bit qLoRA)"):
        if not isinstance(module, LoraLinear):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        name = name[len(prefix) :]

        new_qweight, error_f_norm = init_lora_qlora_2bit(
            weight=module.weight,
            bnb_quant_type=bnb_quant_type,
            compute_device=module.weight.device if compute_device is None else compute_device,
        )
        error_dict[name] = error_f_norm

        module.weight.data = new_qweight

    return error_dict
