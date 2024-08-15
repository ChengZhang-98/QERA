import tqdm
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig
from peft.utils.loftq_utils import is_bnb_4bit_available, _SafetensorLoader
from peft.utils.loftq_utils import _low_rank_decomposition as _low_rank_decomposition_loftq
from peft.tuners.lora import Linear4bit

from loqer.datasets import get_data_module
from loqer.models import find_layers_to_approximate, find_layers_to_register_scale_hook
from loqer.statistic_profiler import register_scale_hooks, share_scales
from loqer.evaluate import evaluate_perplexity


@torch.no_grad()
def init_lora_loftq(qweight, weight: torch.Tensor, num_iters, num_bits, reduced_rank, compute_device):
    assert num_bits == 4
    compute_device = "cuda" if compute_device is None else compute_device

    quant_state = qweight.quant_state
    dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)

    weight = weight.to(device=compute_device, dtype=torch.float32)
    residual = weight.clone()

    for i in range(num_iters):
        torch.cuda.empty_cache()
        if num_bits == 4:
            qweight = bnb.nn.Params4bit(
                residual.cpu(),
                requires_grad=False,
                compress_statistics=quant_state.nested,
                quant_type=quant_state.quant_type,
                blocksize=quant_state.blocksize,
            ).to(compute_device)
            dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)
        else:
            raise NotImplementedError

        residual = weight - dequantized_weight

        output = _low_rank_decomposition_loftq(residual, reduced_rank)
        L, R = output["L"], output["R"]
        residual = weight - torch.mm(L, R)

    lora_A, lora_B = R, L

    return lora_A, lora_B, qweight


def replace_lora_weights_loftq(
    peft_model, model_path=None, adapter_name="default", num_iters: int = 4, num_bits: int = 4, compute_device=None
):
    """
    *: LoftQ (Official)
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

    for name, module in tqdm.tqdm(named_modules.items(), desc="Replacing LoRA adapters (LoftQ)"):
        if not isinstance(module, Linear4bit):
            continue
        if not name.startswith(prefix):
            raise TypeError(f"Not peft model")

        name = name[len(prefix) :]
        ori_weight = safetensor_loader.get_tensor(name + ".weight")

        reduced_rank = module.r[adapter_name]

        lora_A, lora_B, new_qweight = init_lora_loftq(
            qweight=module.weight,
            weight=ori_weight,
            num_iters=num_iters,
            num_bits=num_bits,
            reduced_rank=reduced_rank,
            compute_device=compute_device,
        )

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
        module.weight.data = new_qweight


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
    assert num_bits == 4
    assert scale.ndim in [1, 2]
    assert scale.shape[0] == weight.shape[1]
    compute_device = "cuda" if compute_device is None else compute_device

    if scale.ndim == 1:
        scale = torch.diag(scale)

    quant_state = qweight.quant_state
    dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, quant_state)

    weight = weight.to(device=compute_device, dtype=torch.float32)
    dequantized_weight = dequantized_weight.to(device=compute_device, dtype=torch.float32)
    scale = scale.to(device=compute_device, dtype=torch.float32)

    residual = weight - dequantized_weight
    residual_scaled = residual @ scale

    # residual_scaled = residual
    torch.cuda.empty_cache()
    L, R_ = _low_rank_decomposition_loqer(residual_scaled, reduced_rank)

    R = torch.linalg.solve(scale, R_.T).T
    # R = R_

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


@torch.no_grad()
def TEST_check_equivalence():
    """
    This test verifies that
    - qLoRA initialization has (Lora_B @ Lora_A) == 0
    - HuggingFace's LoftQ implementation `hf_replace_lora_weights_loftq` with a dummy callback is equivalent to our reproduction of LoftQ official method `replace_lora_weights_loftq`
    - Official method `replace_lora_weights_loqer` is equivalent to (`replace_lora_weights_loftq` + 1 iteration + dummy scale)
    """
    from peft.utils.loftq_utils import replace_lora_weights_loftq as hf_replace_lora_weights_loftq

    def calculate_error(logits_ref, logits):
        return torch.pow(logits_ref - logits, 2).mean()

    # model_id = "bigscience/bloomz-560m"
    rank = 64
    model_id = "Cheng98/TinyLlama_v1.1"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    model.cuda()
    s = """Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!"""

    # FP32 ref
    inputs = tokenizer(s.splitlines(), return_tensors="pt", padding=True).to("cuda")
    logits_base = model(**inputs).logits

    dummy_scales = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            dummy_scales[name] = torch.diag(torch.ones(module.weight.shape[1])).float()

    # qlora
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
    model.eval()
    lora_config = LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear", r=rank)
    peft_model = get_peft_model(model, lora_config)

    logits_qlora = peft_model(**inputs).logits
    print("🔍 qlora error", calculate_error(logits_base, logits_qlora))

    # loftq (HF)
    def dummy_callback(model, module_name):
        return True

    current_mse = float("inf")

    def minimize_output_error_callback(model, module_name):
        """Callable to replace weights with LoFTQ if the mse is lower than the current best one."""
        nonlocal current_mse

        logits = model(**inputs).logits
        mse = calculate_error(logits_base, logits)
        if mse < current_mse:
            current_mse = mse
            return True
        return False

    hf_replace_lora_weights_loftq(peft_model, callback=dummy_callback)
    logits_loftq = peft_model(**inputs).logits
    print("🔍 loftq error (HF, dummy callback)", calculate_error(logits_base, logits_loftq))

    hf_replace_lora_weights_loftq(peft_model, callback=minimize_output_error_callback)
    logits_loftq = peft_model(**inputs).logits
    print("🔍 loftq error (HF, minimize output error callback, 1 iter)", calculate_error(logits_base, logits_loftq))
    hf_replace_lora_weights_loftq(peft_model, callback=minimize_output_error_callback)
    logits_loftq = peft_model(**inputs).logits
    print("🔍 loftq error (HF, minimize output error callback, 2 iter)", calculate_error(logits_base, logits_loftq))
    hf_replace_lora_weights_loftq(peft_model, callback=minimize_output_error_callback)
    logits_loftq = peft_model(**inputs).logits
    print("🔍 loftq error (HF, minimize output error callback, 3 iter)", calculate_error(logits_base, logits_loftq))
    hf_replace_lora_weights_loftq(peft_model, callback=minimize_output_error_callback)
    logits_loftq = peft_model(**inputs).logits
    print("🔍 loftq error (HF, minimize output error callback, 4 iter)", calculate_error(logits_base, logits_loftq))

    # loftq (Official)
    num_iters = 4
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
    model.eval()
    peft_model = get_peft_model(model, lora_config)

    replace_lora_weights_loftq(peft_model, num_iters=num_iters)
    print(
        f"🔍 loftq error (Official, num_iters={num_iters})", calculate_error(logits_base, peft_model(**inputs).logits)
    )

    # loqer+ (dummy = (loftq + 1 iter))
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
    model.eval()
    peft_model = get_peft_model(model, lora_config)

    replace_lora_weights_loqer(peft_model, scale_dict=dummy_scales)
    print("🔍 loqer+ error (dummy = (loftq + 1 iter))", calculate_error(logits_base, peft_model(**inputs).logits))


@torch.no_grad()
def TEST_LoQER():
    def calculate_error(logits_ref, logits):
        return torch.pow(logits_ref - logits, 2).mean()

    model_id = "Cheng98/TinyLlama_v1.1"
    loqer_scaling_mode = "diag"
    torch_dtype = torch.float32
    calibration_set = "wikitext2"
    calibration_batch_size = 1
    perplexity_max_seq_length = 2048
    num_calibration_samples = 128
    num_workers = 8
    rank = 64

    # profile scale dict
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, _attn_implementation="eager"
    )
    model.eval()
    model.cuda()

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    layers_to_register_and_share = find_layers_to_register_scale_hook(model)
    profiler_factory = register_scale_hooks(
        model,
        layers_to_register_and_share=layers_to_register_and_share,
        mode=loqer_scaling_mode,
        torch_dtype=torch_dtype,
        mode_map=None,
    )

    calibration_datamodule = get_data_module(
        calibration_set,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=perplexity_max_seq_length,
        num_raw_samples=28 * num_calibration_samples,
        num_workers=num_workers,
    )
    calibration_dataloader = DataLoader(
        calibration_datamodule["train"],
        batch_size=calibration_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
    )

    profile_outputs = evaluate_perplexity(
        model=model,
        eval_dataloader=calibration_dataloader,
        num_samples=num_calibration_samples,
        progress_bar=True,
        description="Calibrating",
    )
    print("profile_outputs", profile_outputs)

    profiler_factory.remove_all_hooks()
    scale_dict = profiler_factory.get_scale_dict(progress_bar=True)

    share_scales(scale_dict, layers_to_register_and_share)

    # test logits
    def calculate_error(logits_ref, logits):
        return torch.pow(logits_ref - logits, 2).mean()

    s = """Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!"""

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(s.splitlines(), return_tensors="pt", padding=True).to("cuda")

    # logits base
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    model.cuda()
    logits_base = model(**inputs).logits

    # qlora
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
    model.eval()
    lora_config = LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear", r=rank)
    peft_model = get_peft_model(model, lora_config)
    logits_qlora = peft_model(**inputs).logits
    print("🔍 qlora error", calculate_error(logits_base, logits_qlora))

    # loqer+
    replace_lora_weights_loqer(peft_model, scale_dict=scale_dict)
    logits_loqer = peft_model(**inputs).logits
    print("🔍 loqer+ error", calculate_error(logits_base, logits_loqer))


def TEST_perplexity():
    """
    If profiled on wikitext2, LoQER+ indeed has the lowest perplexity. Thus I guess the training will converge faster.

    Calibrating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:56<00:00,  2.27it/s]
    🔍 FP32 perplexity {'loss': 2.0786648616194725, 'perplexity': 7.993788971820187, 'num_samples': 128, 'seq_len': 2048, 'batch_size': 1}
    Computing scale: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 88/88 [00:00<00:00, 51478.21it/s]
    `low_cpu_mem_usage` was None, now set to True since model is quantized.
    qlora: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:18<00:00,  6.85it/s]
    🔍 qlora perplexity {'loss': 2.125852474011481, 'perplexity': 8.380038208843667, 'num_samples': 128, 'seq_len': 2048, 'batch_size': 1}
    Replacing LoRA adapters (LoftQ): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1702/1702 [02:49<00:00, 10.03it/s]
    loftq: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:18<00:00,  6.77it/s]
    🔍 loftq perplexity (4 iters) {'loss': 2.1296635707840323, 'perplexity': 8.412036280559624, 'num_samples': 128, 'seq_len': 2048, 'batch_size': 1}
    `low_cpu_mem_usage` was None, now set to True since model is quantized.
    Replacing LoRA adapter (LoQER+): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1702/1702 [00:43<00:00, 38.86it/s]
    loqer+: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:18<00:00,  6.75it/s]
    🔍 loqer+ perplexity {'loss': 2.121054043993354, 'perplexity': 8.339923502724137, 'num_samples': 128, 'seq_len': 2048, 'batch_size': 1}
    """
    model_id = "Cheng98/TinyLlama_v1.1"
    loqer_scaling_mode = "diag"
    torch_dtype = torch.float32
    calibration_set = "wikitext2"
    calibration_batch_size = 1
    perplexity_max_seq_length = 2048
    num_calibration_samples = 128
    num_workers = 8
    rank = 64
    bnb_4bit_quant_type = "nf4"

    # profile scale dict
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, _attn_implementation="eager"
    )
    model.eval()
    model.cuda()

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    layers_to_register_and_share = find_layers_to_register_scale_hook(model)
    profiler_factory = register_scale_hooks(
        model,
        layers_to_register_and_share=layers_to_register_and_share,
        mode=loqer_scaling_mode,
        torch_dtype=torch_dtype,
        mode_map=None,
    )

    calibration_datamodule = get_data_module(
        calibration_set,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=perplexity_max_seq_length,
        num_raw_samples=28 * num_calibration_samples,
        num_workers=num_workers,
    )
    calibration_dataloader = DataLoader(
        calibration_datamodule["train"],
        batch_size=calibration_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
    )

    profile_outputs = evaluate_perplexity(
        model=model,
        eval_dataloader=calibration_dataloader,
        num_samples=num_calibration_samples,
        progress_bar=True,
        description="Calibrating",
    )
    print("🔍 FP32 perplexity", profile_outputs)

    profiler_factory.remove_all_hooks()
    scale_dict = profiler_factory.get_scale_dict(progress_bar=True)

    share_scales(scale_dict, layers_to_register_and_share)

    # qlora
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
    model.eval()
    lora_config = LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear", r=rank)
    peft_model = get_peft_model(model, lora_config)
    peft_model.eval()
    peft_model.cuda()
    perplexity_qlora = evaluate_perplexity(
        model=peft_model,
        eval_dataloader=calibration_dataloader,
        num_samples=num_calibration_samples,
        progress_bar=True,
        description="qlora",
    )
    print("🔍 qlora perplexity", perplexity_qlora)

    num_iters = 4
    replace_lora_weights_loftq(peft_model, num_iters=num_iters)
    perplexity_loftq = evaluate_perplexity(
        model=peft_model,
        eval_dataloader=calibration_dataloader,
        num_samples=num_calibration_samples,
        progress_bar=True,
        description="loftq",
    )
    print(f"🔍 loftq perplexity ({num_iters} iters)", perplexity_loftq)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
    model.eval()
    lora_config = LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear", r=rank)
    peft_model = get_peft_model(model, lora_config)
    peft_model.eval()
    peft_model.cuda()
    replace_lora_weights_loqer(peft_model, scale_dict=scale_dict)
    perplexity_loqer = evaluate_perplexity(
        model=peft_model,
        eval_dataloader=calibration_dataloader,
        num_samples=num_calibration_samples,
        progress_bar=True,
        description="loqer+",
    )
    print("🔍 loqer+ perplexity", perplexity_loqer)


if __name__ == "__main__":
    # TEST_check_equivalence()
    # TEST_LoQER()
    TEST_perplexity()
