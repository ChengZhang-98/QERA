import torch
from torch.utils.data import DataLoader
import transformers
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft import TaskType
from tabulate import tabulate
from tqdm.auto import tqdm
import pandas as pd
import yaml
import datetime

from loqer.datasets import get_data_module_for_peft
from loqer.models import find_layers_to_register_scale_hook
from loqer.statistic_profiler import register_scale_hooks, share_scales
from loqer.fine_tuning import (
    replace_lora_weights_loftq_4bit,
    replace_lora_weights_loqer_4bit,
    replace_lora_weights_loftq_kbit,
    replace_lora_weight_qlora_kbit,
    replace_lora_weight_loqer_kbit,
)


@torch.no_grad()
def collect_logits_error(model, batches, logits_ref):
    model.eval()
    logits_adapted = []
    for batch in batches:
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs_adapted = model(**batch)
        logits_adapted.append(outputs_adapted.logits)

    error = 0
    for logits_a, logits_r in zip(logits_adapted, logits_ref):
        error += torch.nn.functional.mse_loss(logits_a, logits_r).cpu().item()
    error = error / len(logits_adapted)

    return error


@torch.no_grad()
def create_adapted_model(
    model_name,
    lora_target_modules,
    lora_rank,
    adapter_init,
    quant_type,
    quant_bits,
    loftq_num_iters,
    scale_dict,
    mxint_block_size,
):
    bnb_config = None
    if quant_type in ["fp", "nf"] and quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4" if quant_type == "fp" else "nf4",
            llm_int8_skip_modules=["classifier", "lm_head"],
        )
        model = transformers.AutoModelForMaskedLM.from_pretrained(model_name, quantization_config=bnb_config)
        model.eval()
    else:
        model = transformers.AutoModelForMaskedLM.from_pretrained(model_name, _attn_implementation="eager")
        model.cuda()

    lora_config = LoraConfig(
        inference_mode=True,
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.1,
        target_modules=lora_target_modules,
        init_lora_weights=True,
    )
    model = get_peft_model(model, lora_config)
    model.eval()

    error_dict = None
    if adapter_init == "loqer":
        if bnb_config is not None:
            error_dict = replace_lora_weights_loqer_4bit(model, scale_dict=scale_dict)
        else:
            error_dict = replace_lora_weight_loqer_kbit(
                model,
                scale_dict=scale_dict,
                quant_type=quant_type,
                num_bits=quant_bits,
                mxint_block_size=mxint_block_size,
            )
    elif adapter_init == "loftq":
        if bnb_config is not None:
            error_dict = replace_lora_weights_loftq_4bit(model, num_iters=loftq_num_iters)
        else:
            error_dict = replace_lora_weights_loftq_kbit(
                model,
                quant_type=quant_type,
                num_bits=quant_bits,
                num_iters=loftq_num_iters,
                mxint_block_size=mxint_block_size,
            )
    else:
        if bnb_config is not None:
            pass
        else:
            error_dict = replace_lora_weight_qlora_kbit(
                model,
                quant_type=quant_type,
                num_bits=quant_bits,
                mxint_block_size=mxint_block_size,
            )

    return model, error_dict


@torch.no_grad()
def profile_scales(model_ref, loqer_scaling_mode, dataloader, num_batches):
    layers_to_register_and_share = find_layers_to_register_scale_hook(model_ref)
    profiler_factory = register_scale_hooks(model_ref, layers_to_register_and_share, loqer_scaling_mode, torch.float32)

    model_ref.eval()
    model_ref.cuda()
    for i, batch in enumerate(dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        _ = model_ref(**batch)
        if i >= num_batches:
            break
    profiler_factory.remove_all_hooks()
    scale_dict = profiler_factory.get_scale_dict(True)
    share_scales(scale_dict, layers_to_register_and_share)
    return scale_dict


@torch.no_grad()
def sweep(
    quant_type,
    ranks: list[int],
    bits: list[int],
    adapter_init_methods: list[str],
    loftq_num_iters: list[int],
    model_name,
    dataloader,
    num_batches,
):
    LOQER_SCALING_MODE = "diag"
    MXINT_BLOCK_SIZE = 64
    if "roberta" in model_name:
        lora_target_modules = r"roberta\.encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|(attention\.output\.dense)|(intermediate\.dense)|(output\.dense))"
    else:
        raise RuntimeError(f"Unsupported model: {model_name}")

    model_ref = transformers.AutoModelForMaskedLM.from_pretrained(model_name, _attn_implementation="eager")

    scale_dict = profile_scales(
        model_ref, loqer_scaling_mode=LOQER_SCALING_MODE, dataloader=dataloader, num_batches=num_batches
    )

    batches = []
    logits_ref = []
    for i in range(num_batches):
        batch = next(iter(dataloader))
        batches.append(batch)
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs_ref = model_ref(**batch)
        logits_ref.append(outputs_ref.logits)

    output_error = []
    approx_error = []

    total_num_iters = 0
    for rank in ranks:
        for bit in bits:
            for adapter_init in adapter_init_methods:
                if adapter_init != "loftq":
                    total_num_iters += 1
                else:
                    total_num_iters += len(loftq_num_iters)
    prog_bar = tqdm(total=total_num_iters, desc="Sweeping")

    for rank in ranks:
        for bit in bits:
            for adapter_init in adapter_init_methods:
                if adapter_init != "loftq":
                    loftq_num_iters_ = [0]
                else:
                    loftq_num_iters_ = loftq_num_iters

                for loftq_num_iter in loftq_num_iters_:
                    adapted_model, error_dict = create_adapted_model(
                        model_name=model_name,
                        lora_target_modules=lora_target_modules,
                        lora_rank=rank,
                        adapter_init=adapter_init,
                        quant_type=quant_type,
                        quant_bits=bit,
                        loftq_num_iters=loftq_num_iter,
                        scale_dict=scale_dict,
                        mxint_block_size=MXINT_BLOCK_SIZE,
                    )

                    error = collect_logits_error(
                        model=adapted_model,
                        batches=batches,
                        logits_ref=logits_ref,
                    )

                    adp_name = adapter_init if adapter_init != "loftq" else f"{adapter_init} ({loftq_num_iter}-iter)"
                    output_error.append([rank, bit, adp_name, error, quant_type])
                    if not (adapter_init == "loftq" and loftq_num_iter != max(loftq_num_iters)):
                        approx_error.append(
                            dict(
                                rank=rank,
                                quant_type=quant_type,
                                n_bits=bit,
                                adapter_init=adp_name,
                                approx_error=error_dict,
                                output_error=error,
                            )
                        )
                    prog_bar.update(1)

    return output_error, approx_error


if __name__ == "__main__":
    transformers.set_seed(42)
    model_name = "Cheng98/roberta-base"
    num_batches = 64
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    calibration_datamodule = get_data_module_for_peft(
        "wikitext2_mlm",
        tokenizer=tokenizer,
        model_config=None,
        pad_to_max_length=True,
        max_length=512,
        num_workers=8,
        overwrite_cache=False,
    )
    calibration_dataloader = DataLoader(
        calibration_datamodule["train"],
        batch_size=2,
        shuffle=False,
        num_workers=8,
        collate_fn=data_collator,
    )

    # model_ref = transformers.AutoModelForMaskedLM.from_pretrained(model_name, _attn_implementation="eager")
    # model_ref.cuda()

    # collect_logits_error(model_ref=model_ref, model=model_ref, batches=calibration_dataloader, num_batches=num_batches)

    output_error, approx_error = sweep(
        quant_type="mxint",
        ranks=[4, 16],
        bits=[3],
        adapter_init_methods=["loftq", "loqer"],
        loftq_num_iters=[1, 2, 3, 4, 5],
        model_name=model_name,
        dataloader=calibration_dataloader,
        num_batches=64,
    )

    # print(tabulate(output_error, headers=["Rank", "Bits", "Adapter Init", "Output Error", "Quant Type"]))

    df = pd.DataFrame(data=output_error, columns=["rank", "bits", "adapter_init", "output_error", "quant_type"])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    df.to_pickle(f"roberta_output_error_{timestamp}.pkl")

    if False:
        with open(f"roberta_approx_error_{timestamp}.yaml", "w") as f:
            yaml.dump(approx_error, f)

    print(df)
