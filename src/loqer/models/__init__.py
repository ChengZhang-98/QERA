import logging

import torch
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)
from transformers.models.opt.modeling_opt import (
    OPTForCausalLM,
    OPTForSequenceClassification,
)
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
    MistralForSequenceClassification,
)
from .llama_decoder import (
    quantize_llama_model,
    find_layers_to_approximate_llama,
    find_layers_to_register_scale_hook_llama,
)
from .opt_decoder import quantize_opt_model, find_layers_to_approximate_opt
from .mistral_decoder import quantize_mistral_model, find_layers_to_approximate_mistral

logger = logging.getLogger(__name__)


@torch.no_grad()
def quantize_model(model, loqer_config) -> None:
    if isinstance(model, (LlamaForCausalLM, LlamaForSequenceClassification)):
        q_model = quantize_llama_model(model, loqer_config)
    elif isinstance(model, (OPTForCausalLM, OPTForSequenceClassification)):
        q_model = quantize_opt_model(model, loqer_config)
    elif isinstance(model, (MistralForCausalLM, MistralForSequenceClassification)):
        q_model = quantize_mistral_model(model, loqer_config)
    else:
        msg = f"Model {type(model).__name__} not supported for quantization"
        raise NotImplementedError(msg)

    logger.debug("Quantized model: %s", q_model)
    return q_model


def find_layers_to_approximate(model):
    if isinstance(model, (LlamaForCausalLM, LlamaForSequenceClassification)):
        return find_layers_to_approximate_llama(model)
    elif isinstance(model, (OPTForCausalLM, OPTForSequenceClassification)):
        return find_layers_to_approximate_opt(model)
    elif isinstance(model, (MistralForCausalLM, MistralForSequenceClassification)):
        return find_layers_to_approximate_mistral(model)
    else:
        msg = f"Model {type(model).__name__} not supported for layer approximation"
        raise NotImplementedError(msg)


def find_layers_to_register_scale_hook(model):
    if isinstance(model, (LlamaForCausalLM, LlamaForSequenceClassification)):
        return find_layers_to_register_scale_hook_llama(model)
    else:
        msg = f"Model {type(model).__name__} not supported for scale hook registration"
        raise NotImplementedError(msg)
