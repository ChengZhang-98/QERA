import logging
from copy import deepcopy

from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2Layer,
    DebertaV2Encoder,
)

from ..quantize import get_quantized_layer_cls
from ..utils import find_matched_pattern, get_layer_name, set_layer_by_name

logger = logging.getLogger(__name__)


def build_loqer_config_deberta_v2(model: DebertaV2Model, loqer_config: dict):
    assert isinstance(model, DebertaV2Model)
    parsed_config = {}

    encoder: DebertaV2Encoder = model.encoder

    for module in encoder.modules():
        if not isinstance(module, nn.Linear):
            continue

        fc_name = get_layer_name(model, module)
        matched_entry = find_matched_pattern(fc_name, loqer_config.keys())
        assert matched_entry is not None, f"Cannot find matched entry for {fc_name}"
        if isinstance(loqer_config[matched_entry], str):
            matched_entry = loqer_config[matched_entry]

        parsed_config[fc_name] = deepcopy(loqer_config[matched_entry])
    return parsed_config


def quantize_deberta_v2(model: DebertaV2Model, loqer_config: dict):
    assert isinstance(model, DebertaV2Model)
    loqer_config = build_loqer_config_deberta_v2(model, loqer_config)

    for module in model.encoder.modules():
        if not isinstance(module, nn.Linear):
            continue

        fc_name = get_layer_name(model, module)
        if fc_name not in loqer_config:
            raise ValueError(f"Cannot find {fc_name} in loqer_config")

        quantized_fc_cls = get_quantized_layer_cls("linear", loqer_config[fc_name])
        new_fc = quantized_fc_cls(
            module.in_features, module.out_features, bias=module.bias is not None, q_config=loqer_config[fc_name]
        )
        new_fc.load_state_dict(module.state_dict())
        set_layer_by_name(model, fc_name, new_fc)

    model._no_split_modules = ["DebertaV2Layer"]

    return model


def find_layers_to_register_scale_hook_deberta_v2(model: DebertaV2Model):
    assert isinstance(model, DebertaV2Model)
    assert model.config._attn_implementation == "eager"
    layers_to_register = []

    decoder_layer: DebertaV2Layer
    for decoder_layer in model.encoder.layer:
        k_name = get_layer_name(model, decoder_layer.attention.self.key_proj)
        q_name = get_layer_name(model, decoder_layer.attention.self.query_proj)
        v_name = get_layer_name(model, decoder_layer.attention.self.value_proj)
        layers_to_register.append(dict(target_layer=k_name, layers_sharing_scale=[q_name, v_name]))

        o_name = get_layer_name(model, decoder_layer.attention.output.dense)
        layers_to_register.append(dict(target_layer=o_name, layers_sharing_scale=[]))

        fc1_name = get_layer_name(model, decoder_layer.intermediate.dense)
        layers_to_register.append(dict(target_layer=fc1_name, layers_sharing_scale=[]))

        fc2_name = get_layer_name(model, decoder_layer.output.dense)
        layers_to_register.append(dict(target_layer=fc2_name, layers_sharing_scale=[]))

    return layers_to_register


def find_layers_to_approximate_deberta_v2(model: DebertaV2Model):
    assert isinstance(model, DebertaV2Model)
    layers_to_approximate = []
    for layer_name, layer in model.encoder.layer.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        layers_to_approximate.append(layer_name)
    return layers_to_approximate
