from .linear import LinearLoQER

QUANTIZED_MODULE_MAP = {
    "linear": {
        "loqer": LinearLoQER,
    }
}


def get_quantized_layer_cls(op: str, q_config: dict) -> LinearLoQER:
    assert op in QUANTIZED_MODULE_MAP, f"Unsupported quantized op: {op}"
    assert q_config["name"] in QUANTIZED_MODULE_MAP[op], f"Unsupported quantized config: {q_config}"
    return QUANTIZED_MODULE_MAP[op][q_config["name"]]
