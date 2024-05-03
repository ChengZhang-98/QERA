import sys
import re
import torch
from accelerate import infer_auto_device_map


def find_matched_pattern(query: str, patterns: list[str]) -> str | None:
    patterns: list[re.Pattern] = [re.compile(pattern) for pattern in patterns]

    matched_patterns = []

    for pattern in patterns:
        if pattern.fullmatch(query):
            matched_patterns.append(pattern)

    if len(matched_patterns) > 1:
        raise ValueError(f"Multiple patterns matched: {matched_patterns}")

    return matched_patterns[0].pattern if len(matched_patterns) == 1 else None


def get_layer_name(module, layer):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is layer:
            return name
    raise ValueError(f"Cannot find op {layer} in module {module}")


def get_layer_by_name(module, layer_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == layer_name:
            return m
    raise ValueError(f"Cannot find op {layer_name} in module {module}")


def set_layer_by_name(module, name, new_layer):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = module
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_layer)
    else:
        setattr(module, name, new_layer)


def setattr_recursive(obj, attr, value):
    if "." not in attr:
        setattr(obj, attr, value)
    else:
        layer = attr.split(".")
        setattr_recursive(getattr(obj, layer[0]), ".".join(layer[1:]), value)


def create_device_map(model, device_map) -> dict[str, int]:
    if device_map == "auto":
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=model._no_split_modules,
        )
    elif device_map == "auto-balance":
        max_memory = {i: torch.cuda.mem_get_info(i)[0] // 2 for i in range(torch.cuda.device_count())}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=model._no_split_modules,
            max_memory=max_memory,
        )
        n_devices = torch.cuda.device_count()
        n_decoder_layers = model.config.num_hidden_layers
        n_layers_per_device = n_decoder_layers // n_devices
        balanced_device_map = {}
        current_device = 0
        current_decoder_idx = 0

        for layer_name in device_map:
            if ".layers." in layer_name:
                if (current_decoder_idx + 1) % n_layers_per_device == 0:
                    current_device += 1
                current_decoder_idx += 1
            balanced_device_map[layer_name] = min(current_device, n_devices - 1)
        device_map = balanced_device_map
    else:
        assert isinstance(device_map, dict)
    return device_map


def enable_exception_hook(debugger="ipdb"):

    if debugger == "pudb":

        def excepthook(etype, evalue, etb):
            from IPython.core import ultratb
            import pudb

            ultratb.FormattedTB()(etype, evalue, etb)
            for exc in [KeyboardInterrupt, FileNotFoundError]:
                if issubclass(etype, exc):
                    sys.exit(-1)
            pudb.post_mortem(etb)

    elif debugger == "ipdb":

        def excepthook(etype, evalue, etb):
            from IPython.core import ultratb
            import ipdb

            ultratb.FormattedTB()(etype, evalue, etb)
            for exc in [KeyboardInterrupt, FileNotFoundError]:
                if issubclass(etype, exc):
                    sys.exit(-1)
            ipdb.post_mortem(etb)

    else:
        raise ValueError(f"Unknown debugger: {debugger}")
