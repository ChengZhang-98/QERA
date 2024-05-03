import math
import torch

CLAMP_MIN = 1e-6


class ScaleHookFactoryDiagonal:
    """
    scale = diag( sqrt( E[ x_1^2]), sqrt( E[ x_2^2]), ..., sqrt( E[ x_n^2] ) )
    """

    def __init__(self) -> None:
        self.scales = {}
        self.n_samples = {}
        self.compute_device = None
        self.handles = []

    def get_scale_hook(self, name: str) -> callable:
        self.scales[name] = None
        self.n_samples[name] = 0

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.view(-1, x.shape[-1])
            num_samples, _ = x.shape
            x = x.pow(2).sum(0)

            self.n_samples[name] += num_samples
            if self.scales[name] is None:
                scale = x
                self.compute_device = x.device
            else:
                scale = self.scales[name].to(x.device)
                scale = scale + x

            self.scales[name] = scale

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self) -> dict[str, torch.Tensor]:
        for name in self.scales:
            scale = self.scales[name].to(self.compute_device)
            scale = torch.sqrt(scale / self.n_samples[name])
            scale = torch.clamp(scale, min=CLAMP_MIN)
            self.scales[name] = scale

        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def register_scale_hooks(
    model: torch.nn.Module,
    mode: str = "diagonal",
):
    if mode == "diagonal":
        hook_factory = ScaleHookFactoryDiagonal()
    else:
        raise ValueError(f"mode {mode} is not supported")

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "lm_head" in name:
            continue

        handle = module.register_forward_hook(hook_factory.get_scale_hook(name))
        hook_factory.handles.append(handle)

    return hook_factory
