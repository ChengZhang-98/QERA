import math
import logging
import torch

logger = logging.getLogger(__name__)

CLAMP_MIN = 1e-6
NUM_MATRIX_SQRT_ITERATIONS = 200


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
                if self.compute_device.type == "cpu":
                    logger.warning("Using CPU for computing scale, this may be slow")
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


def sqrt_newton_schulz(A, numIters=200):
    """Newton-Schulz iterations method to get matrix square root.

    Code copied from https://github.com/pytorch/pytorch/issues/25481

    Page 231, Eq 2.6b
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.6.8799&rep=rep1&type=pdf

    Args:
        A: the symmetric PSD matrix whose matrix square root be computed
        numIters: Maximum number of iterations.

    Returns:
        A^0.5

    Tensorflow Source:
        https://github.com/tensorflow/tensorflow/blob/df3a3375941b9e920667acfe72fb4c33a8f45503/tensorflow/contrib/opt/python/training/matrix_functions.py#L26C1-L73C42
    Torch Source:
        https://github.com/msubhransu/matrix-sqrt/blob/cc2289a3ed7042b8dbacd53ce8a34da1f814ed2f/matrix_sqrt.py#L74
    """

    normA = torch.linalg.matrix_norm(A, keepdim=True)
    err = normA + 1.0
    I = torch.eye(*A.shape[-2:], dtype=A.dtype, device=A.device)
    Z = torch.eye(*A.shape[-2:], dtype=A.dtype, device=A.device).expand_as(A)
    Y = A / normA
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y_new = Y.bmm(T)
        Z_new = T.bmm(Z)

        # This method require that we check for divergence every step.
        # Compute the error in approximation.
        mat_a_approx = torch.bmm(Y_new, Y_new) * normA
        residual = A - mat_a_approx
        current_err = torch.linalg.matrix_norm(residual, keepdim=True) / normA
        if torch.all(current_err > err):
            break

        err = current_err
        Y = Y_new
        Z = Z_new

    sA = Y * torch.sqrt(normA)

    return sA


class ScaleHookFactoryRxx:
    """
    For row vector x,
    scale = E[ x^T x ] ^ 0.5, Rxx = E[ x^T x ] is the auto-correlation matrix
    """

    def __init__(self) -> None:
        self.scales = {}
        self.n_samples = {}
        self.compute_device = None
        self.handles = []

    def get_scale_hook(self, name: str) -> callable:
        """ """

        self.scales[name] = None
        self.n_samples[name] = 0

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.reshape(-1, x.shape[-1])
            n_samples, in_features = x.shape
            if self.scales[name] is None:
                self.compute_device = x.device
                if self.compute_device.type == "cpu":
                    logger.warning("Using CPU for computing Rxx, this may be slow")
                self.scales[name] = torch.zeros(x.shape[-1], x.shape[-1], dtype=x.dtype)

            scales = self.scales[name].to(x.device)
            scales += torch.einsum("bi,bj->ij", x, x)  # batched outer product
            self.scales[name] = scales.to("cpu")
            self.n_samples[name] += n_samples

        return scale_hook

    def get_scale_dict(self) -> dict[str, torch.Tensor]:
        for name in self.scales:
            scale = self.scales[name].to(self.compute_device)
            scale = scale / self.n_samples[name]
            scale = sqrt_newton_schulz(scale.unsqueeze(0), numIters=NUM_MATRIX_SQRT_ITERATIONS).squeeze(0)
            self.scales[name] = scale.to("cpu")

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
    elif mode == "rxx":
        hook_factory = ScaleHookFactoryRxx()
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
