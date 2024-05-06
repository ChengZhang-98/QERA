import math
import logging
import torch
import numpy as np
from scipy import linalg as spla
from numpy import linalg as la
import tqdm

logger = logging.getLogger(__name__)

# CLAMP_MIN = 1e-6
NUM_MATRIX_SQRT_ITERATIONS = 200


class ScaleHookFactoryDiagonal:
    """
    scale = diag( sqrt( E[ x_1^2]), sqrt( E[ x_2^2]), ..., sqrt( E[ x_n^2] ) )
    """

    def __init__(self, torch_dtype) -> None:
        self.scales = {}
        self.n_samples = {}
        self.compute_devices = {}
        self.torch_dtype = torch_dtype
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
                self.compute_devices[name] = x.device
                if self.torch_dtype is None:
                    self.torch_dtype = x.dtype
                if self.compute_devices[name].type == "cpu":
                    logger.warning("Using CPU for computing scale, this may be slow")
                scale = x.to(self.torch_dtype)
            else:
                scale = self.scales[name].to(self.compute_devices[name])
                scale = scale + x.to(self.torch_dtype)

            self.scales[name] = scale

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self, progress_bar=False) -> dict[str, torch.Tensor]:
        scale_names_prog_bar = tqdm.tqdm(
            self.scales, desc="Computing scale", disable=not progress_bar, total=len(self.scales)
        )

        for name in scale_names_prog_bar:
            # for name in self.scales:
            scale = self.scales[name].to(self.compute_devices[name])
            scale = torch.sqrt(scale) * (1 / math.sqrt(self.n_samples[name]))
            # scale = torch.clamp(scale, min=CLAMP_MIN)
            self.scales[name] = scale

        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def sqrtm_newton_schulz(A, numIters=200):
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


# def find_nearestPD(A):
#     """Find the nearest positive-definite matrix to input

#     A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
#     credits [2].

#     [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

#     [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
#     matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
#     """

#     B = (A + A.T) / 2
#     _, s, V = la.svd(B)

#     H = np.dot(V.T, np.dot(np.diag(s), V))

#     A2 = (B + H) / 2

#     A3 = (A2 + A2.T) / 2

#     if isPD(A3):
#         return A3

#     spacing = np.spacing(la.norm(A))
#     # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
#     # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
#     # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
#     # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
#     # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
#     # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
#     # `spacing` will, for Gaussian random matrixes of small dimension, be on
#     # othe order of 1e-16. In practice, both ways converge, as the unit test
#     # below suggests.
#     I = np.eye(A.shape[0])
#     k = 1
#     while not isPD(A3):
#         mineig = np.min(np.real(la.eigvals(A3)))
#         A3 += I * (-mineig * k**2 + spacing)
#         k += 1

#     return A3


# def isPD(B):
#     """Returns true when input is positive-definite, via Cholesky"""
#     try:
#         _ = la.cholesky(B)
#         return True
#     except la.LinAlgError:
#         return False


def sqrtm_scipy(A: np.ndarray):
    if not isinstance(A, np.ndarray):
        raise RuntimeError("input matrix must be a numpy array")
    A_sqrt = spla.sqrtm(A)
    return A_sqrt


class ScaleHookFactoryRxx:
    """
    For row vector x,
    scale = E[ x^T x ] ^ 0.5, where Rxx = E[ x^T x ] is the auto-correlation matrix

    For numerical stability, we compute (x^T x) in torch_dtype (float32 is preferred) and accumulate in float64 (hard-coded).
    Then sqrt is computed in float64 (hard-coded).
    """

    def __init__(self, torch_dtype) -> None:
        self.scales = {}
        self.n_samples = {}
        self.compute_devices = {}
        self.torch_dtype = torch_dtype
        self.handles = []

    @torch.no_grad()
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
                if self.torch_dtype is None:
                    self.torch_dtype = x.dtype
                self.compute_devices[name] = x.device
                if self.compute_devices[name].type == "cpu":
                    logger.warning("Using CPU for computing Rxx, this may be slow")
                self.scales[name] = torch.zeros(in_features, in_features, dtype=torch.float64)  # *: hard-coded float64

            compute_device = self.compute_devices[name]
            scales = self.scales[name].to(compute_device)
            x = x.to(self.torch_dtype)
            # batched outer product
            # *: outer product in self.torch_dtype (float32 is preferred), then accumulate in float64
            delta = torch.einsum("bi,bj->ij", x, x).to(torch.float64)  # *: hard-coded float64
            scales += delta
            self.scales[name] = scales.cpu()
            self.n_samples[name] += n_samples

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(
        self, progress_bar=False, sqrtm_implementation: str = "newton_schulz", sqrtm_num_iters: int = 200
    ) -> dict[str, torch.Tensor]:

        if sqrtm_implementation == "iterative":
            ...
            scale_names_prog_bar = tqdm.tqdm(
                self.scales, desc="Computing scale", disable=not progress_bar, total=len(self.scales)
            )
            for name in scale_names_prog_bar:
                compute_device = self.compute_devices[name]
                scale = self.scales[name].to(compute_device)
                scale = scale.unsqueeze(0)
                scale = sqrtm_newton_schulz(scale, numIters=sqrtm_num_iters)
                scale = scale.squeeze(0)
                scale = scale * (1 / math.sqrt(self.n_samples[name]))
                scale = scale.cpu()
                self.scales[name] = scale
        elif sqrtm_implementation == "blocked":
            # convert to numpy
            for name in self.scales:
                self.scales[name] = self.scales[name].numpy()

            # compute the square root
            scale_names_prog_bar = tqdm.tqdm(
                self.scales, desc="Computing scale", total=len(self.scales), disable=not progress_bar
            )
            for name in scale_names_prog_bar:
                scale = self.scales[name]
                scale = sqrtm_scipy(scale)
                self.scales[name] = scale

            # convert to torch tensor
            for name in self.scales:
                scale = self.scales[name]
                n_samples = self.n_samples[name]
                scale = torch.from_numpy(scale).to(self.torch_dtype).to(self.compute_devices[name])
                scale = scale * (1 / math.sqrt(n_samples))
                self.scales[name] = scale
        else:
            raise ValueError(f"Unknown sqrtm_implementation: {sqrtm_implementation}")

        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class ScaleHookFactoryDummy:
    """
    dummy scale, which is torch.ones
    """

    def __init__(self, torch_dtype) -> None:
        self.scales = {}
        self.in_features = {}
        self.handles = []
        self.torch_dtype = None

    def get_scale_hook(self, name: str) -> callable:
        self.scales[name] = None
        self.in_features[name] = None

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ):
            if self.in_features[name] is None:
                if self.torch_dtype is None:
                    self.torch_dtype = input[0].dtype
                x = input[0]
                x = x.view(-1, x.shape[-1])
                self.in_features[name] = x.shape[-1]

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        for name in self.scales:
            self.scales[name] = torch.ones(self.in_features[name], dtype=self.torch_dtype)
        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def register_scale_hooks(
    model: torch.nn.Module,
    mode: str = "diagonal",
    torch_dtype: torch.dtype = None,
):
    if mode in ["diagonal", "diag"]:
        hook_factory = ScaleHookFactoryDiagonal(torch_dtype)
    elif mode == "rxx":
        hook_factory = ScaleHookFactoryRxx(torch_dtype)
    elif mode == "dummy":
        hook_factory = ScaleHookFactoryDummy()
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
