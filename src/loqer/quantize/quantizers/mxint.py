import torch


def group_tensor(x: torch.Tensor, block_size: int, block_axis: int) -> tuple[torch.Tensor, tuple, tuple]:
    """Group the elements into blocks along the specified axis.
    - Only support 1D, 2D, or 3D tensor.
    - When x is 3D tensor, cannot group along batch axis (block_axis=0).
    - Use the view and permute to restore grouped x to the original shape.

    :param torch.Tensor x: 1D, 2D, or 3D tensor
    :param int block_size: number of elements in each block
    :param int block_axis: Group the elements into blocks along the specified axis
    :raises ValueError: illegal block_axis
    :raises NotImplementedError: illegal tensor dimension and shape
    :return tuple[torch.Tensor, tuple, tuple]: grouped tensor, view_args, permute_args

    .. code-block:: python

        >>> x = torch.arange(12).reshape(3, 4)
        >>> block_size = 2
        >>> block_axis = -1
        >>> print(x)
        tensor([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x, view_args, permute_args = _group_tensor(x, block_size, block_axis)
        >>> print(x)
        tensor([[ 0,  1],
                [ 2,  3],
                [ 4,  5],
                [ 6,  7],
                [ 8,  9],
                [10, 11]])
        >>> print(x.view(view_args).permute(permute_args))
        tensor([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
    """

    if block_axis < 0:
        block_axis = x.ndim + block_axis

    ori_shape = x.size()
    if x.ndim == 1:
        return x.reshape(-1, block_size), ori_shape, (0,)
    elif x.ndim == 2:
        if block_axis == 0:
            permute_args = (1, 0)
            x = x.permute(1, 0)
            view_args = x.size()
            x = x.contiguous()
            return x.reshape(-1, block_size), view_args, permute_args
        else:
            permute_args = (0, 1)
            return x.view(-1, block_size), ori_shape, permute_args
    elif x.ndim == 3:
        if block_axis == 1:
            permute_args = (0, 2, 1)
            x = x.permute(0, 2, 1)
            view_args = x.size()
            x = x.contiguous()
            return x.reshape(-1, block_size), view_args, permute_args
        elif block_axis == 2:
            permute_args = (0, 1, 2)
            view_args = x.size()
            return x.reshape(-1, block_size), view_args, permute_args
        else:
            raise ValueError("cannot group along batch axis for 3D tensor")
    else:
        raise NotImplementedError("Only support 1D, 2D tensor, and 3D activation tensor")


def pad_zeros_if_necessary(x: torch.Tensor, block_size: int, block_axis: int) -> torch.Tensor:
    """Append zeros to x if the number of elements along block_axis is not a multiple of block_size, else return x.

    :param torch.Tensor x: input tensor
    :param int block_size: number of elements in each block
    :param int block_axis: group the elements into blocks along the specified axis
    :return torch.Tensor: padded tensor
    """

    if x.shape[block_axis] % block_size == 0:
        return x

    pad_size = block_size - x.shape[block_axis] % block_size
    pad_shape = list(x.shape)
    pad_shape[block_axis] = pad_size
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    x = torch.cat([x, pad], dim=block_axis)
    return x


@torch.no_grad()
def extract_bf16_components(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract sign, exponent, and mantissa from bfloat16 tensor.
    - Note that subnormal numbers will be set to 0 before extraction.
    - The leading 1 is not included in the mantissa.
    - bfloat16 (https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) format has:
        - 1 bit for sign
        - 8 bits for exponent, exponent bias = 127
        - 7 bits for mantissa

    :param torch.Tensor x: torch.bfloat16 tensor
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: sign (torch.bool), exponent (torch.uint8), mantissa (torch.uint8), is_zero (torch.bool)
    """

    assert x.dtype == torch.bfloat16, "Only support torch.bfloat16 dtype"

    sign = x < 0
    x = x.abs()

    # set subnormal numbers to 0
    is_normal = x >= torch.finfo(torch.bfloat16).smallest_normal
    x = torch.where(is_normal, x, torch.zeros_like(x))

    is_zero = x == 0

    # "re" = reinterpret
    re_int_x = x.view(dtype=torch.int16)

    # get the exponent by removing the 7-bit mantissa (>>7)
    exponent = re_int_x.bitwise_right_shift(7).to(dtype=torch.uint8)
    # get the mantissa by masking the 8-bit exponent
    mantissa = re_int_x.bitwise_and(0x7F).to(dtype=torch.uint8)

    return sign, exponent, mantissa, is_zero


@torch.no_grad()
def compose_bf16_components(sign, exponent, mantissa):
    assert sign.dtype == torch.bool, "sign must be torch.bool"
    assert exponent.dtype == torch.uint8, "exponent must be torch.uint8"
    assert mantissa.dtype == torch.uint8, "mantissa must be torch.uint8"
    # mask the implicit leading bit
    mantissa = mantissa.bitwise_and(0x7F)
    # "re" = reinterpret
    # sign << 15
    re_int_x = sign.to(dtype=torch.int16).bitwise_left_shift(15)
    # exponent << 7
    re_int_x = re_int_x.bitwise_or(exponent.to(dtype=torch.int16).bitwise_left_shift(7))
    # mantissa
    re_int_x = re_int_x.bitwise_or(mantissa.to(dtype=torch.int16))
    re_bf_x = re_int_x.view(dtype=torch.bfloat16)
    return re_bf_x


def _check_shape_mxint(x: torch.Tensor, block_size: int, block_axis: int):
    assert x.ndim >= 1, "x must have at least 1 dimension"
    # assert (
    #     x.shape[block_axis] % block_size == 0
    # ), f"block_size (={block_size}) must divide the number of elements along block_axis (= {x.shape[block_axis]})"

    if x.ndim == 1:
        assert block_axis in [0, -1], "block_axis must be 0 or -1 for 1D tensor"
    elif x.ndim == 2:
        assert block_axis in [0, 1, -1, -2], "block_axis must be 0, 1, -1, or -2 for 2D tensor"
    elif x.ndim == 3:
        assert block_axis != 0, "cannot group along batch axis for 3D tensor"
        assert block_axis in [1, 2, -2, -1], "block_axis must be 1, 2, -2, or -1 for 3D tensor"
    else:
        raise NotImplementedError("Only support 1D, 2D tensor, and 3D activation tensor")
    return True


@torch.no_grad()
def quantize_bf16_to_mxint(
    x: torch.Tensor, width: int, block_size: int, block_axis: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, tuple]]:
    """Cast bfloat16 tensor to mxint tensor. Round mantissa by truncating or rounding to nearest even.

    :param torch.Tensor x: input 1D, 2D, or 3D tensor
    :param int width: number of bits for the shared mantissa: 1-bit sign + (width-1)-bit mantissa
    :param int block_size: number of elements in each block
    :param int block_axis: group the elements into blocks along the specified axis
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, tuple]]: sign (torch.bool), exponent (torch.uint8), mantissa (torch.uint8), reshape_kwargs
    """

    assert x.dtype == torch.bfloat16, "Only support torch.bfloat16 dtype"
    assert width <= 8, "width must be <= 8"
    assert _check_shape_mxint(x, block_size, block_axis)

    # quantize
    # group the elements into blocks along the specified axis
    ori_shape = x.size()
    x = pad_zeros_if_necessary(x, block_size, block_axis)  # padded x
    x, view_args, permute_args = group_tensor(x, block_size, block_axis)  # [num_blocks, block_size]

    # extract sign, exponent, mantissa for each element
    sign, exponent, mantissa, is_zero = extract_bf16_components(x)

    # find max exponent in each block
    group_max_exp = exponent.max(dim=1, keepdim=True).values  # [num_blocks, 1]

    # shift the mantissa to the left by (group_max_exp - exponent)
    mantissa_shift = group_max_exp - exponent
    set_leading_1 = (mantissa_shift != 0) & (~is_zero)
    # before that, set the leading 1 for (the mantissa whose exponent is not the max) to 1
    mantissa = torch.where(set_leading_1, mantissa | 0x80, mantissa)
    # *: shift right for truncating rounding
    mantissa = mantissa.bitwise_right_shift(mantissa_shift)
    # mask the implicit leading bit since mantissa should have 7 bit
    truncation_mask = (255 << (8 - width)) & 0x7F
    mantissa = mantissa.bitwise_left_shift(mantissa_shift).bitwise_and(truncation_mask)

    reshape_kwargs = {
        "view_args": view_args,
        "permute_args": permute_args,
        "orig_shape": ori_shape,
    }

    return sign, exponent, mantissa, reshape_kwargs


@torch.no_grad()
def dequantize_mxint_to_bf16(sign, exponent, mantissa, reshape_kwargs):
    """Compose mxint components to bfloat16 tensor."""
    view_args = reshape_kwargs["view_args"]
    permute_args = reshape_kwargs["permute_args"]
    ori_shape = reshape_kwargs["orig_shape"]
    x = compose_bf16_components(sign, exponent, mantissa)
    x = x.view(view_args).permute(permute_args)

    # if len(ori_shape) == n, then slice x to ori_shape by x[:ori_shape[0], :ori_shape[1], ..., :ori_shape[n-1]]
    x = x[tuple(slice(ori_shape[i]) for i in range(len(ori_shape)))]
    return x


@torch.no_grad()
def emulated_quantizer_bf16_to_mxint(x: torch.Tensor, width: int, block_size: int, block_axis: int) -> torch.Tensor:
    """Emulated quantizer from bfloat16 to mxint8.

    :param torch.Tensor x: torch.bfloat16 tensor
    :param int block_size: number of elements in each block
    :param int block_axis: group the elements into blocks along the specified axis
    :return torch.Tensor: emulated mxint tensor with the same shape as x, dtype=torch.bfloat16
    """
    sign, exponent, mantissa, reshape_kwargs = quantize_bf16_to_mxint(x, width, block_size, block_axis)
    x = dequantize_mxint_to_bf16(sign, exponent, mantissa, reshape_kwargs)
    return x


@torch.no_grad()
def emulated_mxint_quantizer(x: torch.Tensor, width: int, block_size: int, block_axis: int) -> torch.Tensor:
    assert x.dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]
    x = x.to(dtype=torch.bfloat16)
    return emulated_quantizer_bf16_to_mxint(x, width, block_size, block_axis)


class MXINTQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        width: int,
        block_size: int,
        block_axis: int,
    ):
        return emulated_mxint_quantizer(x, width, block_size, block_axis)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def mxint_quantizer(x: torch.Tensor, width: int, block_size: int, block_axis: int):
    """Emulated quantizer from bfloat16 to mxint8.

    :param torch.Tensor x: torch.bfloat16 tensor
    :param int block_size: number of elements in each block
    :param int block_axis: group the elements into blocks along the specified axis
    :return torch.Tensor: emulated mxint tensor with the same shape as x, dtype=torch.bfloat16
    """
    return MXINTQuantize.apply(x, width, block_size, block_axis)
