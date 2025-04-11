"""Replace nn.Modules to quantized modules"""
# pylint: disable=C0116,W0123,W0611
from typing import Optional, Tuple

import torch
from torch import nn
from torch.ao.nn import quantized as nnq


__all__ = [
    "replace_nn_modules",
]

REPLACEABLE_MODULES = [
    "Conv2d",
    "Linear",
]


class QuantModuleWrapper(nn.Module):
    """
    Return a wrapped module that contains quantize -> module -> dequantize
    flow

    Args:
        module (nn.Module): nnq.Module, like nnq.Conv2d or nnq.Linear
        scales (float or Tensor of float): quantization scale(s)
        zero_point (int or Tensor of int): quantization zero_point(s)
        dtype (torch dtype): quantized dtype, QInt8 by default

    """

    def __init__(
        self,
        module: nn.Module,
        scales: Optional[float],
        zero_points: Optional[int],
        dtype=torch.qint8,
    ):
        super().__init__()

        self.qmodule = module
        self.scales = scales
        self.zero_points = zero_points
        self.dtype = dtype

    def forward(self, x):
        if isinstance(
            x, Tuple
        ):  # for modules like ConvAdd2d, assume inputs are all quantized
            x = self.qmodule(*x)
        elif isinstance(self.scales, float) or (
            isinstance(x, torch.Tensor) and self.scales.numel() == 1
        ):
            x = torch.quantize_per_tensor(
                x, self.scales.item(), self.zero_points.item(), self.dtype
            )
            x = self.qmodule(x)
        elif self.scales.numel() > 1:
            x = torch.quantize_per_channel(
                x, self.scales, self.zero_points, x.ndim - 1, self.dtype
            )
            x = self.qmodule(x)
        else:
            raise ValueError(f"Unsupported input: {type(x)}")

        return x.dequantize()


def replace_nn_modules(
    module: nn.Module, act_pre_process: nn.Module, dtype=torch.qint8
):
    """replace nn.Modules to nnq.Modules"""
    module_name = module._get_name()
    assert module_name in REPLACEABLE_MODULES, f"Module {module_name} is in-replaceable"

    qmodule = eval(f"nnq.{module_name}.from_float(module)")
    scales, zero_points = act_pre_process.calculate_qparams()
    out_mod = QuantModuleWrapper(qmodule, scales, zero_points, dtype)

    return out_mod
