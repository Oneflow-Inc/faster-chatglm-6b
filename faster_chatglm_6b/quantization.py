# This code is base on https://huggingface.co/THUDM/chatglm-6b/blob/main/quantization.py
from torch.nn import Linear
from torch.nn.parameter import Parameter

import torch
import os

import numpy as np

def _pack_int8_to_int4(x):
    np_x = x.numpy()
    l = np_x[..., 0::2]
    r = np_x[..., 1::2]
    l = np.left_shift(l, 4)
    r = np.bitwise_and(r, np.int8(0xF))
    packed = torch.tensor(np.bitwise_or(l, r), device=x.device)
    return packed

class QuantizedLinear(Linear):
    def __init__(self, weight_bit_width: int, weight_tensor=None, bias_tensor=None, *args, **kwargs):
        super(QuantizedLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        shape = self.weight.shape
        del self.weight

        use_uint8 = os.environ.get("ONEFLOW_CHATGLM_USE_UINT8", "1") == "1"

        if weight_tensor is None:
            self.weight = torch.empty(
                shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
            )
            self.weight_scale = torch.empty(shape[0], dtype=kwargs["params_dtype"], device=kwargs["device"])
        else:
            if use_uint8:
                self.symmetric = False
                min_values = weight_tensor.float().min(dim=-1, keepdim=True).values
                max_values = weight_tensor.float().max(dim=-1, keepdim=True).values
                self.weight_scale = ((max_values - min_values) / float(2 ** weight_bit_width - 1)).half()
                self.weight = torch.round((weight_tensor - min_values) / self.weight_scale).to(torch.uint8)
                self.weight_zero = min_values.half()
                if weight_bit_width == 4:
                    self.weight = _pack_int8_to_int4(self.weight)
            else:
                self.symmetric = True
                self.weight_scale = (weight_tensor.abs().float().max(dim=-1, keepdim=True).values / ((2 ** (weight_bit_width - 1)) - 1)).half()
                self.weight = torch.round(weight_tensor / self.weight_scale).to(torch.int8)
                self.weight_zero = None
                if weight_bit_width == 4:
                    self.weight = _pack_int8_to_int4(self.weight)

        self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)
        self.bias = Parameter(bias_tensor.to(kwargs["device"]), requires_grad=False)

    def forward(self, input):
        output = torch._C.fused_linear_with_groupwise_quantized_weight(
            input, self.weight, self.weight_scale, w_zero=self.weight_zero,
            b=self.bias, num_bits=self.weight_bit_width, symmetric=self.symmetric,
            group_dim=-1, group_size=-1
        )
        return output


def quantize(model, weight_bit_width):
    """Replace fp16 linear with quantized linear"""
    assert(weight_bit_width in [4, 8]), "only support 4-bit or 8-bit quantization."
    for layer in model.layers:
        layer.attention.query_key_value = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.attention.query_key_value.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.attention.query_key_value.bias,
            in_features=layer.attention.query_key_value.in_features,
            out_features=layer.attention.query_key_value.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.attention.query_key_value.weight.device,
        )
        layer.attention.dense = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.attention.dense.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.attention.dense.bias,
            in_features=layer.attention.dense.in_features,
            out_features=layer.attention.dense.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.attention.dense.weight.device,
        )
        layer.mlp.dense_h_to_4h = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.mlp.dense_h_to_4h.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.mlp.dense_h_to_4h.bias,
            in_features=layer.mlp.dense_h_to_4h.in_features,
            out_features=layer.mlp.dense_h_to_4h.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.mlp.dense_h_to_4h.weight.device,
        )
        layer.mlp.dense_4h_to_h = QuantizedLinear(
            weight_bit_width=weight_bit_width,
            weight_tensor=layer.mlp.dense_4h_to_h.weight.to(torch.cuda.current_device()),
            bias_tensor=layer.mlp.dense_4h_to_h.bias,
            in_features=layer.mlp.dense_4h_to_h.in_features,
            out_features=layer.mlp.dense_4h_to_h.out_features,
            bias=True,
            dtype=torch.half,
            device=layer.mlp.dense_4h_to_h.weight.device,
        )
    return model
