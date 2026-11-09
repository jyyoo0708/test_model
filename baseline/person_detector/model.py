"""Tiny center-point detector. Output channels: heatmap, offset xy, size wh."""
import copy
import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, a, b, stride=1):
        super().__init__(nn.Conv2d(a, b, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class PersonDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(1, 8, 2)
        self.c2 = ConvBlock(8, 16, 2)
        self.c3 = ConvBlock(16, 24, 2)
        self.c4 = ConvBlock(24, 32, 2)
        self.context = ConvBlock(32, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse1 = ConvBlock(32 + 24, 24)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse2 = ConvBlock(24 + 16, 16)
        self.head = ConvBlock(16, 16)
        self.heatmap = nn.Conv2d(16, 1, 1)
        self.offset = nn.Conv2d(16, 2, 1)
        self.size = nn.Conv2d(16, 2, 1)
        nn.init.constant_(self.heatmap.bias, -2.19)
        with torch.no_grad():
            self.size.bias.copy_(torch.logit(torch.tensor([.18, .45])))

    def forward(self, x):
        a = self.c1(x)
        b = self.c2(a)
        c = self.c3(b)
        d = self.context(self.c4(c))
        e = self.fuse1(torch.cat([self.up1(d), c], dim=1))
        f = self.head(self.fuse2(torch.cat([self.up2(e), b], dim=1)))
        return torch.cat([self.heatmap(f), self.offset(f), self.size(f)], dim=1)

    def fused(self):
        model = copy.deepcopy(self).eval()
        for module in model.modules():
            if isinstance(module, ConvBlock):
                module[0] = torch.nn.utils.fusion.fuse_conv_bn_eval(module[0], module[1])
                module[1] = nn.Identity()
        return model


def memory_profile(model):
    layers, hooks = [], []
    def capture(name):
        def hook(module, inputs, output):
            row = {"name": name, "type": type(module).__name__, "shape": list(output.shape),
                   "output_bytes_int8": output.numel(), "weight_count": 0, "bias_count": 0, "macs": 0}
            if isinstance(module, nn.Conv2d):
                row.update(weight_count=module.weight.numel(), bias_count=module.bias.numel(),
                           macs=output.numel()*module.in_channels*module.kernel_size[0]*module.kernel_size[1])
            layers.append(row)
        return hook
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Upsample)):
            hooks.append(module.register_forward_hook(capture(name)))
    with torch.inference_mode():
        model(torch.zeros(1, 1, 96, 96))
    for hook in hooks:
        hook.remove()
    # Concats are functional calls and therefore not covered by module hooks.
    concat = (32+24)*12*12 + (24+16)*24*24 + 5*24*24
    weights = sum(r["weight_count"] for r in layers)
    biases = sum(r["bias_count"] for r in layers)
    parts = {"weights_int8": weights, "bias_int32": biases*4,
             "input_all_outputs_upsampling_concat_int8": 96*96 + sum(r["output_bytes_int8"] for r in layers) + concat,
             "largest_conv_int32_accumulator": max(r["output_bytes_int8"] for r in layers if r["type"] == "Conv2d")*4,
             "scales_float32": (biases+len(layers)+1)*4,
             "postprocess_and_other_workspace_reserved": 16384}
    return {"layers": layers, "parts": parts, "total_bytes": sum(parts.values()), "limit_bytes": 300000,
            "within_limit": sum(parts.values()) <= 300000, "parameters": weights+biases,
            "macs": sum(r["macs"] for r in layers),
            "assumptions": "Future batch1 direct int8 convolution, int32 bias/accumulator, all outputs/concat/upsamples retained; 16KiB postprocess reserve. Not an integer implementation."}
