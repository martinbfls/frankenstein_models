"""Model families.

Blocks are the unit the chimera generator copies from. Two residual block types:
  - BasicBlock   (family "resnet", 3x3 convs, BN)          -> channel spaces: in, out
  - ConvNeXtBlock(family "convnext", dwconv 7x7, LN, exp4) -> channel spaces: dim, hidden

Networks keep their residual blocks in a flat ``self.blocks`` ModuleList with a
parallel ``self.block_meta`` list, plus ``self.stem`` / downsample layers inside each
block wrapper, and ``self.head``. This flat layout is what chimera.py walks to build
a donor registry and to reproduce an architecture exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- family A
class BasicBlock(nn.Module):
    family = "resnet"

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))
        else:
            self.downsample = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        sc = x if self.downsample is None else self.downsample(x)
        return F.relu(out + sc)


# ----------------------------------------------------------------------------- family B
class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm over the channel dimension."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class ConvNeXtBlock(nn.Module):
    family = "convnext"

    def __init__(self, dim, out_ch=None, stride=1, layer_scale=1e-6):
        super().__init__()
        # stride/out_ch handled by an optional down projection so the block type is
        # drop-in within the flat block list (matches BasicBlock's (in,out,stride) role).
        self.in_ch = dim
        self.out_ch = out_ch or dim
        self.stride = stride
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        if stride != 1 or self.out_ch != dim:
            self.proj = nn.Sequential(LayerNorm2d(dim),
                                      nn.Conv2d(dim, self.out_ch, 2, stride) if stride != 1
                                      else nn.Conv2d(dim, self.out_ch, 1))
        else:
            self.proj = None

    def forward(self, x):
        h = self.dwconv(x)
        h = self.norm(h)
        h = h.permute(0, 2, 3, 1)
        h = self.pwconv2(F.gelu(self.pwconv1(h)))
        h = self.gamma * h
        h = h.permute(0, 3, 1, 2)
        x = x + h
        if self.proj is not None:
            x = self.proj(x)
        return x


# ----------------------------------------------------------------------------- networks
class Net(nn.Module):
    """Generic assembled network: stem -> blocks -> head. Used by all families and by
    the chimera generator (which fills weights afterwards)."""

    def __init__(self, stem, blocks, block_meta, head, family):
        super().__init__()
        self.stem = stem
        self.blocks = nn.ModuleList(blocks)
        self.block_meta = block_meta           # list of dicts: family, in_ch, out_ch, stride
        self.head = head
        self.family = family

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class Head(nn.Module):
    def __init__(self, dim, num_classes, norm=False):
        super().__init__()
        self.norm = LayerNorm2d(dim) if norm else None
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ----------------------------------------------------------------------------- builders
def build_resnet(widths, blocks, num_classes=10):
    stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                         nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
    blk, meta = [], []
    in_ch = widths[0]
    for s, (w, nb) in enumerate(zip(widths, blocks)):
        for j in range(nb):
            stride = 2 if (s > 0 and j == 0) else 1
            blk.append(BasicBlock(in_ch, w, stride))
            meta.append({"family": "resnet", "in_ch": in_ch, "out_ch": w, "stride": stride})
            in_ch = w
    head = Head(widths[-1], num_classes, norm=False)
    return Net(stem, blk, meta, head, "resnet")


def build_convnext(dims, depths, num_classes=10):
    stem = nn.Sequential(nn.Conv2d(3, dims[0], 3, 1, 1),
                         LayerNorm2d(dims[0]))
    blk, meta = [], []
    for s, (d, nb) in enumerate(zip(dims, depths)):
        for j in range(nb):
            # last block of a non-final stage carries the stride-2 down projection
            down = (s < len(dims) - 1 and j == nb - 1)
            out_ch = dims[s + 1] if down else d
            stride = 2 if down else 1
            blk.append(ConvNeXtBlock(d, out_ch=out_ch, stride=stride))
            meta.append({"family": "convnext", "in_ch": d, "out_ch": out_ch, "stride": stride})
    head = Head(dims[-1], num_classes, norm=True)
    return Net(stem, blk, meta, head, "convnext")


class MaxPool2d(nn.MaxPool2d):
    """nn.MaxPool2d whose backward is itself differentiable. Recent torch refuses a second
    derivative through max_pool2d unless the indices are requested; Witches' Brew and FLIP
    both differentiate through the gradient. No parameters: checkpoints are unaffected."""

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                            ceil_mode=self.ceil_mode, return_indices=True)[0]


def build_vgg(cfg, num_classes=10):
    layers, in_ch = [], 3
    for v in cfg:
        if v == "M":
            layers.append(MaxPool2d(2, 2))
        else:
            layers += [nn.Conv2d(in_ch, v, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_ch = v
    stem = nn.Sequential(*layers)
    last = [v for v in cfg if v != "M"][-1]
    head = Head(last, num_classes, norm=False)
    return Net(stem, [], [], head, "vgg")


def build_from_arch(arch, num_classes=10):
    fam = arch["family"]
    if fam == "resnet":
        return build_resnet(arch["widths"], arch["blocks"], num_classes)
    if fam == "convnext":
        return build_convnext(arch["dims"], arch["depths"], num_classes)
    if fam == "vgg":
        return build_vgg(arch["cfg"], num_classes)
    raise ValueError(fam)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
