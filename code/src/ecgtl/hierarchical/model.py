"""Backbone bersama, perhatian per kelas, fusi, dan auxiliary loss."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models


def backbone(arch, pretrained):
    weights = {'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
               'mobilenet_v2': models.MobileNet_V2_Weights.IMAGENET1K_V1,
               'shufflenet_v2_x1_0': models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1}
    net = getattr(models, arch)(weights=weights[arch] if pretrained else None)
    dim = 1024 if arch.startswith('shuffle') else 1280
    if hasattr(net, 'classifier'):
        net.classifier = nn.Identity()
    else:
        net.fc = nn.Identity()
    name, conv = next((n, m) for n, m in net.named_modules() if isinstance(m, nn.Conv2d))
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    bias=conv.bias is not None)
    with torch.no_grad():
        if pretrained:
            new.weight.copy_(conv.weight.sum(1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    owner = net
    parts = name.split('.')
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], new)
    return net, dim


class HierarchicalNet(nn.Module):
    def __init__(self, arch, scenario, cfg, pretrained=True, encoder=None, feature_dim=None):
        super().__init__()
        self.encoder, dim = backbone(arch, pretrained and scenario != 'S1') if encoder is None else (encoder, feature_dim)
        self.arch, self.scenario, self.dim = arch, scenario, dim
        self.frozen = scenario in ['S2', 'S3']
        self.chunk = cfg['model']['feature_chunk']
        self.checkpointing = cfg['training']['gradient_checkpointing']
        self.dropout = nn.Dropout(cfg['model']['dropout'])
        self.global_head = nn.Linear(dim, 3)
        self.beat_head = nn.Linear(dim, 3)
        self.attention = nn.Linear(dim, 3)
        self.register_buffer('global_mean', torch.zeros(dim))
        self.register_buffer('global_std', torch.ones(dim))
        self.register_buffer('local_mean', torch.zeros(dim))
        self.register_buffer('local_std', torch.ones(dim))
        for param in self.encoder.parameters():
            param.requires_grad = not self.frozen

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def encode(self, images):
        parts = []
        for part in images.split(self.chunk):
            if self.training and not self.frozen and self.checkpointing:
                # Running BN statistics must update only on the initial forward.
                parts.append(checkpoint(self.encoder, part, use_reentrant=False,
                                        context_fn=lambda: (torch.enable_grad(), _bn_recompute(self.encoder))))
            else:
                parts.append(self.encoder(part))
        return torch.cat(parts)

    def forward(self, global_x, local_x, mask):
        if not mask.any(1).all():
            raise ValueError('Setiap window harus memiliki setidaknya satu crop.')
        if global_x.ndim == 2:
            g, local = global_x, local_x
        else:
            g = self.encode(global_x)
            flat = self.encode(local_x[mask])
            local = flat.new_zeros((*mask.shape, self.dim))
            local[mask] = flat
        g = (g - self.global_mean) / self.global_std
        local = (local - self.local_mean) / self.local_std
        global_logits = self.global_head(self.dropout(g))
        beat_logits = self.beat_head(self.dropout(local))
        attention = self.attention(local).masked_fill(~mask[..., None], -torch.inf).softmax(1)
        local_logits = (attention * beat_logits).sum(1)
        pg, pl = global_logits.log_softmax(-1), local_logits.log_softmax(-1)
        logp = torch.cat([pg[:, :1] + pl, pg[:, 1:]], -1)
        return {'logp': logp, 'global_logits': global_logits, 'beat_logits': beat_logits,
                'attention': attention, 'local_logits': local_logits}


class _bn_recompute:
    def __init__(self, module):
        self.layers = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def __enter__(self):
        self.values = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in self.layers]

    def __exit__(self, *args):
        with torch.no_grad():
            for m, (mean, var, count) in zip(self.layers, self.values):
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)
                m.num_batches_tracked.copy_(count)


def inverse_weights(counts, tau):
    counts = np.asarray(counts, dtype=float)
    if tau < 1 or np.any(counts <= 0):
        raise ValueError('Bobot membutuhkan tau >= 1 dan semua kelas pada subtrain.')
    ratio = counts.max() / counts.min()
    gamma = min(1., math.log(tau) / math.log(ratio)) if ratio > 1 else 0.
    weights = (1 / counts) ** gamma
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def loss_terms(output, batch, weights, cfg):
    y = batch['y']
    global_y = torch.where(y < 3, 0, y - 2)
    final = F.nll_loss(output['logp'], y, weight=weights['final'], reduction='none').mean()
    global_loss = F.cross_entropy(output['global_logits'], global_y, weight=weights['global'], reduction='none').mean()
    valid = batch['mask'] & (batch['beat'] >= 0) & (y[:, None] < 3)
    beat_loss = F.cross_entropy(output['beat_logits'].transpose(1, 2), batch['beat'],
                               weight=weights['local'], ignore_index=-1, reduction='none')
    per_window = (beat_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
    local = per_window[valid.any(1)].mean() if valid.any() else beat_loss.sum() * 0
    total = final + cfg['lambda_global'] * global_loss + cfg['lambda_local'] * local
    return total, {'final': final.detach(), 'global': global_loss.detach(), 'local': local.detach()}
