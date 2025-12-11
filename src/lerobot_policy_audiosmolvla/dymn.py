
import math
import os
import urllib.parse
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.ops.misc import ConvNormActivation

### code from EfficientAT
# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------

def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


# ------------------------------------------------------------------------------
# Config & Blocks (from dy_block.py)
# ------------------------------------------------------------------------------

class DynamicInvertedResidualConfig:
    def __init__(
            self,
            input_channels: int,
            kernel: int,
            expanded_channels: int,
            out_channels: int,
            use_dy_block: bool,
            activation: str,
            stride: int,
            dilation: int,
            width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_dy_block = use_dy_block
        self.use_hs = activation == "HS"
        self.use_se = False
        self.stride = stride
        self.dilation = dilation
        self.width_mult = width_mult

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size):
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class DynamicConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 context_dim,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 padding=0,
                 groups=1,
                 att_groups=1,
                 bias=False,
                 k=4,
                 temp_schedule=(30, 1, 1, 0.05)
                 ):
        super(DynamicConv, self).__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.k = k
        self.T_max, self.T_min, self.T0_slope, self.T1_slope = temp_schedule
        self.temperature = self.T_max
        self.att_groups = att_groups

        self.residuals = nn.Sequential(
                nn.Linear(context_dim, k * self.att_groups)
        )

        weight = torch.randn(k, out_channels, in_channels // groups, kernel_size, kernel_size)

        if bias:
            self.bias = nn.Parameter(torch.zeros(k, out_channels), requires_grad=True)
        else:
            self.bias = None

        self._initialize_weights(weight, self.bias)

        weight = weight.view(1, k, att_groups, out_channels,
                             in_channels // groups, kernel_size, kernel_size)

        weight = weight.transpose(1, 2).view(1, self.att_groups, self.k, -1)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def _initialize_weights(self, weight, bias):
        init_func = partial(nn.init.kaiming_normal_, mode="fan_out")
        for i in range(self.k):
            init_func(weight[i])
            if bias is not None:
                nn.init.zeros_(bias[i])

    def forward(self, x, g=None):
        b, c, f, t = x.size()
        g_c = g[0].view(b, -1)
        residuals = self.residuals(g_c).view(b, self.att_groups, 1, -1)
        attention = F.softmax(residuals / self.temperature, dim=-1)

        aggregate_weight = (attention @ self.weight).transpose(1, 2).reshape(b, self.out_channels,
                                                                             self.in_channels // self.groups,
                                                                             self.kernel_size, self.kernel_size)

        aggregate_weight = aggregate_weight.view(b * self.out_channels, self.in_channels // self.groups,
                                                 self.kernel_size, self.kernel_size)
        x = x.view(1, -1, f, t)
        if self.bias is not None:
            aggregate_bias = torch.mm(attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, self.out_channels, output.size(-2), output.size(-1))
        return output

    def update_params(self, epoch):
        t0 = self.T_max - self.T0_slope * epoch
        t1 = 1 + self.T1_slope * (self.T_max - 1) / self.T0_slope - self.T1_slope * epoch
        self.temperature = max(t0, t1, self.T_min)


class DyReLU(nn.Module):
    def __init__(self, channels, context_dim, M=2):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.M = M

        self.coef_net = nn.Sequential(
                nn.Linear(context_dim, 2 * M)
        )

        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.] * M + [0.5] * M).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * M - 1)).float())

    def get_relu_coefs(self, x):
        theta = self.coef_net(x)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x, g):
        raise NotImplementedError


class DyReLUB(DyReLU):
    def __init__(self, channels, context_dim, M=2):
        super(DyReLUB, self).__init__(channels, context_dim, M)
        self.coef_net[-1] = nn.Linear(context_dim, 2 * M * self.channels)

    def forward(self, x, g):
        assert x.shape[1] == self.channels
        assert g is not None
        b, c, f, t = x.size()
        h_c = g[0].view(b, -1)
        theta = self.get_relu_coefs(h_c)

        relu_coefs = theta.view(-1, self.channels, 1, 1, 2 * self.M) * self.lambdas + self.init_v
        x_mapped = x.unsqueeze(-1) * relu_coefs[:, :, :, :, :self.M] + relu_coefs[:, :, :, :, self.M:]
        if self.M == 2:
            result = torch.maximum(x_mapped[:, :, :, :, 0], x_mapped[:, :, :, :, 1])
        else:
            result = torch.max(x_mapped, dim=-1)[0]
        return result


class CoordAtt(nn.Module):
    def __init__(self):
        super(CoordAtt, self).__init__()

    def forward(self, x, g):
        g_cf, g_ct = g[1], g[2]
        a_f = g_cf.sigmoid()
        a_t = g_ct.sigmoid()
        out = x * a_f * a_t
        return out


class DynamicWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, g):
        return self.module(x)


class ContextGen(nn.Module):
    def __init__(self, context_dim, in_ch, exp_ch, norm_layer, stride: int = 1):
        super(ContextGen, self).__init__()

        self.joint_conv = nn.Conv2d(in_ch, context_dim, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.joint_norm = norm_layer(context_dim)
        self.joint_act = nn.Hardswish(inplace=True)

        self.conv_f = nn.Conv2d(context_dim, exp_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv_t = nn.Conv2d(context_dim, exp_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)

        if stride > 1:
            self.pool_f = nn.AvgPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0))
            self.pool_t = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, stride), padding=(0, 1))
        else:
            self.pool_f = nn.Sequential()
            self.pool_t = nn.Sequential()

    def forward(self, x, g):
        cf = F.adaptive_avg_pool2d(x, (None, 1))
        ct = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
        f, t = cf.size(2), ct.size(2)

        g_cat = torch.cat([cf, ct], dim=2)
        g_cat = self.joint_norm(self.joint_conv(g_cat))
        g_cat = self.joint_act(g_cat)

        h_cf, h_ct = torch.split(g_cat, [f, t], dim=2)
        h_ct = h_ct.permute(0, 1, 3, 2)
        h_c = torch.mean(g_cat, dim=2, keepdim=True)
        g_cf, g_ct = self.conv_f(self.pool_f(h_cf)), self.conv_t(self.pool_t(h_ct))

        g = (h_c, g_cf, g_ct)
        return g


class DY_Block(nn.Module):
    def __init__(
            self,
            cnf: DynamicInvertedResidualConfig,
            context_ratio: int = 4,
            max_context_size: int = 128,
            min_context_size: int = 32,
            temp_schedule: tuple = (30, 1, 1, 0.05),
            dyrelu_k: int = 2,
            dyconv_k: int = 4,
            no_dyrelu: bool = False,
            no_dyconv: bool = False,
            no_ca: bool = False,
            **kwargs: Any
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        self.context_dim = np.clip(make_divisible(cnf.expanded_channels // context_ratio, 8),
                                   make_divisible(min_context_size * cnf.width_mult, 8),
                                   make_divisible(max_context_size * cnf.width_mult, 8)
                                   )

        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        if cnf.expanded_channels != cnf.input_channels:
            if no_dyconv:
                self.exp_conv = DynamicWrapper(
                    nn.Conv2d(cnf.input_channels, cnf.expanded_channels, kernel_size=1, bias=False)
                )
            else:
                self.exp_conv = DynamicConv(
                    cnf.input_channels, cnf.expanded_channels, self.context_dim, kernel_size=1,
                    k=dyconv_k, temp_schedule=temp_schedule, bias=False
                )

            self.exp_norm = norm_layer(cnf.expanded_channels)
            self.exp_act = DynamicWrapper(activation_layer(inplace=True))
        else:
            self.exp_conv = DynamicWrapper(nn.Identity())
            self.exp_norm = nn.Identity()
            self.exp_act = DynamicWrapper(nn.Identity())

        stride = 1 if cnf.dilation > 1 else cnf.stride
        padding = (cnf.kernel - 1) // 2 * cnf.dilation
        if no_dyconv:
            self.depth_conv = DynamicWrapper(
                nn.Conv2d(
                    cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                    groups=cnf.expanded_channels, stride=stride, dilation=cnf.dilation, padding=padding, bias=False
                )
            )
        else:
            self.depth_conv = DynamicConv(
                cnf.expanded_channels, cnf.expanded_channels, self.context_dim, kernel_size=cnf.kernel,
                k=dyconv_k, temp_schedule=temp_schedule, groups=cnf.expanded_channels, stride=stride,
                dilation=cnf.dilation, padding=padding, bias=False
            )
        self.depth_norm = norm_layer(cnf.expanded_channels)
        self.depth_act = DynamicWrapper(activation_layer(inplace=True)) if no_dyrelu \
            else DyReLUB(cnf.expanded_channels, self.context_dim, M=dyrelu_k)

        self.ca = DynamicWrapper(nn.Identity()) if no_ca else CoordAtt()

        if no_dyconv:
            self.proj_conv = DynamicWrapper(
                nn.Conv2d(cnf.expanded_channels, cnf.out_channels, kernel_size=1, bias=False)
            )
        else:
            self.proj_conv = DynamicConv(
                cnf.expanded_channels, cnf.out_channels, self.context_dim, kernel_size=1,
                k=dyconv_k, temp_schedule=temp_schedule, bias=False,
            )

        self.proj_norm = norm_layer(cnf.out_channels)

        self.context_gen = ContextGen(self.context_dim, cnf.input_channels, cnf.expanded_channels,
                                      norm_layer=norm_layer, stride=stride)

    def forward(self, x, g=None):
        inp = x
        g = self.context_gen(x, g)
        x = self.exp_conv(x, g)
        x = self.exp_norm(x)
        x = self.exp_act(x, g)
        x = self.depth_conv(x, g)
        x = self.depth_norm(x)
        x = self.depth_act(x, g)
        x = self.ca(x, g)
        x = self.proj_conv(x, g)
        x = self.proj_norm(x)
        if self.use_res_connect:
            x += inp
        return x


# ------------------------------------------------------------------------------
# DyMN Model (Core)
# ------------------------------------------------------------------------------

model_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
model_dir = "resources"
pretrained_models = {
    "dymn10_as": urllib.parse.urljoin(model_url, "dymn10_as.pt"),
}


class DyMN(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[DynamicInvertedResidualConfig],
            last_channel: int,
            num_classes: int = 527,
            head_type: str = "mlp",
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.2,
            in_conv_kernel: int = 3,
            in_conv_stride: int = 2,
            in_channels: int = 1,
            context_ratio: int = 4,
            max_context_size: int = 128,
            min_context_size: int = 32,
            dyrelu_k=2,
            dyconv_k=4,
            no_dyrelu: bool = False,
            no_dyconv: bool = False,
            no_ca: bool = False,
            temp_schedule: tuple = (30, 1, 1, 0.05),
            **kwargs: Any,
    ) -> None:
        super(DyMN, self).__init__()

        if block is None:
            block = DY_Block

        norm_layer = norm_layer if norm_layer is not None else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        self.layers = nn.ModuleList()

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.in_c = ConvNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=in_conv_kernel,
                stride=in_conv_stride,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
        )

        for cnf in inverted_residual_setting:
            if cnf.use_dy_block:
                b = block(cnf,
                          context_ratio=context_ratio,
                          max_context_size=max_context_size,
                          min_context_size=min_context_size,
                          dyrelu_k=dyrelu_k,
                          dyconv_k=dyconv_k,
                          no_dyrelu=no_dyrelu,
                          no_dyconv=no_dyconv,
                          no_ca=no_ca,
                          temp_schedule=temp_schedule
                          )
            else:
                 # We only support dy blocks for this condensed file
                 raise NotImplementedError("Only DY_Block supported in this version")
            self.layers.append(b)

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.out_c = ConvNormActivation(
            lastconv_input_channels,
            lastconv_output_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Hardswish,
        )

        self.head_type = head_type
        if self.head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(lastconv_output_channels, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(last_channel, num_classes),
            )
        else:
            raise NotImplementedError(f"Head '{self.head_type}' unknown")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _feature_forward(
        self, x: Tensor, return_fmaps: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        fmaps = []
        x = self.in_c(x)
        if return_fmaps:
            fmaps.append(x)

        for layer in self.layers:
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)

        x = self.out_c(x)
        if return_fmaps:
            fmaps.append(x)
            return x, fmaps
        return x

    def _clf_forward(self, x: Tensor):
        embed = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x).squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x, embed

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._feature_forward(x)
        x, embed = self._clf_forward(x)
        return x, embed

    def update_params(self, epoch):
        for module in self.modules():
            if isinstance(module, DynamicConv):
                module.update_params(epoch)


# ------------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------------

class AugmentMelSTFT(nn.Module):
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 fmin=0.0, fmax=None, fmin_aug_range=10, fmax_aug_range=2000):
        super().__init__()
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer('window', torch.hann_window(win_length, periodic=False), persistent=False)
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
        x = (x ** 2).sum(dim=-1)  # power mag
        
        fmin = self.fmin
        fmax = self.fmax
        if self.training:
             fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
             fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels,  self.n_fft, self.sr,
                                        fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        
        melspec = torch.matmul(mel_basis, x)
        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization
        return melspec


# ------------------------------------------------------------------------------
# Integrated Model Class
# ------------------------------------------------------------------------------

class DyMNMedium(nn.Module):
    def __init__(self, pretrained=True, device='cpu'):
        super().__init__()
        self.device = device
        self.pretrained = pretrained
        # Config for DyMN Medium (width_mult=1.0)
        width_mult = 1.0
        use_dy_blocks = "all"
        pretrained_name = "dymn10_as" if pretrained else None
        
        # Build Config
        bneck_conf = partial(DynamicInvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(DynamicInvertedResidualConfig.adjust_channels, width_mult=width_mult)
        
        # Activations: RE=ReLU, HS=HardSwish
        # Medium/Standard config usually has 15 blocks
        activations = ["RE", "RE", "RE", "RE", "RE", "RE", "HS", "HS", "HS", "HS", "HS", "HS", "HS", "HS", "HS"]
        use_dy_block = [True] * 15 
        
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, use_dy_block[0], activations[0], 1, 1),
            bneck_conf(16, 3, 64, 24, use_dy_block[1], activations[1], 2, 1),
            bneck_conf(24, 3, 72, 24, use_dy_block[2], activations[2], 1, 1),
            bneck_conf(24, 5, 72, 40, use_dy_block[3], activations[3], 2, 1),
            bneck_conf(40, 5, 120, 40, use_dy_block[4], activations[4], 1, 1),
            bneck_conf(40, 5, 120, 40, use_dy_block[5], activations[5], 1, 1),
            bneck_conf(40, 3, 240, 80, use_dy_block[6], activations[6], 2, 1),
            bneck_conf(80, 3, 200, 80, use_dy_block[7], activations[7], 1, 1),
            bneck_conf(80, 3, 184, 80, use_dy_block[8], activations[8], 1, 1),
            bneck_conf(80, 3, 184, 80, use_dy_block[9], activations[9], 1, 1),
            bneck_conf(80, 3, 480, 112, use_dy_block[10], activations[10], 1, 1),
            bneck_conf(112, 3, 672, 112, use_dy_block[11], activations[11], 1, 1),
            bneck_conf(112, 5, 672, 160, use_dy_block[12], activations[12], 2, 1),
            bneck_conf(160, 5, 960, 160, use_dy_block[13], activations[13], 1, 1),
            bneck_conf(160, 5, 960, 160, use_dy_block[14], activations[14], 1, 1),
        ]
        last_channel = adjust_channels(1280)

        # Instantiate Model
        self.model = DyMN(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            num_classes=527,
            temp_schedule=(1.0, 1, 1, 0.05) # Pretrained final temp
        )

        if pretrained_name:
            url = pretrained_models.get(pretrained_name)
            if url:
                state_dict = load_state_dict_from_url(url, model_dir=model_dir, map_location="cpu")
                # Handle classification head mismatch if any (though we use default 527 here)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights for {pretrained_name}")
        
        self.to(device)
        
        if self.pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    
    def forward(self, spec: Tensor):
        """
        Args:
            spectrogram: (Batch, 1, n_mels, time)
        Returns:
             embeddings: (Batch, EmbedDim)
        """
        embeddings = self.model._feature_forward(spec)
        
        return embeddings

