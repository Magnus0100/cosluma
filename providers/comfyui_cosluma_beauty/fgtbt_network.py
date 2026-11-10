# Inference adapter for Xi0ngxinyu/FGTBT, revision 32627a248c66b5a5defec347fa197860347d080d.
# Copyright (c) 2025 熊新宇. FGTBT additions: MIT, see FGTBT-LICENSE.
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# SegFormer portions: NVIDIA Source Code License, see SegFormer-LICENSE.
# Adaptations: removed training/registry dependencies; identical inference layers.
from functools import partial
import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_

class LinearEmbedding(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Linear(channels, 768)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))

class Fusion(nn.Module):
    """The upstream mmcv ConvModule: bias-free convolution, BN and ReLU."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3072, 768, 1, bias=False)
        self.bn = nn.BatchNorm2d(768)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))

class Upsample(nn.Module):

    def __init__(self):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)

    def forward(self, x):
        return self.upsample(x)

class LandmarkHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_seg = nn.Conv2d(128, 124, 1)
        self.dropout = nn.Dropout2d(0.1)
        for stage, channels in enumerate((64, 128, 320, 512), 1):
            setattr(self, 'linear_c' + str(stage), LinearEmbedding(channels))
        self.linear_fuse = Fusion()
        self.linear_pred = nn.Conv2d(768, 256, 1)
        self.conv1 = ConvBlock(256, 128)
        self.upSample = Upsample()
        self.conv2 = nn.Conv2d(128, 124, 3, padding=1, bias=False)

    def forward(self, inputs):
        features = []
        for stage, value in enumerate(inputs, 1):
            embedded = getattr(self, 'linear_c' + str(stage))(value)
            embedded = embedded.permute(0, 2, 1).reshape(value.shape[0], -1, *value.shape[2:])
            features.append(F.interpolate(embedded, size=inputs[0].shape[2:], mode='bilinear', align_corners=False))
        x = self.linear_fuse(torch.cat(features[::-1], dim=1))
        return self.conv2(self.upSample(self.conv1(self.linear_pred(self.dropout(x)))))

class FGTBT(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = MixVisionTransformerEVP(img_size=480, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06), depths=[3, 6, 18, 3], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1, scale_factor=4, prompt_type='highpass', tuning_stage='1234', input_type='hpf', freq_nums=0.25, handcrafted_tune=True, embedding_tune=True, adaptor='adaptor', batch_size=2)
        self.encoder.decode_head = LandmarkHead()

    def forward(self, x):
        return self.encoder.decode_head(self.encoder.backbone(x))

def custom_max(x, dim, keepdim=True):
    temp_x = x
    for i in dim:
        temp_x = torch.max(temp_x, dim=i, keepdim=True)[0]
    if not keepdim:
        temp_x = temp_x.squeeze()
    return temp_x

class SemanticAttentionModule(nn.Module):

    def __init__(self, in_features, reduction_rate=16):
        super(SemanticAttentionModule, self).__init__()
        self.linear = []
        self.linear.append(nn.Linear(in_features=in_features, out_features=in_features // reduction_rate))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=in_features // reduction_rate, out_features=in_features))
        self.linear = nn.Sequential(*self.linear)

    def forward(self, x):
        max_x = custom_max(x, dim=(0, 2, 3), keepdim=False).unsqueeze(0)
        avg_x = torch.mean(x, dim=(0, 2, 3), keepdim=False).unsqueeze(0)
        max_x = self.linear(max_x)
        avg_x = self.linear(avg_x)
        att = max_x + avg_x
        att = torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
        return x * att

class CSAM(nn.Module):

    def __init__(self, num_slices, embed_dims, semantic=True, positional=True, slice=True, uncertainty=True, rank=5):
        super(CSAM, self).__init__()
        self.semantic = semantic
        self.positional = positional
        self.slice = slice
        if semantic:
            self.semantic_att = SemanticAttentionModule(embed_dims)

    def forward(self, x):
        if self.semantic:
            x = self.semantic_att(x)
        return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = (kv[0], kv[1])
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias)

class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        planes = int(out_planes / 2)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.BatchNorm2d(in_planes), nn.ReLU(True), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, embed_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x, False)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x, False)
        x = self.proj(x)
        x = self.bn3(x)
        x = F.relu(x, False)
        x = self.conv3(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return (x, H, W)

class MixVisionTransformerEVP(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        self.scale_factor = kwargs['scale_factor']
        self.prompt_type = kwargs['prompt_type']
        self.tuning_stage = str(kwargs['tuning_stage'])
        self.input_type = kwargs['input_type']
        self.freq_nums = kwargs['freq_nums']
        self.handcrafted_tune = kwargs['handcrafted_tune']
        self.embedding_tune = kwargs['embedding_tune']
        self.adaptor = kwargs['adaptor']
        self.batch_size = kwargs['batch_size']
        self.prompt_generator = PromptGenerator(self.scale_factor, self.prompt_type, self.embed_dims, self.tuning_stage, self.depths, self.input_type, self.freq_nums, self.handcrafted_tune, self.embedding_tune, self.adaptor, self.batch_size, img_size)

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        if self.handcrafted_tune:
            handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(x)
        else:
            handcrafted1, handcrafted2, handcrafted3, handcrafted4 = (None, None, None, None)
        x, H, W = self.patch_embed1(x)
        if '1' in self.tuning_stage:
            prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, 1)
        for i, blk in enumerate(self.block1):
            if '1' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt1, 1, i)
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed2(x)
        if '2' in self.tuning_stage:
            prompt2 = self.prompt_generator.init_prompt(x, handcrafted2, 2)
        for i, blk in enumerate(self.block2):
            if '2' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt2, 2, i)
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed3(x)
        if '3' in self.tuning_stage:
            prompt3 = self.prompt_generator.init_prompt(x, handcrafted3, 3)
        for i, blk in enumerate(self.block3):
            if '3' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt3, 3, i)
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed4(x)
        if '4' in self.tuning_stage:
            prompt4 = self.prompt_generator.init_prompt(x, handcrafted4, 4)
        for i, blk in enumerate(self.block4):
            if '4' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt4, 4, i)
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

class PromptGenerator(nn.Module):

    def __init__(self, scale_factor, prompt_type, embed_dims, tuning_stage, depths, input_type, freq_nums, handcrafted_tune, embedding_tune, adaptor, batch_size, img_size):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dims = embed_dims
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage
        self.depths = depths
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor
        self.batch_size = batch_size
        if self.handcrafted_tune:
            if '1' in self.tuning_stage:
                self.handcrafted_generator1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=self.embed_dims[0] // self.scale_factor)
            if '2' in self.tuning_stage:
                self.handcrafted_generator2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=self.embed_dims[0] // self.scale_factor, embed_dim=self.embed_dims[1] // self.scale_factor)
            if '3' in self.tuning_stage:
                self.handcrafted_generator3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=self.embed_dims[1] // self.scale_factor, embed_dim=self.embed_dims[2] // self.scale_factor)
            if '4' in self.tuning_stage:
                self.handcrafted_generator4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=self.embed_dims[2] // self.scale_factor, embed_dim=self.embed_dims[3] // self.scale_factor)
        if self.embedding_tune:
            if '1' in self.tuning_stage:
                self.embedding_generator1 = nn.Linear(self.embed_dims[0], self.embed_dims[0] // self.scale_factor)
            if '2' in self.tuning_stage:
                self.embedding_generator2 = nn.Linear(self.embed_dims[1], self.embed_dims[1] // self.scale_factor)
            if '3' in self.tuning_stage:
                self.embedding_generator3 = nn.Linear(self.embed_dims[2], self.embed_dims[2] // self.scale_factor)
            if '4' in self.tuning_stage:
                self.embedding_generator4 = nn.Linear(self.embed_dims[3], self.embed_dims[3] // self.scale_factor)
        if self.adaptor == 'adaptor':
            if '1' in self.tuning_stage:
                for i in range(self.depths[0]):
                    lightweight_mlp = nn.Sequential(nn.Linear(self.embed_dims[0] * (self.scale_factor + 2) // self.scale_factor, self.embed_dims[0] // self.scale_factor), nn.GELU())
                    setattr(self, 'lightweight_mlp1_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp1 = nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])
                self.bn1 = nn.BatchNorm2d(self.embed_dims[0] * (self.scale_factor + 2) // self.scale_factor)
                self.conv1 = nn.Conv2d(self.embed_dims[0] * (self.scale_factor + 2) // self.scale_factor, self.embed_dims[0] * (self.scale_factor + 2) // self.scale_factor, 1, 1, 0)
                self.shared_csam1 = CSAM(self.batch_size * 4, self.embed_dims[0] * (self.scale_factor + 2) // self.scale_factor)
            if '2' in self.tuning_stage:
                for i in range(self.depths[1]):
                    lightweight_mlp = nn.Sequential(nn.Linear(self.embed_dims[1] * (self.scale_factor + 2) // self.scale_factor, self.embed_dims[1] // self.scale_factor), nn.GELU())
                    setattr(self, 'lightweight_mlp2_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp2 = nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])
                self.bn2 = nn.BatchNorm2d(self.embed_dims[1] * (self.scale_factor + 2) // self.scale_factor)
                self.conv2 = nn.Conv2d(self.embed_dims[1] * (self.scale_factor + 2) // self.scale_factor, self.embed_dims[1] * (self.scale_factor + 2) // self.scale_factor, 1, 1, 0)
                self.shared_csam2 = CSAM(self.batch_size * 4, self.embed_dims[1] * (self.scale_factor + 2) // self.scale_factor)
            if '3' in self.tuning_stage:
                for i in range(self.depths[2]):
                    lightweight_mlp = nn.Sequential(nn.Linear(self.embed_dims[2] * (self.scale_factor + 2) // self.scale_factor, self.embed_dims[2] // self.scale_factor), nn.GELU())
                    setattr(self, 'lightweight_mlp3_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp3 = nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])
                self.bn3 = nn.BatchNorm2d(self.embed_dims[2] * (self.scale_factor + 2) // self.scale_factor)
                self.conv3 = nn.Conv2d(self.embed_dims[2] * (self.scale_factor + 2) // self.scale_factor, self.embed_dims[2] * (self.scale_factor + 2) // self.scale_factor, 1, 1, 0)
                self.shared_csam3 = CSAM(self.batch_size * 4, self.embed_dims[2] * (self.scale_factor + 2) // self.scale_factor)
            if '4' in self.tuning_stage:
                for i in range(self.depths[3]):
                    lightweight_mlp = nn.Sequential(nn.Linear(self.embed_dims[3] * (self.scale_factor + 2) // self.scale_factor, self.embed_dims[3] // self.scale_factor), nn.GELU())
                    setattr(self, 'lightweight_mlp4_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp4 = nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])
                self.bn4 = nn.BatchNorm2d(self.embed_dims[3] * (self.scale_factor + 2) // self.scale_factor)
                self.conv4 = nn.Conv2d(self.embed_dims[3] * (self.scale_factor + 2) // self.scale_factor, self.embed_dims[3] * (self.scale_factor + 2) // self.scale_factor, 1, 1, 0)
                self.shared_csam4 = CSAM(self.batch_size * 4, self.embed_dims[3] * (self.scale_factor + 2) // self.scale_factor)

    def init_handcrafted(self, x):
        if self.input_type == 'hpf':
            x = self.hpf(x, 20, self.prompt_type)
        B = x.shape[0]
        if '1' in self.tuning_stage:
            handcrafted1, H1, W1 = self.handcrafted_generator1(x)
        else:
            handcrafted1 = None
        if '2' in self.tuning_stage:
            handcrafted2, H2, W2 = self.handcrafted_generator2(handcrafted1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous())
        else:
            handcrafted2 = None
        if '3' in self.tuning_stage:
            handcrafted3, H3, W3 = self.handcrafted_generator3(handcrafted2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous())
        else:
            handcrafted3 = None
        if '4' in self.tuning_stage:
            handcrafted4, H4, W4 = self.handcrafted_generator4(handcrafted3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous())
        else:
            handcrafted4 = None
        return (handcrafted1, handcrafted2, handcrafted3, handcrafted4)

    def init_prompt(self, embedding_feature, handcrafted_feature, block_num):
        if self.embedding_tune:
            embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))
            embedding_feature = embedding_generator(embedding_feature)
        if self.handcrafted_tune:
            handcrafted_feature = handcrafted_feature
        return (handcrafted_feature, embedding_feature)

    def get_prompt(self, x, prompt, block_num, depth_num):
        feat = torch.cat([x, prompt[0], prompt[1]], dim=2)
        B, A_2, C = feat.size()
        A = int(math.sqrt(A_2))
        feat = feat.view(B, C, A, A)
        res = feat
        csam = getattr(self, 'shared_csam' + str(block_num))
        feat = csam(feat)
        bn = getattr(self, 'bn' + str(block_num))
        feat = bn(feat)
        feat = F.relu(feat, True)
        conv11 = getattr(self, 'conv' + str(block_num))
        feat = conv11(feat) + res
        feat = feat.view(B, A_2, -1)
        if self.adaptor == 'adaptor':
            lightweight_mlp = getattr(self, 'lightweight_mlp' + str(block_num) + '_' + str(depth_num))
            feat = lightweight_mlp(feat)
            shared_mlp = getattr(self, 'shared_mlp' + str(block_num))
            feat = shared_mlp(feat)
        x = x + feat
        return x

    def hpf(self, x, sigma, prompt_type):
        device = x.device
        b, c, h, w = x.shape
        y = torch.arange(h, device=device) - h // 2
        x_coords = torch.arange(w, device=device) - w // 2
        y_grid, x_grid = torch.meshgrid(y, x_coords, indexing='ij')
        D_squared = x_grid ** 2 + y_grid ** 2
        lowpass_mask = torch.exp(-D_squared / (2 * sigma ** 2))
        highpass_mask = 1 - lowpass_mask
        if prompt_type == 'highpass':
            mask = highpass_mask
        elif prompt_type == 'lowpass':
            mask = lowpass_mask
        else:
            raise ValueError("prompt_type 必须为 'highpass' 或 'lowpass'")
        mask = mask.unsqueeze(0).unsqueeze(0)
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'), dim=(-2, -1))
        fft_filtered = fft * mask
        fft_filtered = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        inv = torch.fft.ifft2(fft_filtered, norm='forward').real
        inv = torch.abs(inv)
        return inv

class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
