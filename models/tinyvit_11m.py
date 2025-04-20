# models/tinyvit_11m.py
"""
Implementation of the TinyViT-11M model variant.

NOTE: This implementation defines its own set of building blocks (LayerNorm,
MBConv with SE, Conv-based Attention, etc.) which differ from those in
`blocks.py` used by the `tinyvit_5m` implementation in `tinyvit.py`.
This might reflect specific choices for the 11M variant in the source this
code was based on, or differences from the original paper's details.
It uses MBConv blocks (with 7x7 depthwise) in early stages and
Conv-based Transformer blocks in later stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

class LayerNorm(nn.Module):
    """ Custom Layer Normalization specific to this TinyViT-11M implementation.
        Operates on channels_last format (B, C, H, W).
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # Expecting input shape B, C, H, W
        mean = x.mean(1, keepdim=True) # Mean over channel dim
        var = x.var(1, keepdim=True, unbiased=False) # Var over channel dim
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        # Apply affine transformationreshaping weight/bias to (1, C, 1, 1)
        return x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

class ConvMlp(nn.Module):
    """ MLP block implemented with 1x1 Convolutions. """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Using Conv2d for MLP allows operating directly on (B, C, H, W) format
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MBConv(nn.Module):
    """ MobileNetV2-style Inverted Bottleneck Block with 7x7 Depthwise Conv and SE layer. """
    def __init__(self, in_dim, out_dim, stride=1, expand_ratio=4, drop_path=0.):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_dim * expand_ratio) # Renamed from exp_dim for clarity
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        layers = []
        # Expansion phase (1x1 Conv)
        layers.append(nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(LayerNorm(hidden_dim)) # Use custom LayerNorm
        layers.append(nn.GELU()) # Use GELU activation consistent with Transformers
        self.expand = nn.Sequential(*layers)

        # Depthwise convolution (7x7)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, stride=stride,
                                padding=3, groups=hidden_dim, bias=False) # padding=3 for 7x7
        self.norm_dw = LayerNorm(hidden_dim)
        self.act_dw = nn.GELU()

        # Squeeze-and-Excitation layer
        # Reduced channels based on hidden_dim, not fixed 4 (more flexible)
        se_ratio = 0.25 # Standard ratio
        se_reduced_dim = max(1, int(hidden_dim * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Conv2d(hidden_dim, se_reduced_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(se_reduced_dim, hidden_dim, kernel_size=1, bias=True),
            nn.Sigmoid() # Sigmoid for channel gating
        )

        # Projection phase (1x1 Conv)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            LayerNorm(out_dim)
        )

        # Skip connection
        self.use_shortcut = stride == 1 and in_dim == out_dim
        if not self.use_shortcut:
             # If dimensions/stride change, use a 1x1 Conv in shortcut
            self.shortcut_conv = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=0, bias=False),
                LayerNorm(out_dim)
            )

    def forward(self, x):
        shortcut = x # Store original input

        # Main path
        out = self.expand(x)
        out = self.dwconv(out)
        out = self.norm_dw(out)
        out = self.act_dw(out)
        out = out * self.se(out) # Apply SE block
        out = self.project(out)

        # Apply skip connection
        if self.use_shortcut:
            out = shortcut + self.drop_path(out)
        else:
             out = self.shortcut_conv(shortcut) + self.drop_path(out) # Apply conv to shortcut first

        return out


class Attention(nn.Module):
    """ Convolutional Attention block. """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Use Conv2d for QKV projection - operates on (B, C, H, W)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # Use Conv2d for output projection
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W # Sequence length equivalent
        qkv = self.qkv(x) # (B, 3*C, H, W)

        # Reshape and permute for attention calculation
        # (B, 3*C, H, W) -> (B, 3, num_heads, head_dim, N) -> (3, B, num_heads, N, head_dim)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2] # Each (B, num_heads, N, head_dim)

        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values and reshape back
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        # -> (B, num_heads, head_dim, N) -> (B, C, N) -> (B, C, H, W)
        out = (attn @ v).transpose(-2, -1).reshape(B, C, N).view(B, C, H, W)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class TransformerBlock(nn.Module):
    """ Standard Transformer Block using Convolutional Attention and MLP. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Note: Norm is applied *before* attention/MLP, standard Pre-LN structure
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class StageBlock(nn.Module):
    """ A stage consisting of multiple MBConv or Transformer blocks. """
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm, is_mbconv=False):
        super().__init__()
        # drop_path can be a list or scalar. If scalar, create list for blocks.
        if isinstance(drop_path, (list, tuple)):
             dp_rates = drop_path
        else:
             dp_rates = [drop_path] * depth

        self.blocks = nn.ModuleList([])
        for i in range(depth):
            if is_mbconv:
                # Pass appropriate parameters for MBConv used here
                self.blocks.append(MBConv(
                    in_dim=dim,
                    out_dim=dim,
                    expand_ratio=mlp_ratio, # Assuming mlp_ratio serves as expand_ratio for MBConv stages
                    drop_path=dp_rates[i]
                ))
            else:
                self.blocks.append(TransformerBlock(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop, attn_drop=attn_drop, drop_path=dp_rates[i],
                    act_layer=act_layer, norm_layer=norm_layer))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding using a Convolution. """
    def __init__(self, in_chans=3, dim=96): # Default embed_dim=96 for 11M/21M variants
        super().__init__()
        # 7x7 conv, stride 4 -> H/4 x W/4 patches
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=7, stride=4, padding=3)
        self.norm = LayerNorm(dim) # Followed by LayerNorm

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x) # Apply norm after projection
        return x

class TinyViT11M(nn.Module):
    """
    TinyViT-11M model.

    Args:
        img_size (int): Input image size. Default 224.
        in_chans (int): Number of input image channels. Default 3.
        num_classes (int): Number of classes for classification head. Default 1000.
        embed_dims (list[int]): Embedding dimension for each stage. Default [96, 192, 384, 768] for 21M? Needs check for 11M.
                                Let's use the report's 11M target params: [64, 128, 256, 512] might be closer.
        depths (list[int]): Depth (number of blocks) for each stage. Default [2, 2, 6, 2].
        num_heads (list[int]): Number of attention heads for each *Transformer* stage. Default [3, 6, 12, 24]? Needs check for 11M.
                                Let's use [2, 4, 8, 16] as potentially closer to 11M.
        mlp_ratios (list[float]): Ratio of MLP hidden dim to embedding dim. Default [4, 4, 4, 4].
        drop_rate (float): Dropout rate. Default 0.0.
        attn_drop_rate (float): Attention dropout rate. Default 0.0.
        drop_path_rate (float): Stochastic depth rate. Default 0.1.
        norm_layer (nn.Module): Normalization layer. Default LayerNorm (custom).
        act_layer (nn.Module): Activation function. Default nn.GELU.
        mbconv_stages (int): Number of initial stages to use MBConv blocks. Default 2.
    """
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512], # Based on report's 11M target
                 depths=[2, 2, 6, 2],           # Common structure
                 num_heads=[2, 4, 8, 16],         # Scaled down from 21M defaults
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNorm, act_layer=nn.GELU, mbconv_stages=2):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(depths)

        # Patch embedding (using embed_dims[0])
        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=embed_dims[0])

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Build stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            is_mbconv = (i < mbconv_stages)
            stage = StageBlock(
                dim=embed_dims[i],
                depth=depths[i],
                # num_heads is ignored if is_mbconv is True in StageBlock logic
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=True, # Common practice
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur : cur + depths[i]], # Pass the list slice for this stage
                norm_layer=norm_layer,
                act_layer=act_layer,
                is_mbconv=is_mbconv
            )
            self.stages.append(stage)
            cur += depths[i]

            # Downsampling between stages (except after the last stage)
            if i < self.num_stages - 1:
                # Downsampling uses norm -> conv2d
                self.stages.append(
                    nn.Sequential(
                        norm_layer(embed_dims[i]),
                        nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=2, stride=2)
                    )
                )

        # Classifier head
        self.norm = norm_layer(embed_dims[-1]) # Norm before pooling
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
             trunc_normal_(m.weight, std=0.02)
             if m.bias is not None:
                  nn.init.zeros_(m.bias)
        # LayerNorm weights/biases are initialized in its definition

    def forward_features(self, x):
        x = self.patch_embed(x) # (B, C0, H/4, W/4)

        # Iterate through stages (which include StageBlocks and Downsampling)
        for stage_module in self.stages:
            x = stage_module(x)

        # Apply final norm and Global Average Pooling
        x = self.norm(x)      # (B, C_last, H_final, W_final) -> (B, C_last, H_final, W_final)
        x = x.mean([2, 3]) # Global Avg Pooling: (B, C_last, H, W) -> (B, C_last)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def tinyvit_11m(pretrained=False, **kwargs):
    """
    TinyViT-11M model. Aims for approximately 11M parameters.
    Uses specific MBConv (7x7 dw + SE) and ConvAttention blocks defined locally.

    Note: Parameter count depends heavily on embed_dims, depths, num_heads.
          The defaults [64, 128, 256, 512] with depths [2, 2, 6, 2]
          should be checked against the 11M target.
    """
    if pretrained:
        raise NotImplementedError("Pretrained weights are not available for this implementation.")

    # Default parameters aiming for ~11M
    model_kwargs = dict(
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.1, # Default from paper/common practice
        mbconv_stages=2,   # Use MBConv for first 2 stages
        **kwargs
    )
    model = TinyViT11M(**model_kwargs)

    # Optional: Add parameter count check here if needed
    # param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Instantiated tinyvit_11m with ~{param_count/1e6:.2f}M parameters.")

    return model

# Verify model and parameter count
if __name__ == "__main__":
    # Test with default parameters aiming for 11M
    model = tinyvit_11m(num_classes=1000)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_count:,} (~{param_count/1e6:.2f}M)") # Should be ~11M

    # Example check for layer names (useful for attention viz)
    # print("\nModel Layer Names (containing 'attn'):")
    # for name, module in model.named_modules():
    #     if 'attn' in name.lower():
    #          print(name)
