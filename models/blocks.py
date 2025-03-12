import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B,C,H,W -> B,C,HW -> B,HW,C
        x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    """
    Patch Merging Layer for hierarchical feature maps
    """
    def __init__(self, in_channels, out_channels, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm_layer(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)
        
    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        
        x = x.view(B, H, W, C)
        
        # Gather patches in 2x2 grid
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
        x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C)  # B, H/2*W/2, 4*C
        
        x = self.norm(x)
        x = self.reduction(x)  # B, H/2*W/2, out_channels
        
        return x

class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Conv (MBConv) block
    """
    def __init__(self, in_channels, out_channels, expand_ratio=4, drop_path=0.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )
        
        # Depthwise convolution
        self.dwconv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, 3, padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )
        
        # Projection phase
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.skip = in_channels == out_channels
        
    def forward(self, x):
        """
        x: B, C, H, W
        """
        # Reshape to 4D if input is 3D (B, L, C)
        if len(x.shape) == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.transpose(1, 2).view(B, C, H, W)
            reshape_needed = True
        else:
            reshape_needed = False
            
        shortcut = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Projection
        x = self.project_conv(x)
        
        # Skip connection
        if self.skip:
            x = shortcut + self.drop_path(x)
            
        # Reshape back if needed
        if reshape_needed:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)
            
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    Multi-head Attention block
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer Block with LayerScale and Stochastic Depth
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # Layer scale for attention
        self.ls1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # Layer scale for MLP
        self.ls2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        
    def forward(self, x):
        if self.ls1 is not None:
            x = x + self.drop_path(self.ls1.unsqueeze(0).unsqueeze(0) * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ls2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x