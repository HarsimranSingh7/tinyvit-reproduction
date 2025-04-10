import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MBConv(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, expand_ratio=4, drop_path=0.):
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Expansion
        exp_dim = int(in_dim * expand_ratio)
        self.conv1 = nn.Conv2d(in_dim, exp_dim, 1, 1, 0, bias=False)
        self.bn1 = LayerNorm(exp_dim)
        
        # Depthwise
        self.conv2 = nn.Conv2d(exp_dim, exp_dim, 7, stride, 3, groups=exp_dim, bias=False)
        self.bn2 = LayerNorm(exp_dim)
        
        # SE Layer
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(exp_dim, exp_dim // 4, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv2d(exp_dim // 4, exp_dim, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        
        # Projection
        self.conv3 = nn.Conv2d(exp_dim, out_dim, 1, 1, 0, bias=False)
        self.bn3 = LayerNorm(out_dim)
        
        # Identity path
        if stride == 1 and in_dim == out_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, stride, 0, bias=False),
                LayerNorm(out_dim)
            )
            
    def forward(self, x):
        # Expansion
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        
        # Depthwise
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out)
        
        # SE
        out = out * self.se(out)
        
        # Projection
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Residual
        out = self.shortcut(x) + self.drop_path(out)
        return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
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
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class StageBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm, is_mbconv=False):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            if is_mbconv:
                self.blocks.append(MBConv(dim, dim, 1, mlp_ratio, drop_path=drop_path))
            else:
                self.blocks.append(TransformerBlock(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop, attn_drop=attn_drop, drop_path=drop_path, 
                    act_layer=act_layer, norm_layer=norm_layer))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=7, stride=4, padding=3)
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class TinyViT11M(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=LayerNorm, 
                 act_layer=nn.GELU):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(depths)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=embed_dims[0])
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        # Build stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = StageBlock(
                dim=embed_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur:cur+depths[i]],
                norm_layer=norm_layer,
                act_layer=act_layer,
                is_mbconv=(i < 2)  # Use MBConv for first two stages
            )
            self.stages.append(stage)
            cur += depths[i]
            
            # Downsampling between stages
            if i < self.num_stages - 1:
                self.stages.append(
                    nn.Sequential(
                        norm_layer(embed_dims[i]),
                        nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=2, stride=2)
                    )
                )
        
        # Classifier head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        
        for stage in self.stages:
            x = stage(x)
        
        # Global average pooling
        x = self.norm(x)
        x = x.mean([2, 3])
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def tinyvit_11m(pretrained=False, **kwargs):
    model = TinyViT11M(
        embed_dims=[64, 128, 256, 512],
        depths=[1, 2, 4, 2],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        **kwargs
    )
    return model

# Verify model and parameter count
if __name__ == "__main__":
    model = tinyvit_11m()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_count:,}")
