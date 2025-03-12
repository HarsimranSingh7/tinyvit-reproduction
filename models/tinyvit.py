import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .blocks import PatchEmbed, PatchMerging, MBConv, TransformerBlock

class TinyViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_channels=3,
        num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        mlp_ratios=[4, 4, 4, 4],
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_size=4,
        use_mbconv=True,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(depths)
        self.embed_dims = embed_dims
        self.use_mbconv = use_mbconv
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer
        )
        
        # Build stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.stages = nn.ModuleList()
        curr_idx = 0
        
        for i in range(self.num_stages):
            # Each stage consists of blocks and an optional patch merging layer
            stage = nn.ModuleList()
            
            # Add blocks
            if i == 0 and use_mbconv:
                # First stage uses MBConv blocks
                for j in range(depths[i]):
                    stage.append(
                        MBConv(
                            in_channels=embed_dims[i],
                            out_channels=embed_dims[i],
                            expand_ratio=mlp_ratios[i],
                            drop_path=dpr[curr_idx]
                        )
                    )
                    curr_idx += 1
            else:
                # Other stages use Transformer blocks
                for j in range(depths[i]):
                    stage.append(
                        TransformerBlock(
                            dim=embed_dims[i],
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratios[i],
                            drop=drop_rate,
                            attn_drop=0.,
                            drop_path=dpr[curr_idx],
                            norm_layer=norm_layer,
                            layer_scale_init_value=layer_scale_init_value
                        )
                    )
                    curr_idx += 1
            
            self.stages.append(stage)
            
            # Add patch merging layer except for the last stage
            if i < self.num_stages - 1:
                self.stages.append(
                    PatchMerging(
                        in_channels=embed_dims[i],
                        out_channels=embed_dims[i+1],
                        norm_layer=norm_layer
                    )
                )
        
        # Final norm layer
        self.norm = norm_layer(embed_dims[-1])
        
        # Classifier head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        
        # Process through stages
        for stage in self.stages:
            if isinstance(stage, nn.ModuleList):
                for block in stage:
                    x = block(x)
            else:
                x = stage(x)
        
        # Global average pooling
        x = self.norm(x.mean(dim=1))
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def tinyvit_5m(pretrained=False, **kwargs):
    """TinyViT-5M model with ~5.4M parameters"""
    model = TinyViT(
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        mlp_ratios=[4, 4, 4, 4],
        **kwargs
    )
    return model