# models/tinyvit.py
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath # Ensure DropPath is imported if used implicitly
from .blocks import PatchEmbed, PatchMerging, MBConv, TransformerBlock

class TinyViT(nn.Module):
    """
    TinyViT model architecture.

    Based on the paper: "TinyViT: Fast Pretraining Distillation for Small Vision Transformers"
    This implementation focuses on the ConvNet-based variants.

    Args:
        img_size (int): Input image size. Default: 224.
        in_channels (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        embed_dims (list[int]): Embedding dimension for each stage.
        depths (list[int]): Depth (number of blocks) for each stage.
        num_heads (list[int]): Number of attention heads for each Transformer stage.
        mlp_ratios (list[float]): Ratio of MLP hidden dim to embedding dim for each stage.
        drop_rate (float): Dropout rate. Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_size (int): Patch size for embedding. Default: 4.
        use_mbconv (bool): Whether to use MBConv blocks in the initial stage(s). Default: True.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-5. Set 0 to disable.
    """
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
        use_mbconv=True, # Typically True for TinyViT 5M/11M initial stages
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(depths)
        self.embed_dims = embed_dims
        self.use_mbconv = use_mbconv # Store this if needed later, though logic uses it directly

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer # Pass norm_layer here if PatchEmbed supports it
        )

        # Build stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.stages = nn.ModuleList()
        curr_idx = 0

        for i in range(self.num_stages):
            # Each stage consists of blocks and an optional patch merging layer
            blocks = nn.ModuleList() # Use 'blocks' for clarity within the stage

            # Add blocks
            # Determine if current stage uses MBConv (e.g., only first stage)
            is_mbconv_stage = (i == 0 and use_mbconv) # Example: Only stage 0 uses MBConv

            if is_mbconv_stage:
                # First stage uses MBConv blocks
                # Note: Original TinyViT paper might use MBConv in first two stages. Adjust logic if needed.
                for j in range(depths[i]):
                    blocks.append(
                        MBConv(
                            in_channels=embed_dims[i], # MBConv in blocks.py takes in_channels, out_channels
                            out_channels=embed_dims[i],
                            expand_ratio=mlp_ratios[i],
                            drop_path=dpr[curr_idx + j] # Access dpr correctly
                        )
                    )
            else:
                # Other stages use Transformer blocks
                for j in range(depths[i]):
                    blocks.append(
                        TransformerBlock(
                            dim=embed_dims[i],
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratios[i],
                            drop=drop_rate,
                            attn_drop=0., # Standard practice to keep attn_drop=0 unless specified
                            drop_path=dpr[curr_idx + j], # Access dpr correctly
                            norm_layer=norm_layer,
                            layer_scale_init_value=layer_scale_init_value
                        )
                    )

            self.stages.append(blocks) # Append the ModuleList of blocks for this stage
            curr_idx += depths[i] # Increment current index *after* processing the stage depth

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
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)): # Include BatchNorm if used
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d): # Basic Conv2d init if needed
             trunc_normal_(m.weight, std=.02)
             if m.bias is not None:
                  nn.init.zeros_(m.bias)


    def forward_features(self, x):
        x = self.patch_embed(x)

        # Process through stages
        stage_idx = 0
        while stage_idx < len(self.stages):
            blocks = self.stages[stage_idx]
            # Apply blocks in the stage
            for blk in blocks:
                 x = blk(x)
            stage_idx += 1
            # Apply patch merging if it exists
            if stage_idx < len(self.stages) and isinstance(self.stages[stage_idx], PatchMerging):
                 x = self.stages[stage_idx](x)
                 stage_idx += 1

        # Global average pooling (common practice for image classification)
        # Input to norm might be (B, L, C)
        x = self.norm(x) # Apply norm first
        x = x.mean(dim=1) # Global Average Pooling: (B, L, C) -> (B, C)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def tinyvit_5m(pretrained=False, **kwargs):
    """
    TinyViT-5M model. Approximately 5.4M parameters.
    Uses MBConv blocks in the first stage and Transformer blocks in later stages.
    """
    if pretrained:
        # Add logic here if you plan to host/load pretrained weights
        raise NotImplementedError("Pretrained weights are not available for this implementation.")

    model = TinyViT(
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],         # Corresponds to embed_dims length
        num_heads=[2, 4, 5, 10],     # Corresponds to Transformer stages (last 3?) - check consistency
                                     # If stage 0 is MBConv, it doesn't use heads. Adjust num_heads list?
                                     # Let's assume num_heads maps to embed_dims, and is ignored if MBConv used.
        mlp_ratios=[4, 4, 4, 4],
        use_mbconv=True, # Explicitly True for 5M variant style
        **kwargs
    )
    return model

def tinyvit_finetune(num_classes, pretrained_model_path=None, checkpoint_key='state_dict'):
    """
    Loads a pre-trained TinyViT-5M model, replaces the classification head
    for the specified number of classes, and freezes all layers except the head.

    Args:
        num_classes (int): Number of classes for the new classification task.
        pretrained_model_path (str): Path to the pre-trained TinyViT-5M checkpoint (.pth).
        checkpoint_key (str): The key in the checkpoint dictionary holding the model's state_dict.
                              Common keys: 'state_dict', 'model_state_dict', 'model'. If the checkpoint
                              is just the state_dict, use None or adjust loading.

    Returns:
        torch.nn.Module: The modified TinyViT model ready for fine-tuning the head.
    """
    # Create base model architecture (ensure parameters match the pretrained one)
    model = tinyvit_5m(num_classes=num_classes) # Create with the *new* number of classes initially

    if pretrained_model_path:
        try:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu') # Load to CPU first

            if checkpoint_key and checkpoint_key in checkpoint:
                state_dict = checkpoint[checkpoint_key]
            elif 'model_state_dict' in checkpoint: # Common alternative
                 state_dict = checkpoint['model_state_dict']
            else: # Assume the checkpoint *is* the state_dict
                state_dict = checkpoint

            # Handle potential 'module.' prefix from DataParallel saving
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Load weights, ignoring the head (due to mismatch if num_classes changed)
            # strict=False allows loading weights into the body even if head weights are missing/mismatched
            load_result = model.load_state_dict(state_dict, strict=False)

            # Check loading results (optional but good practice)
            if load_result.missing_keys:
                print(f"Warning: Missing keys during fine-tune load: {load_result.missing_keys}")
                # Expected missing: 'head.weight', 'head.bias' if num_classes differs
            if load_result.unexpected_keys:
                 print(f"Warning: Unexpected keys during fine-tune load: {load_result.unexpected_keys}")

            print(f"Loaded pre-trained weights from {pretrained_model_path}")

        except FileNotFoundError:
            print(f"Error: Pretrained model file not found at {pretrained_model_path}")
            print("Model will have random weights.")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Model will have random weights.")

    # Freeze all layers except the classification head
    num_frozen = 0
    for name, param in model.named_parameters():
        if "head." not in name: # Freeze layers *not* in the head module
            param.requires_grad = False
            num_frozen += 1
        else:
            param.requires_grad = True # Ensure head is trainable

    print(f"Froze {num_frozen} parameters. Only the head is trainable.")

    return model
