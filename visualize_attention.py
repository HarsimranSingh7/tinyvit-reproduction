import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2 # Using cv2 for resizing attention maps smoothly
import argparse

# Import your models (make sure they are importable)
from models.tinyvit import tinyvit_5m
from models.tinyvit_11m import tinyvit_11m # Assuming this structure

# Global variable to store attention maps from hooks
attention_maps = {}

def get_attention_hook(layer_name):
    """Returns a hook function to capture attention maps."""
    def hook(model, input, output):
        # Hook location depends on the Attention block implementation
        # Assuming output is the attention map *before* softmax or after value multiplication
        # For typical ViT attention: output shape might be (B, NumHeads, SeqLen, SeqLen) or similar
        # For TinyViT (using Conv2d attention), shape might be (B, C, H, W) after projection
        # **This needs verification based on your specific Attention block's forward pass**

        # --- Placeholder: Assuming output is the raw attention scores/map ---
        # --- Adapt this based on your actual Attention module structure ---
        if isinstance(output, torch.Tensor):
             # Example: Average over heads if shape is (B, H, N, N)
             # if len(output.shape) == 4 and output.shape[1] > 1: # B, Heads, N, N
             #      attn_map = output.mean(dim=1) # Average heads: (B, N, N)
             # else:
             #      attn_map = output

             # Simplification: Store the raw output for now
             attention_maps[layer_name] = output.detach().cpu()
             print(f"Captured output from {layer_name}, shape: {output.shape}") # Debug print
        elif isinstance(output, tuple): # Sometimes blocks return tuples
             # Try to find a tensor that looks like attention
             for item in output:
                  if isinstance(item, torch.Tensor) and item.ndim >= 3: # Guessing it's attention
                      attention_maps[layer_name] = item.detach().cpu()
                      print(f"Captured tensor output (tuple element) from {layer_name}, shape: {item.shape}")
                      break # Take the first likely candidate

    return hook

def preprocess_image(image_path, img_size=224):
    """Preprocess an image for model input."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), # Resize directly to target
        # transforms.CenterCrop(img_size), # Not needed if resizing directly
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        # Return the resized original image for visualization
        img_np = np.array(img.resize((img_size, img_size)))
        return img_tensor, img_np
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None


def visualize_attention_map(model, model_name, target_layer_path, image_path, output_dir, device, img_size=224):
    """
    Loads a model, extracts attention from a target layer for an image, and saves visualization.

    Args:
        model (nn.Module): The loaded model instance.
        model_name (str): Name of the model (for titles).
        target_layer_path (str): Dot-separated path to the target attention layer (e.g., 'stages.6.1.attn').
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the visualization.
        device (torch.device): Device to run inference on.
        img_size (int): Input image size.
    """
    global attention_maps # Use the global dict
    attention_maps = {} # Clear previous maps

    img_tensor, img_np = preprocess_image(image_path, img_size)
    if img_tensor is None:
        return

    model.eval()
    model.to(device)
    img_tensor = img_tensor.to(device)

    # Register the hook
    hook_handle = None
    try:
        target_layer = model
        for part in target_layer_path.split('.'):
            target_layer = getattr(target_layer, part)
        hook_handle = target_layer.register_forward_hook(get_attention_hook(target_layer_path))
        print(f"Registered hook on: {target_layer_path}")
    except AttributeError:
        print(f"Error: Could not find layer path '{target_layer_path}' in the model.")
        print("Please check the model structure and layer names.")
        # You might want to add code here to print model.named_modules() to help find names
        # for name, module in model.named_modules():
        #     print(name)
        return
    except Exception as e:
        print(f"Error registering hook: {e}")
        return

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        # Optional: print top prediction
        probs = F.softmax(output, dim=1)
        top_p, top_class = probs.topk(1, dim=1)
        print(f"Prediction: Class {top_class.item()} with probability {top_p.item():.4f}")

    # Remove the hook
    if hook_handle:
        hook_handle.remove()

    # Process and visualize the captured attention map
    if target_layer_path not in attention_maps:
        print(f"Error: Attention map for '{target_layer_path}' was not captured by the hook.")
        print("Check hook implementation and ensure the layer produces suitable output.")
        return

    attn_map_raw = attention_maps[target_layer_path][0] # Get first item in batch

    # --- Process the attention map ---
    # This part is HIGHLY DEPENDENT on the specific output shape of your hook
    # Example for ViT-like attention (NumHeads, SeqLen, SeqLen):
    # attn_map = attn_map_raw.mean(dim=0) # Average heads: (SeqLen, SeqLen)
    # # We need attention to the [CLS] token or average attention? Let's average.
    # attn_map = attn_map.mean(dim=0) # Average attention *from* all tokens: (SeqLen,)
    # attn_map = attn_map[1:] # Remove CLS token attention value
    # grid_size = int(np.sqrt(attn_map.shape[0]))
    # attn_map = attn_map.reshape(grid_size, grid_size)

    # Example for Conv-Attention (C, H, W) - maybe average channels?
    if attn_map_raw.ndim == 3: # C, H, W
         attn_map = attn_map_raw.mean(dim=0) # Average channels -> (H, W)
    elif attn_map_raw.ndim == 2: # H, W (already averaged?)
         attn_map = attn_map_raw
    else:
         print(f"Error: Unexpected attention map dimension: {attn_map_raw.ndim}. Shape: {attn_map_raw.shape}")
         print("Cannot process this attention map shape. Adapt the processing logic.")
         return

    # Convert to numpy
    attn_map_np = attn_map.numpy()

    # Resize attention map to image size using cv2 for smoother interpolation
    attn_map_resized = cv2.resize(attn_map_np, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Normalize the resized map
    attn_map_resized = (attn_map_resized - np.min(attn_map_resized)) / (np.max(attn_map_resized) - np.min(attn_map_resized))

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Attention map heatmap
    im = axes[1].imshow(attn_map_resized, cmap='jet')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)


    # Overlay
    axes[2].imshow(img_np)
    axes[2].imshow(attn_map_resized, alpha=0.5, cmap='jet') # Adjust alpha for visibility
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')

    plt.suptitle(f'{model_name} - Attention from: {target_layer_path}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    output_filename = os.path.join(output_dir, f"{model_name}_attention_{os.path.basename(image_path)}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Attention visualization saved to {output_filename}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser('Attention Visualization', add_help=False)
    parser.add_argument('--image-path', required=True, type=str, help="Path to the input image.")
    parser.add_argument('--model-path', required=True, type=str, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--model-variant', required=True, type=str, choices=['tinyvit_5m', 'tinyvit_11m'], help="Specify which TinyViT variant to load.")
    parser.add_argument('--target-layer', required=True, type=str, help="Dot-separated path to the target attention layer (e.g., 'stages.6.1.attn'). Find names by printing model.named_modules().")
    parser.add_argument('--num-classes', type=int, default=1000, help="Number of classes the model was trained on.")
    parser.add_argument('--img-size', type=int, default=224, help="Input image size.")
    parser.add_argument('--output-dir', default='./attention_maps_real', type=str, help="Directory to save the output visualization.")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Device to use (cuda or cpu).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load model architecture
    if args.model_variant == 'tinyvit_5m':
        model = tinyvit_5m(num_classes=args.num_classes)
        model_name_prefix = "TinyViT-5M"
    elif args.model_variant == 'tinyvit_11m':
        model = tinyvit_11m(num_classes=args.num_classes)
        model_name_prefix = "TinyViT-11M"
    else:
        # Should not happen due to choices constraint
        print(f"Error: Unknown model variant {args.model_variant}")
        return

    # Load model weights
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        # Adjust keys based on how you saved the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Handle potential DataParallel prefix 'module.'
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        # Optionally allow continuing with random weights for debugging structure
        # print("Warning: Continuing with randomly initialized weights.")
        return # Typically fail if weights can't load

    # Generate visualization
    visualize_attention_map(
        model=model,
        model_name=model_name_prefix,
        target_layer_path=args.target_layer,
        image_path=args.image_path,
        output_dir=args.output_dir,
        device=device,
        img_size=args.img_size
    )

if __name__ == '__main__':
    main()
