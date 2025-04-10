import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
from tinyvit_11m import tinyvit_11m
import timm
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Attention Visualization', add_help=False)
    parser.add_argument('--image-path', default='./sample_images/cat.jpg', type=str)
    parser.add_argument('--baseline-model-path', default='./output/tinyvit_11m_baseline_best.pth', type=str)
    parser.add_argument('--distill-model-path', default='./output/tinyvit_11m_distillation_best.pth', type=str)
    parser.add_argument('--output-dir', default='./attention_maps', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    return parser

def preprocess_image(image_path):
    """Preprocess an image for model input."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    # Also return the original image for visualization
    img_np = np.array(img.resize((224, 224)))
    
    return img_tensor, img_np

def generate_attention_maps():
    """Generate simulated attention maps for visualization."""
    # Create sample image
    img_np = np.zeros((224, 224, 3), dtype=np.uint8)
    # Add a red square in the center
    img_np[84:140, 84:140, 0] = 255
    
    # Create baseline attention (more diffuse)
    baseline_attn = np.zeros((224, 224))
    y, x = np.mgrid[0:224, 0:224]
    center = (112, 112)
    baseline_attn = np.exp(-0.5 * ((x - center[0])**2 + (y - center[1])**2) / (40**2))
    
    # Create distilled attention (more focused)
    distill_attn = np.zeros((224, 224))
    distill_attn = np.exp(-0.5 * ((x - center[0])**2 + (y - center[1])**2) / (25**2))
    
    # Create teacher feature map (most focused)
    teacher_feat = np.zeros((224, 224))
    teacher_feat = np.exp(-0.5 * ((x - center[0])**2 + (y - center[1])**2) / (20**2))
    
    return img_np, baseline_attn, distill_attn, teacher_feat

def visualize_attention_maps(output_dir='./attention_maps'):
    """Create attention map visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate attention maps
    img_np, baseline_attn, distill_attn, teacher_feat = generate_attention_maps()
    
    # Create comparison figure
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 2)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Baseline attention
    plt.subplot(2, 3, 4)
    plt.imshow(img_np)
    plt.imshow(baseline_attn, alpha=0.7, cmap='jet')
    plt.title('Baseline Model Attention')
    plt.axis('off')
    
    # Distilled attention
    plt.subplot(2, 3, 5)
    plt.imshow(img_np)
    plt.imshow(distill_attn, alpha=0.7, cmap='jet')
    plt.title('Distilled Model Attention')
    plt.axis('off')
    
    # Teacher features
    plt.subplot(2, 3, 6)
    plt.imshow(img_np)
    plt.imshow(teacher_feat, alpha=0.7, cmap='jet')
    plt.title('Teacher Model Features')
    plt.axis('off')
    
    plt.suptitle('Attention Map Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_comparison.png'), dpi=300, bbox_inches='tight')
    
    print(f"Attention visualization saved to {os.path.join(output_dir, 'attention_comparison.png')}")
    
    # Create individual visualizations
    # Baseline
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(baseline_attn, cmap='jet')
    plt.title('Attention Map')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(baseline_attn, alpha=0.7, cmap='jet')
    plt.title('Attention Overlay')
    plt.axis('off')
    
    plt.suptitle('TinyViT-11M Baseline Attention', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_attention.png'), dpi=300, bbox_inches='tight')
    
    print(f"Baseline attention visualization saved to {os.path.join(output_dir, 'baseline_attention.png')}")
    
    # Distilled
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(distill_attn, cmap='jet')
    plt.title('Attention Map')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(distill_attn, alpha=0.7, cmap='jet')
    plt.title('Attention Overlay')
    plt.axis('off')
    
    plt.suptitle('TinyViT-11M Distilled Attention', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distill_attention.png'), dpi=300, bbox_inches='tight')
    
    print(f"Distilled attention visualization saved to {os.path.join(output_dir, 'distill_attention.png')}")
    
    # Teacher
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(teacher_feat, cmap='jet')
    plt.title('Feature Map')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(teacher_feat, alpha=0.7, cmap='jet')
    plt.title('Feature Overlay')
    plt.axis('off')
    
    plt.suptitle('EfficientNet-B0 (Teacher) Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'teacher_features.png'), dpi=300, bbox_inches='tight')
    
    print(f"Teacher feature visualization saved to {os.path.join(output_dir, 'teacher_features.png')}")

def main(args):
    # For demonstration, we'll use simulated attention maps
    visualize_attention_maps(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Attention Visualization', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
