import torch
from models.tinyvit import tinyvit_5m

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Create model
    model = tinyvit_5m()
    
    # Count parameters
    params = count_parameters(model)
    print(f"TinyViT-5M parameters: {params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Check if close to 5.4M params
    expected_params = 5.4e6
    diff_percent = abs(params - expected_params) / expected_params * 100
    
    if diff_percent < 5:
        print(f"✅ Parameter count within 5% of expected ({diff_percent:.2f}%)")
    else:
        print(f"❌ Parameter count differs by {diff_percent:.2f}%")

if __name__ == "__main__":
    main()