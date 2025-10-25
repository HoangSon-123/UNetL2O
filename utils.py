# file: utils.py
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import config

def set_seed(seed_value):
    """Đặt seed cho các thư viện để đảm bảo tính tái lập."""
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def count_parameters(model):
    """Đếm số lượng tham số có thể huấn luyện trong model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Lưu checkpoint của model."""
    print(f"=> Saving checkpoint to {filename}")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    """Tải checkpoint của model."""
    print(f"=> Loading checkpoint from {filename}")
    if not os.path.exists(filename):
        print("! Checkpoint not found, starting from scratch.")
        return model, optimizer, 0, config.BEST_LOSS_INIT
        
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def display_model_outputs(model, test_loader, device, num_samples=3):
    """Trực quan hóa kết quả dự đoán so với ảnh gốc."""
    model.eval()
    sample_count = 0
    print("Visualizing model outputs...")
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            for i in range(inputs.size(0)):
                if sample_count >= num_samples:
                    return

                inp = inputs[i].cpu().squeeze()
                out = outputs[i].cpu().squeeze()
                tgt = targets[i].cpu().squeeze()

                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.imshow(inp, cmap='gray')
                plt.title(f'Input (Observed Data) - Sample {sample_count+1}')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(out, cmap='gray')
                plt.title('Output (Reconstructed)')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(tgt, cmap='gray')
                plt.title('Target (Ground Truth)')
                plt.axis('off')

                plt.tight_layout()
                plt.show()

                sample_count += 1
                print(f"Sample count: {sample_count}")