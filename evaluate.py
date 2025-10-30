# file: evaluate_general.py
import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import mse_loss
import numpy as np
import os

import config
from dataset import get_dataloaders
from utils import load_checkpoint, display_model_outputs, set_seed

# Import models
from model import *


def build_model(A, device):
    """Automatically initialize the model according to config.MODEL_TYPE"""
    model_type = config.MODEL_TYPE.lower()

    if model_type == "tvm":
        print("Using CT_TVM_Model")
        model = CT_TVM_Model(A=A).to(device)

    elif model_type == "ffpn":
        print("Using CT_FFPN_Model")
        S = torch.diag(torch.count_nonzero(A, dim=0) ** -1.0).float().to(device)
        model = CT_FFPN_Model(S=S, A=A).to(device)

    elif model_type == "unet":
        print("Using CT_UNet_Model")
        model = CT_UNet_Model(in_channels=1, out_channels=1, features=config.UNET_FEATURES).to(device)

    elif model_type == "unet_l2o":
        print("Using UNetL2O")
        model = UNetL2O(
            A=A,
            lambd=config.LAMBD,
            alpha=config.ALPHA,
            beta=config.BETA,
            delta=config.DELTA,
            K_out_channels=config.K_OUT_CHANNELS,
            max_depth=config.MAX_DEPTH
        ).to(device)

    elif model_type == "ct_l2o":
        print("Using CT_L2O_Model")
        model = CT_L2O_Model(
            A=A,
            lambd=config.LAMBD,
            alpha=config.ALPHA,
            beta=config.BETA,
            delta=config.DELTA,
            K_out_channels=config.K_OUT_CHANNELS,
            max_depth=config.MAX_DEPTH
        ).to(device)

    elif model_type == "scale_ct_l2o":
        print("Using Scale_CT_L2O_Model")
        model = Scale_CT_L2O_Model(
            A=A,
            lambd=config.LAMBD,
            alpha=config.ALPHA,
            beta=config.BETA,
            delta=config.DELTA,
            K_out_channels=config.K_OUT_CHANNELS,
            max_depth=config.MAX_DEPTH
        ).to(device)

    else:
        raise ValueError(f"Invalid MODEL_TYPE '{config.MODEL_TYPE}'!")

    return model


def evaluate_model():
    """Evaluate the model on the test dataset"""
    set_seed(config.SEED)
    print("Loading test data...")
    _, test_loader, A = get_dataloaders(config.BATCH_SIZE)
    print("Test data ready.")

    # Initialize model
    model = build_model(A, config.DEVICE)
    model_save_path = f"{config.MODEL_TYPE}.pth"

    # Check for checkpoint
    if not os.path.exists(model_save_path):
        print(f"Checkpoint not found: {model_save_path}")
        print("Please run 'train_general.py' to train the model first.")
        return

    # Load checkpoint
    optimizer = torch.optim.Adam(model.parameters())
    model, _, _, _ = load_checkpoint(model, optimizer, model_save_path)
    model.eval()

    print(f"\n--- Evaluating model [{config.MODEL_TYPE}] ---")

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_mse = 0.0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", ascii=True)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(inputs)

            for output, target in zip(outputs, targets):
                output_np = output.cpu().squeeze().numpy()
                target_np = target.cpu().squeeze().numpy()

                avg_psnr += psnr(target_np, output_np, data_range=1.0)
                avg_ssim += ssim(target_np, output_np, data_range=1.0)
                avg_mse += mse_loss(output, target).item()

    avg_psnr /= len(test_loader.dataset)
    avg_ssim /= len(test_loader.dataset)
    avg_mse /= len(test_loader.dataset)

    print("\n--- Evaluation Results ---")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE:  {avg_mse:.6f}")

    # Visualization
    print("\n--- Visualizing a few samples ---")
    display_model_outputs(model, test_loader, config.DEVICE, num_samples=3)


if __name__ == "__main__":
    evaluate_model()
