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

# Import các mô hình
from model import*


def build_model(A, device):
    """Tự động khởi tạo model theo config.MODEL_TYPE"""
    model_type = config.MODEL_TYPE.lower()

    if model_type == "tvm":
        print("🔧 Sử dụng CT_TVM_Model")
        model = CT_TVM_Model(A=A).to(device)

    elif model_type == "ffpn":
        print("🔧 Sử dụng CT_FFPN_Model")
        S = torch.diag(torch.count_nonzero(A, dim=0) ** -1.0).float().to(device)
        model = CT_FFPN_Model(S=S, A=A).to(device)

    elif model_type == "unet":
        print("🔧 Sử dụng CT_UNet_Model")
        model = CT_UNet_Model(in_channels=1, out_channels=1, features=config.UNET_FEATURES).to(device)

    elif model_type == "new_ct_l2o":
        print("🔧 Sử dụng New_CT_L2O_Model")
        model = New_CT_L2O_Model(
            A=A,
            lambd=config.LAMBD,
            alpha=config.ALPHA,
            beta=config.BETA,
            delta=config.DELTA,
            K_out_channels=config.K_OUT_CHANNELS,
            max_depth=config.MAX_DEPTH
        ).to(device)

    elif model_type == "ct_l2o":
        print("🔧 Sử dụng CT_L2O_Model")
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
        print("🔧 Sử dụng Scale_CT_L2O_Model")
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
        raise ValueError(f"❌ MODEL_TYPE '{config.MODEL_TYPE}' không hợp lệ!")

    return model


def evaluate_model():
    """Đánh giá model trên tập test"""
    set_seed(config.SEED)
    print("Đang tải dữ liệu test...")
    _, test_loader, A = get_dataloaders(config.BATCH_SIZE)
    print("Dữ liệu test đã sẵn sàng.")

    # Khởi tạo model
    model = build_model(A, config.DEVICE)
    model_save_path = f"{config.MODEL_TYPE}.pth"

    # Kiểm tra checkpoint
    if not os.path.exists(model_save_path):
        print(f"❌ Không tìm thấy checkpoint: {model_save_path}")
        print("Vui lòng chạy 'train_general.py' để huấn luyện trước.")
        return

    # Load checkpoint
    optimizer = torch.optim.Adam(model.parameters())
    model, _, _, _ = load_checkpoint(model, optimizer, model_save_path)
    model.eval()

    print(f"\n--- Đang đánh giá model [{config.MODEL_TYPE}] ---")

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

    print("\n--- Kết quả đánh giá ---")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE:  {avg_mse:.6f}")

    # Hiển thị trực quan
    print("\n--- Trực quan một vài mẫu ---")
    display_model_outputs(model, test_loader, config.DEVICE, num_samples=3)


if __name__ == "__main__":
    evaluate_model()
