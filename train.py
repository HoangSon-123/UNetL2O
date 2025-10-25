# file: train_general.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time

import config
from dataset import get_dataloaders
from utils import set_seed, save_checkpoint, load_checkpoint, count_parameters

# import các model có thể chọn
from model import *

def train_epoch(loader, model, criterion, optimizer, device):
    """Huấn luyện 1 epoch chung cho mọi model."""
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False, ascii=True)

    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.6e}")

    return epoch_loss / len(loader)


def validate_epoch(loader, model, criterion, device):
    """Đánh giá trên tập test (không cần grad)."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    return val_loss / len(loader)


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




def main():
    set_seed(config.SEED)

    # 1️⃣ Tải dữ liệu
    print("Đang tải dữ liệu...")
    train_loader, test_loader, A = get_dataloaders(config.BATCH_SIZE)
    print("Tải dữ liệu hoàn tất.")

    # 2️⃣ Khởi tạo model
    model = build_model(A, config.DEVICE)

    # 3️⃣ Criterion & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 4️⃣ Load checkpoint (nếu có)
    start_epoch = 0
    best_loss = config.BEST_LOSS_INIT
    if os.path.exists(config.SAVE_PATH):
        model, optimizer, start_epoch, best_loss = load_checkpoint(model, optimizer, config.SAVE_PATH)
        print(f"Tiếp tục huấn luyện từ epoch {start_epoch+1}, best_loss={best_loss:.6f}")

    # 5️⃣ Train loop
    print(f"Bắt đầu huấn luyện mô hình [{config.MODEL_TYPE}] ...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(train_loader, model, criterion, optimizer, config.DEVICE)
        val_loss = validate_epoch(test_loader, model, criterion, config.DEVICE)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e} | Time: {epoch_time:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_loss, config.SAVE_PATH)

    print(f"✅ Huấn luyện hoàn tất. Model tốt nhất lưu tại: {config.SAVE_PATH}")


if __name__ == "__main__":
    main()
