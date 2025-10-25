# file: test_models.py
import torch
import torch.nn as nn
import time

import config
from dataset import get_dataloaders
from utils import set_seed, count_parameters

# Import TẤT CẢ các model của bạn từ file model.py
from model import (
    ImplicitL2OModel, # Thêm lại model cơ sở
    CT_L2O_Model,
    New_CT_L2O_Model,
    CT_FFPN_Model,
    CT_UNet_Model,
    CT_TVM_Model,
    Scale_CT_L2O_Model
)

def run_smoke_test():
    """
    Chạy thử một batch qua các model để kiểm tra lỗi.
    """
    print("--- BẮT ĐẦU SMOKE TEST ---")
    set_seed(config.SEED)
    device = config.DEVICE
    
    # 1. Tải 1 batch dữ liệu
    # ==========================
    print("\n[Bước 1] Đang tải 1 batch dữ liệu...")
    try:
        train_loader, _, A = get_dataloaders(batch_size=32) # Chỉ cần batch nhỏ
        S = torch.diag(torch.count_nonzero(A, dim=0) ** -1.0).float().to(device)
        # Lấy 1 batch
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        print(f"Tải dữ liệu OK. Shape Input: {inputs.shape}, Shape Target: {targets.shape}")
        print(f"Ma trận A shape: {A.shape}")
        
    except Exception as e:
        print(f"!!! LỖI khi tải dữ liệu: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Định nghĩa các model cần test
    # ==========================
    print("\n[Bước 2] Định nghĩa các model cơ bản...")
    
    models_to_test = {
        # "CT_UNet_Model (End-to-End)": CT_UNet_Model(in_channels=1, out_channels=1),
        
        # "CT_L2O_Model (Gốc)": CT_L2O_Model(
        #     A=A, K_out_channels=config.K_OUT_CHANNELS
        # ),
        
        # "New_CT_L2O_Model (U-Net L2O)": New_CT_L2O_Model(
        #     A=A, K_out_channels=config.K_OUT_CHANNELS
        # ),
        
        # "CT_TVM_Model (Total Variation)": CT_TVM_Model(A=A.to(device)),
        
        # "Scale_CT_L2O_Model (Scale)": Scale_CT_L2O_Model(
        #     A=A, K_out_channels=config.K_OUT_CHANNELS
        # ),

        "CT_FFPN_Model (FFPN)": CT_FFPN_Model(A,S),
        
    }
    
    
    criterion = nn.MSELoss()
    print(f"\n[Bước 2 Hoàn tất] Tổng cộng {len(models_to_test)} model sẽ được test.")

    # 3. Chạy test từng model
    # ==========================
    print("\n[Bước 3] Bắt đầu test từng model...")
    
    for name, model in models_to_test.items():
        print(f"\n--- Đang test: {name} ---")
        try:
            model = model.to(device)
            model.train() # Đặt ở chế độ train
            
            # Đếm tham số
            print(f"Số tham số: {count_parameters(model):,}")

            # a. Forward pass
            start_time = time.time()
            outputs = model(inputs.view(inputs.size(0), -1).T)
            #outputs = model(inputs)

            fwd_time = time.time() - start_time
            print(f"Forward pass OK (thời gian: {fwd_time:.4f}s). Shape Output: {outputs.shape}")

            # b. Tính Loss
            loss = criterion(outputs, targets)
            print(f"Tính Loss OK. Loss: {loss.item():.6f}")

            # c. Backward pass
            start_time = time.time()
            loss.backward()
            bwd_time = time.time() - start_time
            print(f"Backward pass OK (thời gian: {bwd_time:.4f}s)")
            
            # d. Kiểm tra gradient (tùy chọn)
            grad_check = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            if grad_check:
                print("Kiểm tra gradient: OK (Có gradient)")
            else:
                if count_parameters(model) > 0:
                    print("!!! Cảnh báo: Không tìm thấy gradient!")
                else:
                    print("Kiểm tra gradient: OK (Model không có tham số huấn luyện)")
            
            print(f"✅ TEST THÀNH CÔNG cho model: {name}")

        except Exception as e:
            print(f"❌ LỖI với model {name}:")
            print(e)
            import traceback
            traceback.print_exc() # In ra lỗi chi tiết

    print("\n--- SMOKE TEST HOÀN TẤT ---")

if __name__ == "__main__":
    run_smoke_test()