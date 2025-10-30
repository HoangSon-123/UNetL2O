# file: test_models.py
import torch
import torch.nn as nn
import time

import config
from dataset import get_dataloaders
from utils import set_seed, count_parameters

# Import all models from model.py
from model import (
    ImplicitL2OModel,  # Base model
    CT_L2O_Model,
    New_CT_L2O_Model,
    CT_FFPN_Model,
    CT_UNet_Model,
    CT_TVM_Model,
    Scale_CT_L2O_Model
)


def run_smoke_test():
    """
    Run a smoke test: pass one batch through each model to check for runtime errors.
    """
    print("--- STARTING SMOKE TEST ---")
    set_seed(config.SEED)
    device = config.DEVICE

    # 1. Load a single batch of data
    # ==============================
    print("\n[Step 1] Loading one batch of data...")
    try:
        train_loader, _, A = get_dataloaders(batch_size=32)
        S = torch.diag(torch.count_nonzero(A, dim=0) ** -1.0).float().to(device)

        # Get one batch
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(device), targets.to(device)

        print(f"Data loaded successfully. Input shape: {inputs.shape}, Target shape: {targets.shape}")
        print(f"Matrix A shape: {A.shape}")

    except Exception as e:
        print(f"Error while loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Define models to test
    # ==============================
    print("\n[Step 2] Defining base models...")

    models_to_test = {
        # "CT_UNet_Model (End-to-End)": CT_UNet_Model(in_channels=1, out_channels=1),

        # "CT_L2O_Model (Original)": CT_L2O_Model(
        #     A=A, K_out_channels=config.K_OUT_CHANNELS
        # ),

        # "UNetL2O (U-Net L2O)": UNetL2O(
        #     A=A, K_out_channels=config.K_OUT_CHANNELS
        # ),

        # "CT_TVM_Model (Total Variation)": CT_TVM_Model(A=A.to(device)),

        # "Scale_CT_L2O_Model (Scaled)": Scale_CT_L2O_Model(
        #     A=A, K_out_channels=config.K_OUT_CHANNELS
        # ),

        "CT_FFPN_Model (FFPN)": CT_FFPN_Model(A, S),
    }

    criterion = nn.MSELoss()
    print(f"\n[Step 2 Completed] Total {len(models_to_test)} model(s) will be tested.")

    # 3. Run the test for each model
    # ==============================
    print("\n[Step 3] Starting model tests...")

    for name, model in models_to_test.items():
        print(f"\n--- Testing model: {name} ---")
        try:
            model = model.to(device)
            model.train()  # Set to training mode

            # Count parameters
            print(f"Number of parameters: {count_parameters(model):,}")

            # a. Forward pass
            start_time = time.time()
            outputs = model(inputs.view(inputs.size(0), -1).T)
            # outputs = model(inputs)
            fwd_time = time.time() - start_time
            print(f"Forward pass successful (time: {fwd_time:.4f}s). Output shape: {outputs.shape}")

            # b. Compute loss
            loss = criterion(outputs, targets)
            print(f"Loss computed successfully. Loss: {loss.item():.6f}")

            # c. Backward pass
            start_time = time.time()
            loss.backward()
            bwd_time = time.time() - start_time
            print(f"Backward pass successful (time: {bwd_time:.4f}s)")

            # d. Gradient check (optional)
            grad_check = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            if grad_check:
                print("Gradient check: OK (gradients detected)")
            else:
                if count_parameters(model) > 0:
                    print("Warning: No gradients found!")
                else:
                    print("Gradient check: OK (model has no trainable parameters)")

            print(f"Test successful for model: {name}")

        except Exception as e:
            print(f"Error while testing model {name}:")
            print(e)
            import traceback
            traceback.print_exc()  # Print detailed error traceback

    print("\n--- SMOKE TEST COMPLETED ---")


if __name__ == "__main__":
    run_smoke_test()
