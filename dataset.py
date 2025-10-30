# file: dataset.py
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import config
import numpy as np

def to_tensor(x, device):
    """Convert numpy array or tensor to torch.Tensor on the specified device"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device, dtype=torch.float32)

def load_data(data_path, device):
    """
    Load FFPN data (A, u_true_train, u_true_test, data_obs_train, data_obs_test, M, D)
    from a .pkl or .pt file, convert to Tensor, and move to the correct device.
    """
    print(f"--- Loading data from: {data_path} ---")
    try:
        state = torch.load(data_path, weights_only=False, map_location=device)
    except Exception as e:
        print(f"Error using torch.load: {e}")
        print("-> Trying to load with pickle...")
        with open(data_path, 'rb') as f:
            state = pickle.load(f)

    # Required components
    A = to_tensor(state['A'], device)
    u_train = to_tensor(state['u_true_train'], device)
    u_test = to_tensor(state['u_true_test'], device)
    data_obs_train = to_tensor(state['data_obs_train'], device)
    data_obs_test = to_tensor(state['data_obs_test'], device)

    print(f"Shape of A: {A.shape}")
    print(f"Number of u_train samples: {u_train.shape[0]}")
    print(f"Number of u_test samples: {u_test.shape[0]}")

    return A, u_train, u_test, data_obs_train, data_obs_test


def get_dataloaders(batch_size):
    """
    Create DataLoaders for FFPN (train/test) and return A.
    The mapping is: (data_obs -> u_true)
    """
    A, u_train, u_test, data_obs_train, data_obs_test = \
        load_data(config.DATA_PATH, config.DEVICE)

    # Training dataset
    train_dataset = TensorDataset(data_obs_train, u_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Testing dataset
    test_dataset = TensorDataset(data_obs_test, u_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total train batches: {len(train_loader)} | test batches: {len(test_loader)}")

    # Return the required objects
    return train_loader, test_loader, A
