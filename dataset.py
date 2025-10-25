# file: dataset.py
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import config
import numpy as np

def to_tensor(x, device):
    """Chuyển numpy array hoặc tensor sang torch.Tensor trên đúng device"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device, dtype=torch.float32)

def load_data(data_path, device):
    """
    Tải dữ liệu FFPN (A, u_true_train, u_true_test, data_obs_train, data_obs_test, M, D)
    từ file .pkl hoặc .pt, chuyển về Tensor và sang đúng device.
    """
    print(f"--- Đang tải dữ liệu từ: {data_path} ---")
    try:
        state = torch.load(data_path, weights_only=False, map_location=device)
    except Exception as e:
        print(f"Lỗi khi dùng torch.load: {e}")
        print("-> Đang thử tải bằng pickle...")
        with open(data_path, 'rb') as f:
            state = pickle.load(f)

    # Các thành phần bắt buộc
    A = to_tensor(state['A'], device)
    u_train = to_tensor(state['u_true_train'], device)
    u_test = to_tensor(state['u_true_test'], device)
    data_obs_train = to_tensor(state['data_obs_train'], device)
    data_obs_test = to_tensor(state['data_obs_test'], device)

    print(f"Shape của A: {A.shape}")
    print(f"Số lượng u_train: {u_train.shape[0]}")
    print(f"Số lượng u_test: {u_test.shape[0]}")


    return A, u_train, u_test, data_obs_train, data_obs_test


def get_dataloaders(batch_size):
    """
    Tạo DataLoader cho FFPN (train/test) và trả về A.
    Dữ liệu được ánh xạ: (data_obs -> u_true)
    """
    A, u_train, u_test, data_obs_train, data_obs_test = \
        load_data(config.DATA_PATH, config.DEVICE)

    # Dataset huấn luyện
    train_dataset = TensorDataset(data_obs_train, u_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Dataset kiểm thử
    test_dataset = TensorDataset(data_obs_test, u_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Tổng số batch train: {len(train_loader)} | test: {len(test_loader)}")

    # Trả về các đối tượng cần thiết
    return train_loader, test_loader, A
