# file: config.py
import torch

# Cấu hình chung
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
DATA_PATH = 'FFPN-Lodopab-TrainingData-0.015IndividualNoise.pkl' # Đường dẫn đến file dữ liệu
SAVE_PATH = 'models\l2o_model_unet.pth' # Nơi lưu model tốt nhất

# Siêu tham số DataLoader
BATCH_SIZE = 32

# Siêu tham số Model (New_CT_L2O_Model)
LAMBD = 0.1402
ALPHA = 0.1222
BETA = 0.1386
DELTA = -6.1
K_OUT_CHANNELS = 16
MAX_DEPTH = 200

# Siêu tham số Huấn luyện
LEARNING_RATE = 0.001
NUM_EPOCHS = 1
BEST_LOSS_INIT = 100.0 # Giá trị loss khởi tạo

MODEL_TYPE = "FFPN"  # hoặc "TVM", "UNet"