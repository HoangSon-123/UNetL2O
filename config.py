# file: config.py
import torch

# General configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
DATA_PATH = 'FFPN-Lodopab-TrainingData-0.015IndividualNoise.pkl'  # Path to dataset file
SAVE_PATH = 'models/l2o_model_unet.pth'  # Path to save the best model

# DataLoader hyperparameters
BATCH_SIZE = 32

# Model hyperparameters (New_CT_L2O_Model)
LAMBD = 0.1402
ALPHA = 0.1222
BETA = 0.1386
DELTA = -6.1
K_OUT_CHANNELS = 16
MAX_DEPTH = 200

# Training hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 1
BEST_LOSS_INIT = 100.0  # Initial loss value

MODEL_TYPE = "FFPN"  # Options: "FFPN", "TVM", or "UNet"
