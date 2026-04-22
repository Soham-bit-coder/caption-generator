# config.py — all settings in one place

import torch

# Where is your data?
IMAGES_DIR    = "../data/flickr8k/Images"
CAPTIONS_FILE = "../data/flickr8k/Images/captions.txt"
CHECKPOINT_DIR = "checkpoints"

# Vocabulary: ignore words that appear less than 5 times
VOCAB_THRESHOLD = 5

# Image size ResNet expects
IMG_SIZE = 224

# Model sizes
EMBED_DIM  = 256   # how big each word vector is
HIDDEN_DIM = 512   # how big the LSTM memory is
NUM_LAYERS = 1     # number of LSTM layers (keep 1 for simplicity)

# Training settings
BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 3e-4
TRAIN_SPLIT   = 0.9   # 90% train, 10% val

# Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
