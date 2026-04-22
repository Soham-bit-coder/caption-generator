# dataset.py — loading images and captions for training

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T

from config import IMAGES_DIR, CAPTIONS_FILE, IMG_SIZE, BATCH_SIZE, TRAIN_SPLIT
from utils import Vocabulary


# ---------------------------------------------------------------------------
# How to prepare an image before feeding it to ResNet
# ---------------------------------------------------------------------------

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),       # resize to 224x224
    T.RandomHorizontalFlip(),             # randomly flip for variety
    T.ToTensor(),                         # convert pixels to numbers (0-1)
    T.Normalize(                          # normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Dataset class — pairs one image with one caption
# ---------------------------------------------------------------------------

class Flickr8kDataset(Dataset):
    def __init__(self, dataframe, vocab, transform):
        self.data      = dataframe.reset_index(drop=True)
        self.vocab     = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        img_path = os.path.join(IMAGES_DIR, row["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)   # shape: (3, 224, 224)

        # Encode caption to numbers
        caption = torch.tensor(self.vocab.encode(row["caption"]), dtype=torch.long)

        return image, caption


# ---------------------------------------------------------------------------
# collate_fn — called when building a batch
# Captions have different lengths, so we pad shorter ones with <PAD> index
# ---------------------------------------------------------------------------

def collate_fn(batch, pad_idx):
    images, captions = zip(*batch)
    images   = torch.stack(images)                                    # (B, 3, 224, 224)
    captions = pad_sequence(captions, batch_first=False,              # (max_len, B)
                            padding_value=pad_idx)
    return images, captions


# ---------------------------------------------------------------------------
# load_data — call this from train.py to get data loaders
# ---------------------------------------------------------------------------

def load_data(vocab=None):
    df = pd.read_csv(CAPTIONS_FILE)
    df.columns = df.columns.str.strip().str.lower()

    # Shuffle rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train and validation
    split    = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split]
    val_df   = df.iloc[split:]

    # Build vocabulary from training captions only
    if vocab is None:
        vocab = Vocabulary()
        vocab.build(train_df["caption"].tolist())

    pad_idx = vocab.word2idx["<PAD>"]

    train_dataset = Flickr8kDataset(train_df, vocab, train_transform)
    val_dataset   = Flickr8kDataset(val_df,   vocab, val_transform)

    # wrap collate so it knows the pad index
    def collate(batch):
        return collate_fn(batch, pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, collate_fn=collate)

    return train_loader, val_loader, vocab
