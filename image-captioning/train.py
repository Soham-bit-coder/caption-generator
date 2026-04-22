# train.py — trains the model and saves checkpoints

import os
import torch
import torch.nn as nn
from tqdm import tqdm

from config import DEVICE, NUM_EPOCHS, LEARNING_RATE, CHECKPOINT_DIR
from dataset import load_data
from model import CaptioningModel


def train_one_epoch(model, loader, optimizer, criterion, pad_idx):
    model.train()
    model.encoder.resnet.eval()   # keep ResNet in eval mode (it's frozen)

    total_loss = 0

    for images, captions in tqdm(loader, desc="Training", ncols=80):
        images   = images.to(DEVICE)
        captions = captions.to(DEVICE)   # shape: (seq_len, batch)

        # Input to decoder: all tokens except the last one
        # Target: all tokens except the first one (<SOS>)
        # Example caption: [SOS, a, dog, runs, EOS]
        # Input:           [SOS, a, dog, runs]   ← what the model sees
        # Target:          [a, dog, runs, EOS]   ← what the model should predict
        inputs  = captions[:-1]   # (seq_len-1, batch)
        targets = captions[1:]    # (seq_len-1, batch)

        # Forward pass
        outputs = model(images, inputs)   # (batch, seq_len-1, vocab_size)

        # Reshape for loss calculation
        batch, seq_len, vocab_size = outputs.shape
        loss = criterion(
            outputs.reshape(batch * seq_len, vocab_size),  # (batch*seq_len, vocab_size)
            targets.transpose(0, 1).reshape(-1)            # (batch*seq_len,)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0

    for images, captions in tqdm(loader, desc="Validating", ncols=80):
        images   = images.to(DEVICE)
        captions = captions.to(DEVICE)

        inputs  = captions[:-1]
        targets = captions[1:]

        outputs = model(images, inputs)
        batch, seq_len, vocab_size = outputs.shape
        loss = criterion(
            outputs.reshape(batch * seq_len, vocab_size),
            targets.transpose(0, 1).reshape(-1)
        )
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    print(f"Using device: {DEVICE}")

    # Load data and build vocabulary
    train_loader, val_loader, vocab = load_data()

    # Save vocabulary so inference.py can use it later
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    vocab.save(os.path.join(CHECKPOINT_DIR, "vocab.pkl"))

    pad_idx = vocab.word2idx["<PAD>"]

    # Create model
    model = CaptioningModel(vocab_size=len(vocab)).to(DEVICE)

    # Only train decoder + encoder's FC layer (ResNet backbone is frozen)
    params = (list(model.decoder.parameters()) +
              list(model.encoder.fc.parameters()))
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    # CrossEntropyLoss — ignore padding tokens in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, pad_idx)
        val_loss   = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size":  len(vocab),
            }, os.path.join(CHECKPOINT_DIR, "best.pth"))
            print(f"  → Best model saved!")


if __name__ == "__main__":
    main()
