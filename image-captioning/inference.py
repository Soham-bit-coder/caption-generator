# inference.py — load trained model and caption any image

import sys
import os
import torch
from PIL import Image

from config import DEVICE, CHECKPOINT_DIR
from dataset import val_transform
from model import CaptioningModel
from utils import Vocabulary


def main():
    vocab_path = os.path.join(CHECKPOINT_DIR, "vocab.pkl")
    ckpt_path  = os.path.join(CHECKPOINT_DIR, "best.pth")

    if not os.path.exists(vocab_path):
        print("No vocab found. Run train.py first.")
        return
    if not os.path.exists(ckpt_path):
        print("No checkpoint found. Run train.py first.")
        return

    # Load vocabulary and model
    vocab = Vocabulary.load(vocab_path)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)

    model = CaptioningModel(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Model loaded. Vocab size: {len(vocab)}")

    # Get image path from command line or ask user
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        path = input("Enter image path: ").strip()
        image_paths = [path]

    for path in image_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        # Load and transform image
        image = Image.open(path).convert("RGB")
        image_tensor = val_transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

        # Generate caption
        caption = model.generate(image_tensor, vocab)
        print(f"\nImage:   {path}")
        print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
