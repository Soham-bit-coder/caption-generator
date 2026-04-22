# Image Caption Generator — ResNet50 + LSTM

PyTorch implementation of an image captioning model using a ResNet50 CNN encoder and LSTM decoder, trained on the Flickr8k dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

1. Download Flickr8k from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. Extract so the structure looks like:

```
data/flickr8k/
├── Images/          # ~8000 .jpg files
└── captions.txt     # image,caption CSV
```

`captions.txt` should have a header row `image,caption` with 5 captions per image.

## Train

```bash
python train.py
```

Checkpoints are saved to `checkpoints/`. Best model saved as `checkpoints/best.pth`.

## Inference

```bash
# Single or multiple images
python inference.py path/to/image.jpg

# Interactive mode
python inference.py
```

## Architecture

```
Image (224×224)
    ↓
ResNet50 (frozen backbone) → FC(2048→256) → BN
    ↓ features (256-dim)
LSTM Decoder (h0 initialized from image features)
    ↓
Linear → Vocab logits
```

- Encoder backbone is frozen; only the projection head and decoder are trained
- Teacher forcing during training
- Greedy decoding at inference

## Config

All hyperparameters live in `config.py` — adjust `EMBED_DIM`, `HIDDEN_DIM`, `NUM_EPOCHS`, etc. as needed.
