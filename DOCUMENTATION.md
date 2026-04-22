# Full Technical Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [File by File Explanation](#3-file-by-file-explanation)
4. [Data Flow](#4-data-flow)
5. [Custom Model Deep Dive](#5-custom-model-deep-dive)
6. [API Reference](#6-api-reference)
7. [Mobile App](#7-mobile-app)
8. [How to Take Screenshots for GitHub](#8-how-to-take-screenshots-for-github)

---

## 1. Project Overview

This project has two parts:

**Part A — Custom trained model** (`image-captioning/`)
A ResNet50 CNN encoder + LSTM decoder trained from scratch on the Flickr8k dataset. This teaches you how image captioning works at a fundamental level.

**Part B — Production quality app** (`caption-api/` + `caption-app/`)
A Flask API using the BLIP model (trained on 129 million images) connected to a React Native mobile app. This is what you actually use day to day.

---

## 2. Architecture

### Overall System

```
┌─────────────────────────────────┐
│        React Native App         │
│  - Pick image or video          │
│  - Show caption + history       │
└────────────┬────────────────────┘
             │ HTTP POST (multipart)
             ▼
┌─────────────────────────────────┐
│         Flask API               │
│  POST /caption                  │
│  POST /caption_video            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│         BLIP Model              │
│  Salesforce/blip-image-         │
│  captioning-base                │
│  Trained on 129M image pairs    │
└─────────────────────────────────┘
```

### Video Processing Pipeline

```
Video file uploaded
      ↓
OpenCV opens video
      ↓
Extract 1 frame every 2 seconds (max 8 frames)
      ↓
Run BLIP on each frame → caption per frame
      ↓
Remove duplicate captions
      ↓
Join into one summary sentence
      ↓
Return to app
```

---

## 3. File by File Explanation

### `config.py`

Central settings file. Every other file imports from here. Nothing runs in this file — it just defines constants.

```python
IMAGES_DIR    = "../data/flickr8k/Images"   # where images are
CAPTIONS_FILE = "../data/flickr8k/Images/captions.txt"
EMBED_DIM     = 256    # size of word vectors
HIDDEN_DIM    = 512    # size of LSTM memory
BATCH_SIZE    = 32     # images processed at once
NUM_EPOCHS    = 20     # training passes through full dataset
LEARNING_RATE = 3e-4   # how fast model adjusts weights
```

---

### `utils.py`

Handles vocabulary — converting words to numbers and back.

The model cannot read words. Every word gets a unique integer index.

```
"dog"     → 5
"running" → 6
"beach"   → 7
```

Key methods:

`build(captions)` — scans all captions, counts word frequency, keeps words appearing 5+ times

`encode("a dog runs")` → `[1, 4, 5, 6, 2]`
- 1 = SOS (start)
- 4, 5, 6 = word indices
- 2 = EOS (end)

`decode([1, 4, 5, 6, 2])` → `"a dog runs"`
- skips SOS, PAD, UNK tokens
- stops at EOS

Four special tokens always added first:

| Token | Index | Purpose |
|-------|-------|---------|
| PAD | 0 | Fills short captions to match batch length |
| SOS | 1 | Signals LSTM to start generating |
| EOS | 2 | Signals LSTM to stop generating |
| UNK | 3 | Replaces words not in vocabulary |

---

### `dataset.py`

Loads images and captions and pairs them for training.

Flickr8k has 8000 images with 5 captions each = 40,000 training samples.

Image transform pipeline:
```
Raw photo (any size)
      ↓ Resize to 224×224
      ↓ Convert pixels to 0-1 range
      ↓ Normalize with ImageNet mean/std
Output: tensor of shape (3, 224, 224)
```

Why 224? ResNet50 was trained on 224×224 images and expects this size.

Why normalize? Keeps pixel values in a stable range so training converges faster.

Batching problem — captions have different lengths:
```
Caption 1: [1, 4, 5, 6, 2]           length 5
Caption 2: [1, 8, 9, 10, 11, 12, 2]  length 7
```

Solution — pad with zeros to match longest in batch:
```
Caption 1: [1, 4, 5,  6,  2,  0, 0]
Caption 2: [1, 8, 9, 10, 11, 12, 2]
```

Loss function ignores index 0 (PAD) so padding doesn't affect learning.

---

### `model.py`

The neural network. Two components:

#### EncoderCNN

```python
self.resnet = nn.Sequential(*list(resnet.children())[:-1])
self.fc     = nn.Linear(2048, EMBED_DIM)
```

ResNet50 is a 50-layer convolutional neural network pretrained on ImageNet.
We remove its final classification layer and add our own projection layer.

```
Input image (3, 224, 224)
      ↓ ResNet50 — detects edges, shapes, objects
      ↓ Output: (2048, 1, 1)
      ↓ Squeeze to (2048,)
      ↓ FC layer: 2048 → 256
Output: 256 numbers — image fingerprint
```

The backbone is frozen — we don't change ResNet's weights. We only train the FC layer on top.

#### DecoderLSTM

```python
self.embedding   = nn.Embedding(vocab_size, EMBED_DIM)
self.lstm        = nn.LSTM(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)
self.init_hidden = nn.Linear(EMBED_DIM, HIDDEN_DIM)
self.fc          = nn.Linear(HIDDEN_DIM, vocab_size)
```

The LSTM generates words one at a time. At each step:
1. Current word → embedding → 256 numbers
2. LSTM reads embedding, updates its 512-number memory
3. FC layer converts memory to scores for all vocab words
4. Highest scoring word = prediction

The image fingerprint (256 numbers) seeds the LSTM's initial memory via `init_hidden`.

#### Training vs Inference

Training uses teacher forcing:
```
Input:  [SOS, "a",   "dog",  "is"  ]
Target: ["a", "dog", "is",   "running"]
Model always sees the correct previous word
```

Inference uses autoregressive decoding:
```
SOS → predict "a" → predict "dog" → predict "is" → predict "running" → EOS
Each prediction feeds into the next step
```

---

### `train.py`

The learning loop. Runs for NUM_EPOCHS iterations over the full dataset.

One training step:
```
1. Forward pass:  model(images, captions) → predictions
2. Loss:          CrossEntropyLoss(predictions, targets)
3. Backward pass: loss.backward() — compute gradients
4. Update:        optimizer.step() — adjust weights
```

After each epoch, validation runs on the 10% held-out data.
Best model (lowest validation loss) is saved to `checkpoints/best.pth`.

Loss interpretation:
```
Epoch  1 → Val Loss: 3.52  (model barely knows anything)
Epoch  5 → Val Loss: 3.10  (learning basic patterns)
Epoch 10 → Val Loss: 2.80  (getting better)
Epoch 20 → Val Loss: 2.65  (reasonably trained)
```

Lower loss = better predictions. Loss below 2.5 produces readable captions.

---

### `inference.py`

Loads the saved model and generates captions for any image.

```
1. Load vocab.pkl  → vocabulary mapping
2. Load best.pth   → trained model weights
3. Load image      → resize, normalize, tensor
4. model.generate() → word by word until EOS
5. vocab.decode()  → numbers back to words
6. Print caption
```

Usage:
```bash
python inference.py path/to/image.jpg
```

---

### `caption-api/app.py`

Flask server with two endpoints.

`POST /caption` — image captioning:
```
1. Receive image file
2. Open with Pillow
3. Run BLIP processor + model
4. Return caption string
```

`POST /caption_video` — video captioning:
```
1. Receive video file
2. Save to temp file
3. OpenCV opens video, reads FPS and total frames
4. Sample frames at 2-second intervals (max 8)
5. Run BLIP on each frame
6. Remove duplicate captions
7. Join into summary
8. Delete temp file
9. Return summary + frame count
```

---

### `caption-app/App.js`

React Native app with three main functions:

`pickImage()` — opens gallery, filters to images only
`pickVideo()` — opens gallery, filters to videos, max 60 seconds
`takePhoto()` — opens camera

`generate()` — sends selected media to API:
- image → POST /caption
- video → POST /caption_video

State managed:
```javascript
media   — { uri, type: 'image'|'video' }
caption — generated caption string
frames  — number of video frames analyzed
loading — boolean for spinner
history — last 10 captioned items
```

---

## 4. Data Flow

### Image Caption Request

```
User taps "Generate Caption"
      ↓
App builds FormData with image file
      ↓
POST http://192.168.x.x:5000/caption
      ↓
Flask receives file
      ↓
Pillow opens image → RGB
      ↓
BLIP processor tokenizes image
      ↓
BLIP model.generate() → token ids
      ↓
processor.decode() → caption string
      ↓
JSON response { caption: "..." }
      ↓
App displays caption
```

### Video Caption Request

```
User selects video
      ↓
POST http://192.168.x.x:5000/caption_video
      ↓
Flask saves to temp file
      ↓
OpenCV reads video metadata (fps, frame count)
      ↓
Loop: seek to frame → read → convert BGR→RGB → PIL image → BLIP caption
      ↓
Deduplicate captions
      ↓
Join with ". " separator
      ↓
Delete temp file
      ↓
JSON response { caption: "...", frames_analyzed: 6 }
      ↓
App displays caption + frame count
```

---

## 5. Custom Model Deep Dive

### Why ResNet50?

ResNet50 uses residual connections — shortcuts that let gradients flow through 50 layers without vanishing. This makes it much easier to train deep networks. It achieves 76% accuracy on ImageNet with relatively fast inference.

### Why LSTM?

LSTM (Long Short-Term Memory) solves the vanishing gradient problem in RNNs. It has three gates:

- Forget gate — decides what to remove from memory
- Input gate — decides what new info to store
- Output gate — decides what to output

This lets it remember context across long sequences — important for generating grammatically correct sentences.

### Loss Function

CrossEntropyLoss measures how wrong the model's word predictions are.

```
Correct word: "dog" (index 5)
Model scores: [0.1, 0.2, 0.05, 0.3, 0.8, 0.9, 0.1, ...]
                                              ↑ index 5 = 0.9 (high = good)
Loss = -log(0.9) = 0.10  (low loss, good prediction)

If model scored "dog" at 0.1:
Loss = -log(0.1) = 2.30  (high loss, bad prediction)
```

### Why BLIP is Better

| | Custom Model | BLIP |
|--|--|--|
| Training data | 8,000 images | 129 million images |
| Training time | 20 hours CPU | weeks on TPUs |
| Architecture | ResNet50 + LSTM | ViT + BERT-style transformer |
| Caption quality | basic, generic | detailed, accurate |

---

## 6. API Reference

### POST /caption

Request:
```
Content-Type: multipart/form-data
Body: image=<jpg/png file>
```

Response:
```json
{ "caption": "a dog is running on the beach" }
```

Error:
```json
{ "error": "No image provided" }
```

### POST /caption_video

Request:
```
Content-Type: multipart/form-data
Body: video=<mp4/mov file>
```

Response:
```json
{
  "caption": "a dog is running on the beach. the dog jumps into the water.",
  "frames_analyzed": 6
}
```

### GET /health

Response:
```json
{ "status": "ok", "model": "BLIP" }
```

---

## 7. Mobile App

### Screens

Home screen shows:
- Image/video preview box
- Three buttons: Image, Video, Camera
- Generate Caption button
- Caption result box (purple border)
- History of last 10 captions

### State Flow

```
App starts → no media selected → Generate button disabled
      ↓
User picks image/video
      ↓
Preview shows in box, Generate button enabled
      ↓
User taps Generate
      ↓
Loading spinner shows
      ↓
API responds
      ↓
Caption displays, item added to history
```

---

## 8. How to Take Screenshots for GitHub

### On Android (Expo Go)

1. Run the app on your phone
2. Press `Volume Down + Power` simultaneously
3. Screenshot saves to your gallery

### Transfer to laptop

Connect phone via USB or use Google Photos / WhatsApp to send to yourself.

### Save in the right folder

Create this folder structure:
```
assets/
└── screenshots/
    ├── home.png
    ├── image_caption.png
    └── video_caption.png
```

### They will show in README automatically

The README.md already references these paths:
```markdown
![Home](assets/screenshots/home.png)
![Image](assets/screenshots/image_caption.png)
![Video](assets/screenshots/video_caption.png)
```

Once you push to GitHub, the screenshots appear in the README automatically.

### Recommended screenshots to take

1. Home screen with no media selected
2. An image loaded in the preview box
3. A caption result displayed below an image
4. A video loaded with the 🎬 badge visible
5. A video caption result showing "X frames analysed"
6. The history section with a few items
