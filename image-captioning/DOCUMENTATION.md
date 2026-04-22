# Image Caption Generator — Full Documentation

## What does this project do?

You give it a photo. It writes a caption for that photo.

```
Input:  [photo of a dog running on a beach]
Output: "a dog is running on the beach"
```

It learns by studying 8000 images and their captions from the Flickr8k dataset.

---

## Project Files

```
image-captioning/
├── config.py       → all settings (paths, sizes, learning rate)
├── utils.py        → words to numbers converter (vocabulary)
├── dataset.py      → loads images and captions, builds batches
├── model.py        → the neural network (CNN + LSTM)
├── train.py        → the learning loop
└── inference.py    → use the trained model on any image
```

---

## Full Program Flow

### Step 1 — config.py

Every file imports from config.py. It holds all settings. Nothing runs here.

```
IMAGES_DIR    = where your images are
CAPTIONS_FILE = where captions.txt is
EMBED_DIM     = 256   word vector size
HIDDEN_DIM    = 512   LSTM memory size
BATCH_SIZE    = 32    images processed at once
NUM_EPOCHS    = 20    how many full passes through the data
LEARNING_RATE = 3e-4  how fast the model adjusts its weights
```

---

### Step 2 — utils.py builds the Vocabulary

The model cannot read words. It only understands numbers.
So every word gets assigned a unique number called an index.

```
captions.txt contains 40,000 captions like:
  "A dog is running on the beach"
  "Two children playing in the park"
```

The Vocabulary class does this:
1. Reads all captions
2. Counts how often each word appears
3. Keeps only words appearing 5+ times
4. Assigns each word a number

```
Result:
  <PAD>     → 0
  <SOS>     → 1
  <EOS>     → 2
  <UNK>     → 3
  "a"       → 4
  "dog"     → 5
  "running" → 6
  ...
  Total: ~2818 words
```

The 4 special tokens:

| Token  | Meaning          | Why needed                                      |
|--------|------------------|-------------------------------------------------|
| PAD    | Padding          | Fills short captions so all are the same length |
| SOS    | Start of sentence| Tells LSTM to begin generating                  |
| EOS    | End of sentence  | Tells LSTM to stop generating                   |
| UNK    | Unknown word     | Replaces words not in vocabulary                |

encode and decode example:
```
encode("a dog runs") → [1, 4, 5, 6, 2]
                        SOS  a dog runs EOS

decode([1, 4, 5, 6, 2]) → "a dog runs"
```

---

### Step 3 — dataset.py loads images and captions

Pairs every image with its caption. Flickr8k has 5 captions per image,
so each image appears 5 times as separate training samples.

Image preparation:
```
Original photo (any size)
      ↓  Resize to 224x224  (ResNet requires this)
      ↓  Convert pixels to numbers 0 to 1
      ↓  Normalize using ImageNet average values
Output shape: (3, 224, 224)
  3   = RGB channels
  224 = height
  224 = width
```

Caption preparation:
```
"A dog is running" → encode() → [1, 4, 5, 6, 7, 2]
```

Batching — 32 pairs grouped together. Problem: captions have different lengths.
```
Caption 1: [1, 4, 5, 6, 7, 2]           length 6
Caption 2: [1, 8, 9, 2]                  length 4
Caption 3: [1, 4, 12, 15, 6, 7, 11, 2]  length 8
```

Solution: pad shorter ones with 0 (PAD index)
```
Caption 1: [1, 4,  5,  6,  7,  2,  0, 0]
Caption 2: [1, 8,  9,  2,  0,  0,  0, 0]
Caption 3: [1, 4, 12, 15,  6,  7, 11, 2]
```

The loss function ignores PAD tokens so they do not affect learning.

---

### Step 4 — model.py defines the neural network

Two parts: EncoderCNN and DecoderLSTM.

#### EncoderCNN — the eyes

```
Image (3, 224, 224)
      ↓
  ResNet50 — pretrained, frozen
  Detects: edges → shapes → objects → scene
      ↓
  2048 numbers  (ResNet's internal image summary)
      ↓
  FC layer: 2048 → 256
      ↓
  256 numbers  (compact image fingerprint)
```

ResNet50 is pretrained on 1.2 million images. We do not change its weights.
We only train the small FC layer on top that squishes 2048 → 256.

The 256 numbers are a fingerprint of the image.
Two dog photos will have similar fingerprints.
A car photo will have a very different fingerprint.

#### DecoderLSTM — the writer

The LSTM has a memory (hidden state) of 512 numbers.
It reads one word at a time, updates its memory, then predicts the next word.

```
Image fingerprint (256 numbers)
      ↓  init_hidden layer maps 256 → 512
LSTM memory starts as 512 numbers seeded from the image

Step 1:
  Input: SOS token → embedding → 256 numbers
  LSTM reads it → updates memory
  FC layer scores all 2818 words → "a" wins → output: "a"

Step 2:
  Input: "a" → embedding → 256 numbers
  LSTM reads it → updates memory
  FC layer scores all 2818 words → "dog" wins → output: "dog"

Step 3:
  Input: "dog" → embedding → 256 numbers
  LSTM reads it → updates memory
  FC layer scores all 2818 words → "is" wins → output: "is"

... continues until EOS is predicted
```

The LSTM memory carries context forward. By step 3 the memory
"remembers" that we already said "a dog" and uses that to pick "is" next.

---

### Step 5 — train.py runs the learning loop

This is where the model actually learns. Runs for 20 epochs.

One epoch = the model sees every image+caption pair once.

Inside each epoch:

```
For each batch of 32 images + 32 captions:

  1. Feed images to EncoderCNN → get 32 fingerprints (32, 256)

  2. Feed fingerprints + captions to DecoderLSTM
     Input:  [SOS, a,   dog,  is  ]   (all tokens except last)
     Target: [a,   dog, is,   running] (all tokens except SOS)

  3. Model predicts next word at every position

  4. Compare predictions vs real words → calculate loss
     Loss = how wrong the model was (a single number)
     Low loss = good predictions
     High loss = bad predictions

  5. Backpropagation — trace back through the network
     and figure out which weights caused the error

  6. Optimizer adjusts weights slightly to reduce loss

  7. Repeat for next batch
```

After each epoch, validation runs:
```
Validation = run the model on the 10% of data it never trained on
           = check if it learned general patterns or just memorized
```

If train loss falls but val loss rises, the model is overfitting (memorizing).
If both fall together, the model is genuinely learning.

The best model (lowest val loss) is saved to checkpoints/best.pth.

---

### Step 6 — inference.py generates captions

After training, use this to caption any image.

```
1. Load vocab from checkpoints/vocab.pkl
2. Load model weights from checkpoints/best.pth
3. Load your image → resize → normalize → tensor (1, 3, 224, 224)
4. Pass to EncoderCNN → 256 numbers
5. Pass to DecoderLSTM:
     Feed SOS → predict word 1
     Feed word 1 → predict word 2
     Feed word 2 → predict word 3
     ... until EOS
6. decode() converts number list back to words
7. Print caption
```

---

## End to End Summary

```
captions.txt
      ↓
utils.py → build vocabulary (words → numbers)
      ↓
dataset.py → load images + encode captions → batches of 32
      ↓
model.py → EncoderCNN compresses image to 256 numbers
           DecoderLSTM generates words one by one
      ↓
train.py → repeat for 20 epochs:
           predict → measure loss → adjust weights → save best
      ↓
inference.py → load best model → give any image → get caption
```

---

## What is Validation?

```
All 40,000 captions split into:
  90% = Training set   → model learns from these
  10% = Validation set → model is tested on these (never seen during training)
```

After every epoch you see two numbers:
```
Epoch 1/20 | Train Loss: 3.8 | Val Loss: 4.1
Epoch 2/20 | Train Loss: 3.2 | Val Loss: 3.6
Epoch 3/20 | Train Loss: 2.9 | Val Loss: 3.3
```

Both going down = model is learning well.
Train loss down but val loss going up = model is memorizing, not learning.

Think of it like school:
- Training = studying the textbook
- Validation = practice test with questions you have not seen before
