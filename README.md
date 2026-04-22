# Image & Video Caption Generator

An AI-powered caption generator that uses deep learning to describe images and videos in plain English. Built with PyTorch, BLIP, Flask, and React Native.

---

## Screenshots

> Take screenshots of your running app and place them in `assets/screenshots/` folder.

| Home Screen | Image Caption | Video Caption |
|-------------|---------------|---------------|
| ![Home](assets/screenshots/home.png) | ![Image](assets/screenshots/image_caption.png) | ![Video](assets/screenshots/video_caption.png) |

---

## How It Works

```
Mobile App (React Native)
        ↓  sends image or video
    Flask API (Python)
        ↓  runs BLIP model
    caption text
        ↓  sends back
Mobile App displays caption
```

For images — BLIP reads the image and generates a description directly.

For videos — the API extracts one frame every 2 seconds (up to 8 frames), captions each frame, then combines them into one summary.

---

## Project Structure

```
├── image-captioning/        ← custom trained ResNet + LSTM model
│   ├── config.py            ← all hyperparameters and paths
│   ├── utils.py             ← vocabulary builder (words ↔ numbers)
│   ├── dataset.py           ← Flickr8k data loader
│   ├── model.py             ← EncoderCNN + DecoderLSTM
│   ├── train.py             ← training loop
│   ├── inference.py         ← caption any image from command line
│   └── checkpoints/         ← saved model weights after training
│
├── caption-api/             ← Flask backend using BLIP
│   ├── app.py               ← API with /caption and /caption_video endpoints
│   └── requirements.txt
│
├── caption-app/             ← React Native mobile app
│   ├── App.js               ← main app screen
│   ├── app.json             ← Expo config
│   └── package.json
│
└── data/
    └── flickr8k/
        └── Images/          ← dataset images + captions.txt
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Mobile App | React Native + Expo SDK 54 |
| Backend API | Flask + Flask-CORS |
| AI Model | BLIP (Salesforce) via HuggingFace Transformers |
| Custom Model | ResNet50 (CNN) + LSTM trained on Flickr8k |
| Image Processing | OpenCV, Pillow |
| Deep Learning | PyTorch |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/caption-generator.git
cd caption-generator
```

### 2. Start the Flask API

```bash
cd caption-api
pip install -r requirements.txt
python app.py
```

First run downloads the BLIP model (~1GB). Wait for `BLIP model ready!`

### 3. Find your local IP

```powershell
ipconfig
```

Look for `IPv4 Address` under your WiFi adapter e.g. `192.168.29.11`

### 4. Update IP in the app

Open `caption-app/App.js` line 10:

```js
const API_URL = 'http://YOUR_IP_HERE:5000';
```

### 5. Run the mobile app

```bash
cd caption-app
npm install
npx expo start --clear
```

Scan the QR code with Expo Go on your phone. Make sure your phone and laptop are on the same WiFi.

---

## API Endpoints

### `POST /caption`
Accepts an image file, returns a caption.

```
Request:  multipart/form-data  { image: <file> }
Response: { "caption": "a dog is running on the beach" }
```

### `POST /caption_video`
Accepts a video file, extracts frames, returns a combined caption.

```
Request:  multipart/form-data  { video: <file> }
Response: { "caption": "a dog is running. the dog jumps into water.", "frames_analyzed": 6 }
```

### `GET /health`
Check if the server is running.

```
Response: { "status": "ok", "model": "BLIP" }
```

---

## Training Your Own Model (Optional)

The app uses BLIP by default. If you want to train the custom ResNet+LSTM model on Flickr8k:

### 1. Download Flickr8k dataset

Get it from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place it at:

```
data/flickr8k/Images/   ← all .jpg files
data/flickr8k/Images/captions.txt
```

### 2. Train

```bash
cd image-captioning
pip install -r requirements.txt
python train.py
```

Trains for 20 epochs. Best model saved to `checkpoints/best.pth`.

### 3. Run inference

```bash
python inference.py path/to/image.jpg
```

---

## Model Architecture (Custom)

```
Image (224×224)
      ↓
EncoderCNN — ResNet50 (frozen) → FC(2048→256)
      ↓  256-dim image fingerprint
DecoderLSTM — LSTM(256→512) → FC(512→vocab_size)
      ↓
Caption generated word by word
```

---

## Requirements

- Python 3.9+
- Node.js 18+
- Expo Go app on your phone (SDK 54)
- Phone and laptop on same WiFi network
