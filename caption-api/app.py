# app.py — Flask API with BLIP + Groq LLM for social media captions

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import io
import os
import cv2
import tempfile
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq
from dotenv import load_dotenv

load_dotenv()   # loads GROQ_API_KEY from .env file

app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "paste_your_groq_key_here")
groq_client  = Groq(api_key=GROQ_API_KEY)

# Load BLIP once when server starts
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(DEVICE)
model.eval()
print("BLIP model ready!")


# ---------------------------------------------------------------------------
# Helper — run BLIP on one PIL image
# ---------------------------------------------------------------------------

def caption_single_image(pil_image):
    inputs = processor(pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Helper — rewrite plain description as social media caption using Groq LLM
# ---------------------------------------------------------------------------

def make_social_caption(description, platform="instagram"):
    print(f"[Groq] platform={platform}, description={description}")

    if platform == "instagram":
        prompt = f"""You are a creative Instagram influencer.
I have a photo that shows: "{description}"

Write a catchy Instagram caption for this photo.
- Use 2-3 short lines
- Add emojis throughout
- End with 6-8 relevant hashtags on a new line
- Make it trendy, fun and engaging
- Do NOT include any explanation, just the caption itself"""

    elif platform == "twitter":
        prompt = f"""You are a witty Twitter user.
I have a photo that shows: "{description}"

Write a tweet for this photo.
- Keep it under 250 characters
- Make it punchy and interesting
- Add 1-2 emojis
- Add 2-3 hashtags at the end
- Do NOT include any explanation, just the tweet itself"""

    elif platform == "linkedin":
        prompt = f"""You are a professional LinkedIn content creator.
I have a photo that shows: "{description}"

Write a LinkedIn post for this photo.
- 2-3 professional sentences
- Thoughtful and inspiring tone
- No hashtags, no emojis
- Do NOT include any explanation, just the post itself"""

    else:
        return description

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.9,   # higher = more creative
        )
        result = response.choices[0].message.content.strip()
        print(f"[Groq] result={result[:80]}...")
        return result
    except Exception as e:
        print(f"[Groq] error: {e}")
        return description   # fallback to plain description if Groq fails


# ---------------------------------------------------------------------------
# Image endpoint
# ---------------------------------------------------------------------------

@app.route('/caption', methods=['POST'])
def caption_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        platform    = request.form.get('platform', 'instagram')
        print(f"[API] /caption called, platform={platform}")

        image       = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
        description = caption_single_image(image)
        caption     = make_social_caption(description, platform)

        return jsonify({
            'caption':     caption,
            'description': description
        })
    except Exception as e:
        print(f"[API] error: {e}")
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Video endpoint
# ---------------------------------------------------------------------------

@app.route('/caption_video', methods=['POST'])
def caption_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    try:
        platform   = request.form.get('platform', 'instagram')
        video_file = request.files['video']
        suffix     = os.path.splitext(video_file.filename)[-1] or '.mp4'

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        cap          = cv2.VideoCapture(tmp_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 24
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval     = max(1, int(fps * 2))
        sample_at    = list(range(0, total_frames, interval))[:8]

        raw_captions = []
        for frame_idx in sample_at:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw_captions.append(caption_single_image(pil_image))

        cap.release()
        os.unlink(tmp_path)

        if not raw_captions:
            return jsonify({'error': 'Could not extract frames from video'}), 400

        seen = []
        for c in raw_captions:
            if c not in seen:
                seen.append(c)
        description = ". ".join(seen)

        caption = make_social_caption(description, platform)

        return jsonify({
            'caption':         caption,
            'description':     description,
            'frames_analyzed': len(raw_captions)
        })

    except Exception as e:
        print(f"[API] error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'BLIP + Groq'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
