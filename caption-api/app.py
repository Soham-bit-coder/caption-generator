# app.py — Flask API with image and video captioning using BLIP

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import io
import os
import cv2
import tempfile
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP once when server starts
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(DEVICE)
model.eval()
print("BLIP model ready!")


def caption_single_image(pil_image):
    """Run BLIP on one PIL image and return caption string."""
    inputs = processor(pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Image endpoint
# ---------------------------------------------------------------------------

@app.route('/caption', methods=['POST'])
def caption_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        image   = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
        caption = caption_single_image(image)
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Video endpoint
# ---------------------------------------------------------------------------

@app.route('/caption_video', methods=['POST'])
def caption_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    try:
        # Save uploaded video to a temp file (OpenCV needs a file path)
        video_file = request.files['video']
        suffix = os.path.splitext(video_file.filename)[-1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        # Open video with OpenCV
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample one frame every 2 seconds (max 8 frames to keep it fast)
        interval   = max(1, int(fps * 2))
        sample_at  = list(range(0, total_frames, interval))[:8]

        captions = []
        for frame_idx in sample_at:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # OpenCV uses BGR, PIL needs RGB
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap_text  = caption_single_image(pil_image)
            captions.append(cap_text)

        cap.release()
        os.unlink(tmp_path)   # delete temp file

        if not captions:
            return jsonify({'error': 'Could not extract frames from video'}), 400

        # Remove duplicate captions and join into one summary
        seen = []
        for c in captions:
            if c not in seen:
                seen.append(c)

        summary = ". ".join(seen) + "."
        return jsonify({'caption': summary, 'frames_analyzed': len(captions)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'BLIP'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
