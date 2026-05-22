# How to Add Instagram Caption Generation (LLM Upgrade)

This guide explains how to upgrade the app so it generates real Instagram-style
captions with emojis and hashtags instead of plain descriptions.

---

## What You Are Adding

Before (current):
```
Input:  photo of two friends at beach
Output: "two people are sitting on a beach at sunset"
```

After (upgraded):
```
Input:  photo of two friends at beach
Output: "Sunsets hit different with the right people 🌅✨
         Grateful for moments like these 🙏
         #sunsetvibes #beachdays #blessed #friends #goldenhour"
```

---

## Step 1 — Get a Free Groq API Key

1. Go to https://console.groq.com
2. Click Sign Up (free, no credit card needed)
3. After login go to API Keys section
4. Click Create API Key
5. Copy the key — it looks like: gsk_xxxxxxxxxxxxxxxxxxxx
6. Save it somewhere safe, you will need it in Step 3

---

## Step 2 — Install Groq Python Library

Open PowerShell and run:

```
pip install groq
```

---

## Step 3 — Update caption-api/app.py

Open `caption-api/app.py` and make these changes:

### 3a. Add import at the top of the file

After the existing imports add this line:

```python
from groq import Groq
```

### 3b. Add your API key and create Groq client

After the line `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` add:

```python
GROQ_API_KEY = "paste_your_groq_key_here"
groq_client  = Groq(api_key=GROQ_API_KEY)
```

### 3c. Add a new function to generate Instagram caption

Add this function anywhere before the routes:

```python
def make_instagram_caption(description, platform="instagram"):
    """
    Takes a plain image description from BLIP and rewrites it
    as a social media caption using Groq LLM.
    """

    if platform == "instagram":
        prompt = f"""You are a social media expert.
Write an engaging Instagram caption for this image: "{description}"
Requirements:
- 2 to 3 lines
- Include relevant emojis
- Add 5 to 8 hashtags at the end
- Sound natural and fun, not robotic
Only return the caption, nothing else."""

    elif platform == "twitter":
        prompt = f"""Write a short punchy Twitter/X post for this image: "{description}"
Requirements:
- Maximum 280 characters
- Include 1 or 2 emojis
- Add 2 or 3 hashtags
Only return the tweet text, nothing else."""

    elif platform == "linkedin":
        prompt = f"""Write a professional LinkedIn post for this image: "{description}"
Requirements:
- 2 to 3 sentences
- Professional and thoughtful tone
- No hashtags or emojis
Only return the post text, nothing else."""

    else:
        return description

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()
```

### 3d. Update the /caption route to use the new function

Find the existing `/caption` route. It currently looks like this:

```python
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
```

Replace it with this:

```python
@app.route('/caption', methods=['POST'])
def caption_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        # Get platform from request, default to instagram
        platform = request.form.get('platform', 'instagram')

        image       = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
        description = caption_single_image(image)          # BLIP describes the image
        caption     = make_instagram_caption(description, platform)  # LLM rewrites it

        return jsonify({
            'caption':     caption,
            'description': description    # also return the raw BLIP description
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## Step 4 — Update caption-app/App.js

### 4a. Add platform state

Find this line in App.js:

```javascript
const [history, setHistory]   = useState([]);
```

Add this line directly below it:

```javascript
const [platform, setPlatform] = useState('instagram');
```

### 4b. Add platform selector UI

Find the buttons row section (the View with Image, Video, Camera buttons).
Add this new section ABOVE the Generate button:

```javascript
{/* Platform selector */}
<View style={{ flexDirection: 'row', gap: 8, marginBottom: 16, width: '100%' }}>
  {['instagram', 'twitter', 'linkedin'].map(p => (
    <TouchableOpacity
      key={p}
      onPress={() => setPlatform(p)}
      style={{
        flex: 1,
        padding: 10,
        borderRadius: 10,
        alignItems: 'center',
        backgroundColor: platform === p ? '#6c63ff' : '#1e1e1e',
        borderWidth: 1,
        borderColor: platform === p ? '#6c63ff' : '#2e2e2e',
      }}
    >
      <Text style={{ color: '#fff', fontSize: 12, fontWeight: '600' }}>
        {p === 'instagram' ? '📸 Insta' : p === 'twitter' ? '🐦 Twitter' : '💼 LinkedIn'}
      </Text>
    </TouchableOpacity>
  ))}
</View>
```

### 4c. Send platform to the API

Find the `generate()` function. Find this line:

```javascript
formData.append(isVideo ? 'video' : 'image', {
```

Add this line BEFORE it:

```javascript
formData.append('platform', platform);
```

---

## Step 5 — Restart the Flask server

Stop the running server with CTRL+C then start it again:

```
python app.py
```

---

## Step 6 — Test it

1. Open the app
2. Select Instagram, Twitter, or LinkedIn tab
3. Pick an image
4. Tap Generate Caption
5. You should get a proper social media caption

---

## How It Works

```
You pick image + select platform
        ↓
App sends image + platform name to Flask API
        ↓
BLIP reads image → plain description
"two people sitting on a beach at sunset"
        ↓
Groq LLM rewrites it for the selected platform
        ↓
Instagram: "Sunsets hit different 🌅 #beachvibes"
Twitter:   "Golden hour with the crew 🌅 #sunset"
LinkedIn:  "Reflecting on the importance of taking time to recharge."
        ↓
App displays the caption
```

---

## Troubleshooting

**Error: groq module not found**
Run `pip install groq` again and make sure you are in the right environment.

**Error: invalid API key**
Double check you pasted the full key from console.groq.com correctly.

**Caption is still plain description**
Make sure you saved app.py after editing and restarted the Flask server.

**App not sending platform**
Make sure you added `formData.append('platform', platform)` before the image append line.
