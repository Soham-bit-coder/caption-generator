---
title: Caption Generator API
emoji: 📸
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# Caption Generator API

Flask API for AI-powered image and video captioning.

## Endpoints

- `POST /caption` — generate social media caption from image
- `POST /caption_video` — generate caption from video
- `GET /esp32` — lightweight endpoint for ESP32 devices
- `GET /health` — check server status

## Setup

Set `GROQ_API_KEY` in Space secrets for social media caption generation.
