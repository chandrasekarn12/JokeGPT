# 🤖 JokeGPT – Dad‑Joke Generator

  

Minimal GPT‑style model that spits out painfully wholesome dad jokes.

  

[![HuggingFace Space](https://img.shields.io/badge/Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/nikhchand04/JokeGPT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

  

---

  

## 🌐 Live Demo

  

| Component | URL |

|-----------|-----|

| **JokeGPT Space** | [huggingface.co/spaces/chandrasekarn12/JokeGPT](https://huggingface.co/spaces/nikhchand04/JokeGPT) |

  

Try it out directly in the browser — no installs, no hassle (may be slow because running on free tier).

  

---

  

## 🏃‍♂️ Quick Start (Local)

  

```bash

# 1. Clone & create venv

$ git clone https://github.com/chandrasekarn12/JokeGPT && cd JokeGPT

$ python -m venv .venv && source .venv/bin/activate

$ pip install -r requirements.txt

  

# 2. Prepare data (tokenises dad_jokes.csv)

$ python prepare_tokenizer_data.py

  

# 3. Train the model (fits CPU)

$ python train.py

  

# 4. Generate joke (CLI)

$ python generate.py

```

  

---

  

## 🚀 Deploy on Render (free tier)

  

### 1 FastAPI service

* **Environment → Docker**

* **Dockerfile path:** `fastapi.dockerfile`  

  (or `Dockerfile` if you merged)

* Leave **Build Command** & **Start Command** blank – Render uses the `CMD`.

* Port = `8080` (auto‑detected from `EXPOSE 8080`).

  

### 2 Streamlit UI service

* **Dockerfile path:** `streamlit.dockerfile`  (see `/frontend` folder)

* Add env‑var `BACKEND_URL=https://jokegpt-api.onrender.com` in **Settings → Environment**.

* Port = `8080` (auto).

  

That’s it – Render builds from GitHub and serves two URLs (above).

  

---

  

## 🐳 Docker (local or Cloud Run)

  

```bash

# Build API image

$ docker build -f fastapi.dockerfile -t jokeapi .

$ docker run -p 8000:8080 jokeapi &

  

# Build UI image

$ docker build -f streamlit.dockerfile -t jokeui .

$ docker run -p 8501:8080 -e BACKEND_URL=http://localhost:8000 jokeui

```

  

To push to Google Cloud Run:

```bash

$ docker tag jokeapi us-central1-docker.pkg.dev/<PROJECT>/jokegpt/jokeapi:latest

$ docker push       us-central1-docker.pkg.dev/<PROJECT>/jokegpt/jokeapi:latest

$ gcloud run deploy jokegpt-api --image us-central1-docker.pkg.dev/<PROJECT>/jokegpt/jokeapi:latest --platform managed --region us-central1 --allow-unauthenticated

```

  

---

  

## ⚙️ Project Structure (key files)

  

```

.

JokeGPT/

├── data/ # ✓ Training CSVs + final model weights

│ ├── dad_jokes.csv

|  └── checkpoint.pt

├── src/

│ ├── model.py # GPT-1-like transformer

│ ├── train.py # Training loop

│ ├── generate.py # CLI text generation

| ├── config.py # Model config info

│ └── app.py # Final streamlit UI

├── Old_versions/ # Old code that can be explored and functions (not maintained)

│ |── deployment/ # Deploying containerized version with FastAPI, Streamlit, and Docker

| ├── GPT2/ # Overly complicated GPT-2

| ├── Old_GPT1/ # Overly complicated GPT-1

└── README.md

```

  

## 🧑‍🔬 Training Details

**Param Values**

* Tokens  178 k jokes → 3.7 M tokens
* Model 6 × 192-d GPT-1 (~3.3 M params)
* Hardware  RTX 3060 / 12 GB (fits CPU but slower)
* Time  2 epochs ≈ 15 min GPU / 3 h CPU