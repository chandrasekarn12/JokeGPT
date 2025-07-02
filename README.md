# 🤖 JokeGPT – Dad‑Joke Generator

Minimal GPT‑style model that spits out painfully wholesome dad jokes.

[![Render](https://img.shields.io/badge/Live%20API-Render-green)](https://jokegpt-ui.onrender.com/health)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌐 Live Demo

| Component | URL |
|-----------|-----|
| **API**   | https://jokegpt-api.onrender.com |
| **UI**    | https://jokegpt-ui.onrender.com |

Clone → run locally → or click the links above to try instantly.

---

## 🏃‍♂️ Quick Start (Local)

```bash
# 1. Clone & create venv
$ git clone https://github.com/<your‑handle>/JokeGPT && cd JokeGPT
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# 2. Prepare data (tokenises reddit_dadjokes.csv)
$ python prepare_tokenizer_data.py

# 3. Train a tiny model (fits CPU)
$ python train.py           # creates data/checkpoint.pt

# 4. Generate joke
$ python generate.py        # interactive prompt

# 5. Run API + UI
$ uvicorn app:app --port 8000 &               # terminal 1   (API)
$ BACKEND_URL=http://localhost:8000 \
      streamlit run demo.py --server.fileWatcherType none   # terminal 2 (UI)
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
* **Dockerfile path:** `streamlit.dockerfile`  (see `/frontend` folder)
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
$ docker push       us-central1-docker.pkg.dev/<PROJECT>/jokegpt/jokeapi:latest
$ gcloud run deploy jokegpt-api --image us-central1-docker.pkg.dev/<PROJECT>/jokegpt/jokeapi:latest --platform managed --region us-central1 --allow-unauthenticated
```

---

## ⚙️ Project Structure (key files)

```
.
├── app.py                 # FastAPI server
├── demo.py                # Streamlit UI (calls API)
├── modelGPT2.py           # Tiny GPT‑2‑style model
├── train.py               # Training loop
├── generate.py            # Text generation script
├── fastapi.dockerfile     # Container for API
├── streamlit.dockerfile   # Container for UI
└── data/ checkpoint.pt …  # Model + tokeniser cache
```

---

## 🛣️ Roadmap
1. **Quantisation (bitsandbytes)** – cut latency & memory.
2. **Weights‑and‑Biases logging** – experiment tracking.
3. **GitHub Actions CI/CD** – auto‑publish new images.
4. **Better dataset filtering** – knock out low‑quality jokes.
