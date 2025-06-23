# 🤖 JokeGPT  
*A GPT-style Transformer trained to generate dad jokes from Reddit*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red?logo=pytorch)](https://pytorch.org/)
[![Transformer](https://img.shields.io/badge/Model-GPT--mini-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🧠 Overview

**JokeGPT** is a lightweight GPT-style transformer model, inspired by GPT-1, trained from scratch on hundreds of thousands of Reddit dad jokes. The model will learn to understand structure, style, and punchline timing, generating original and often hilariously bad dad jokes with each run.

This project highlights end-to-end ML engineering:
- Dataset cleaning and tokenization
- Training a character-level language model
- Saving checkpoints and visualizing loss curves
- Deploying the model for joke generation

---

## 🎯 Project Objectives

- Build a character-level GPT-style Transformer model trained on real-world humor data (Reddit dad jokes)
- Demonstrate a full ML workflow: data cleaning, tokenization, training, evaluation, and inference
- Provide an interpretable, minimal GPT implementation using PyTorch and no external modeling libraries
- Enable reproducible joke generation with saved checkpoints and a simple generation interface
- Offer a lightweight, extensible framework for experimenting with Transformer architectures on small text datasets

---

## 📦 Technologies Used

- **Python 3.10+**
- **PyTorch** – Transformer implementation
- **NumPy / Pandas** – data handling
- **Matplotlib** – training curve visualization
- **Reddit Dad Jokes Dataset** – collected from Kaggle

---

## 📁 Project Structure

```bash
JokeGPT/
├── config.py           # All model & training hyperparameters
├── prepare_data.py     # Loads, cleans, tokenizes dad jokes into .bin format
├── model.py            # GPTConfig + Transformer blocks (self-attn, FFN, etc.)
├── train.py            # Full training loop + evaluation + checkpointing
├── generate.py         # Loads trained model and generates jokes
├── data/
│   ├── reddit_dadjokes.csv   # Raw joke data (from Kaggle)
│   ├── train.bin              # Encoded training data
│   ├── val.bin                # Encoded validation data
│   ├── meta.pkl               # Vocabulary + encoders
│   ├── checkpoint.pt          # Final trained weights
│   └── loss_curve.png         # Training loss visualization
