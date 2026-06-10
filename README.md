# 🤖 NanoGPT — Streamlit App

A beginner-friendly web app that lets you **train a GPT model from scratch** and watch it generate Shakespeare-style text — all in your browser.

---

## 🚀 Live Demo
Deploy for free on Streamlit Community Cloud (see below).

---

## 📁 Project Structure
```
nanogpt-streamlit/
├── app.py           ← Main Streamlit app (UI + training loop)
├── model.py         ← GPT model (Head, MultiHeadAttention, Block, GPTLanguageModel)
├── train_utils.py   ← Dataset builder, batch sampler, loss estimator
├── requirements.txt ← Python dependencies
└── README.md        ← This file
```

---

## 🖥️ Run Locally

```bash
# 1. Clone / download this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```
The app will open at `http://localhost:8501`

---

## ☁️ Deploy Free (Recommended)

### Option 1 — Streamlit Community Cloud ⭐ (easiest, free)
1. Push this folder to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — done! You get a public URL like `https://yourname-nanogpt.streamlit.app`

### Option 2 — Hugging Face Spaces (also free)
1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** as the SDK
3. Upload all files → it auto-deploys

### Option 3 — Render.com (free tier)
1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🎓 How It Works

| Component | What it does |
|-----------|-------------|
| `Head` | Single attention head — token learns from past tokens |
| `MultiHeadAttention` | Multiple heads in parallel for richer patterns |
| `FeedForward` | Small MLP applied per-token after attention |
| `Block` | One Transformer block = attention + FFN + residuals |
| `GPTLanguageModel` | Full model = embeddings + N blocks + output head |

Training uses **AdamW** optimizer and **cross-entropy loss** on next-character prediction.

---

## 🔧 Hyperparameter Guide

| Param | Small (fast) | Balanced | Large (slow) |
|-------|-------------|----------|-------------|
| n_embd | 64 | 128 | 256 |
| n_head | 2 | 4 | 6 |
| n_layer | 2 | 3 | 6 |
| Steps | 200 | 500 | 2000 |
| Context | 32 | 64 | 128 |

Start small to verify it works, then scale up!

---

## 📚 References
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---
**Author:** Aman Kumar | Amazon ML Summer School
