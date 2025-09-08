# 🚀 Mistral Setup for Talk-to-your-PDF

This version uses **Mistral API** - **FREE** tier with 1M tokens/month!

## ⚡ Super Quick Setup (2 minutes)

### 1. Get Free Mistral API Key
- Go to https://console.mistral.ai
- Sign up (free)
- Create API key
- Copy the key

### 2. Add to Secrets
Create `.streamlit/secrets.toml`:
```toml
MISTRAL_API_KEY = "your_api_key_here"
SUPABASE_POSTGRES_URL = "your_supabase_url"
```

### 3. Install & Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## ✅ What You Get

- **FREE**: 1M tokens/month (very generous!)
- **Fast**: No local setup needed
- **Quality**: Mistral-small for responses
- **Simple**: Just add API key and run

## 🎯 Models Used

- **mistral-embed**: Fast embeddings (1024 dimensions)
- **mistral-small**: High-quality responses

## 🔥 Why Mistral?

- ✅ **1M free tokens/month** (vs OpenAI's limited trial)
- ✅ **No local installation** (vs Ollama complexity)
- ✅ **Production ready** (reliable API)
- ✅ **2-minute setup** (just API key needed)

Perfect for GitHub projects - users just need one API key!