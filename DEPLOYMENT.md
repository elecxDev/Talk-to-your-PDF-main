# Vercel Deployment Guide

## 🚀 Deploy to Vercel

### 1. Push to GitHub
```bash
git add .
git commit -m "Ready for Vercel deployment"
git push origin main
```

### 2. Connect to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Sign up/login with GitHub
3. Click "New Project"
4. Import your GitHub repository

### 3. Configure Environment Variables
In Vercel dashboard, add these environment variables:

```
MISTRAL_API_KEY = K9pC6Hp0uCfQIwQjV7O5QnMuuF5SOKH4
SUPABASE_POSTGRES_URL = postgresql://postgres:IAMkeshav1.@db.ajhvffcjzewqkugdocnh.supabase.co:5432/postgres
```

### 4. Deploy Settings
- **Framework Preset**: Other
- **Build Command**: `pip install -r requirements.txt`
- **Output Directory**: Leave empty
- **Install Command**: `pip install -r requirements.txt`

### 5. Deploy
Click "Deploy" and wait for build to complete.

## ✅ Environment Variables are Safe
- Environment variables are encrypted on Vercel
- Only accessible to your deployment
- Not visible in public repository
- Secure for production use

## 🔧 Local Development
The app works with both:
- Local: `.streamlit/secrets.toml`
- Production: Vercel environment variables

## 📝 Notes
- First deployment may take 2-3 minutes
- Subsequent deployments are faster
- Auto-deploys on git push to main branch