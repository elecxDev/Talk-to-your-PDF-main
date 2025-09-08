# Talk to your PDF

A modern AI-powered PDF question-answering application built with Streamlit and Mistral AI.

## Features

- **PDF Upload & Processing**: Extract text and generate embeddings from PDF documents
- **AI-Powered Q&A**: Ask questions about your PDF content and get intelligent responses
- **Modern UI**: Clean, responsive interface with gradient designs and smooth animations
- **Free to Use**: Powered by Mistral AI's generous free tier (1M tokens/month)
- **Vector Search**: Efficient similarity search using PostgreSQL with pgvector

## Quick Start

### 1. Get Mistral API Key
- Visit [console.mistral.ai](https://console.mistral.ai)
- Sign up for free account
- Create an API key

### 2. Set up Database
- Create a Supabase account at [supabase.com](https://supabase.com)
- Create a new project
- Get your PostgreSQL connection URL

### 3. Configure Secrets
Create `.streamlit/secrets.toml`:
```toml
MISTRAL_API_KEY = "your_mistral_api_key_here"
SUPABASE_POSTGRES_URL = "your_supabase_postgres_url_here"
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the App
```bash
streamlit run streamlit_app.py
```

## How It Works

1. **Upload PDF**: Choose any PDF document to analyze
2. **Processing**: Text is extracted and converted to vector embeddings using Mistral's embedding model
3. **Storage**: Embeddings are stored in PostgreSQL with pgvector for efficient similarity search
4. **Ask Questions**: Type questions about the PDF content
5. **AI Response**: Get comprehensive answers powered by Mistral AI

## Technology Stack

- **Frontend**: Streamlit with custom CSS
- **AI Models**: Mistral AI (mistral-embed, mistral-small)
- **Database**: PostgreSQL with pgvector extension
- **PDF Processing**: pdfminer.six
- **Vector Search**: Cosine similarity with pgvector

## Project Structure

```
├── streamlit_app.py          # Main application
├── requirements.txt          # Python dependencies
├── create_table.py          # Database setup script
├── .env.example             # Environment variables template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## API Costs

- **Mistral AI**: Free tier includes 1M tokens/month
- **Supabase**: Free tier includes 500MB database storage
- **Total**: Completely free for moderate usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use this project for personal or commercial purposes.