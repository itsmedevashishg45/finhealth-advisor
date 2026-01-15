# 💰 FinHealth Advisor - Multi-Agent RAG System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)

An intelligent financial and healthcare advisory platform powered by Multi-Agent RAG (Retrieval-Augmented Generation) system.

## 🚀 Features

- **Multi-Agent Architecture**: 4 specialized agents (Financial, Healthcare, Research, Coordinator)
- **Advanced RAG Pipeline**: Pinecone vector store with hybrid search
- **Document Processing**: Supports PDF, TXT with intelligent chunking
- **Real-time Monitoring**: RAGAS evaluation metrics
- **Production Ready**: FastAPI backend, Streamlit frontend

## 📋 Prerequisites

- Python 3.11+
- OpenAI API Key
- Pinecone Account
- 8GB+ RAM recommended

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/itsmedevashishg45/finhealth-advisor.git
cd finhealth-advisor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## ▶️ Usage

### Run Streamlit App
```bash
streamlit run streamlit_app/app.py
```

### Run FastAPI Server
```bash
uvicorn api.main:app --reload
```

### Process Documents
```bash
python -m data_ingestion.process_documents --directory data/raw
```

## 📊 Architecture

