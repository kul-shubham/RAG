# 📚 RAG Pipeline: My Local PDF Search Assistant

Welcome to my Retrieval-Augmented Generation (RAG) pipeline! I built this project to create a lightweight, robust document search assistant optimized for local execution on standard laptops. My main goal was to bypass heavy local PyTorch embedding models and massive LLMs, so I designed a highly optimized architecture that provides fast, crash-free document querying even on machines with low RAM and CPU capabilities.

## 🏗️ Architecture Stack & "Why I Chose It"
Here are the specific technologies I chose to make this pipeline as efficient as possible:

- **Ingestion & Chunking**: `LangChain` with `PyMuPDFLoader` and `RecursiveCharacterTextSplitter`.
  - *Why?* LangChain is the industry standard for RAG orchestration. `PyMuPDFLoader` is incredibly fast at extracting raw text from PDFs compared to basic loaders, ensuring my ingestion pipeline doesn't bottleneck.

- **Vector Database**: `ChromaDB`.
  - *Why?* Chroma runs locally without needing a separate Docker container for a database server. It stores vectors directly in a local directory (`chroma_db/`), making the project incredibly portable.

- **Embeddings**: `All-MiniLM-L6-v2` via highly optimized **ONNX-runtime** within Chroma.
  - *Why?* I specifically used this to avoid the PyTorch segfaults and memory bloat that frequently occur on Windows. By relying on native ONNX embeddings, I entirely removed the heavy `sentence-transformers` and `torch` libraries, saving massive amounts of disk space and RAM.

- **LLM Engine**: `Groq API` (`llama-3.1-8b-instant`).
  - *Why?* Groq uses specialized LPUs (Language Processing Units) rather than traditional GPUs, providing blazing-fast inference speeds. By routing text generation to the Groq API, this project requires zero local VRAM and runs instantly.
  
- **Frontend UI**: `Streamlit`.
  - *Why?* Streamlit allows for the rapid creation of a sleek, beautiful data application purely in Python without needing to write any HTML/CSS or React frontend code.

## 🚀 Features
- **Zero VRAM Required**: I offloaded the intense text generation entirely to the Groq API.
- **Crash-Free Execution**: Built to run perfectly on standard Windows environments without PyTorch memory leaks.
- **Modular Codebase**: I organized the code cleanly into a `src/` directory, safely separating ingestion, retrieval, and LLM logic.
- **Interactive Web Interface**: I built a streamlined chat interface that displays expandable source documents and similarity scores alongside the LLM's answers.

## 🛠️ Quickstart

### 1. Prerequisites
- Python 3.12+
- The `uv` package manager (I highly recommend it for incredibly fast dependency resolution)
- A [Groq API Key](https://console.groq.com/)

### 2. Configuration
Create a `.env` file in the root directory and add your API key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```
*(Note: You can easily adjust chunk sizes, text overlapping, and the specific Groq model being used by editing the `config.yaml` file.)*

### 3. Usage

**Step A: Ingest your PDFs**
First, place your PDF research papers into the `data/` directory. Then, build the local vector store:
```bash
uv run python main.py --build
```
*(This extracts text from all PDFs, batches them, embeds them locally, and saves everything to `chroma_db/`)*

**Step B: Query via Terminal**
```bash
uv run python main.py --query "What are the core mechanisms proposed in these papers?"
```

**Step C: Launch the Web App**
For the best experience, run the Streamlit user interface I built to chat with the documents intuitively:
```bash
uv run streamlit run app.py
```

## 📂 Project Structure
```text
RAG/
├── .env                # API Keys
├── config.yaml         # Configuration parameters
├── data/               # Place your PDF documents here!
├── chroma_db/          # Automatically generated local vector database
├── app.py              # Streamlit Web User Interface
├── main.py             # CLI Orchestrator (build DB & query)
│
└── src/
    ├── ingest.py       # PDF loaders and text splitters
    ├── llm.py          # Groq LLM integration and prompt logic
    └── vectorstore.py  # Chroma ONNX wrappers and retriever pipeline
```
