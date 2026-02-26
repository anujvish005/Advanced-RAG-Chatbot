# 🤖 Advanced RAG Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=for-the-badge&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://advanced-rag-chatbot.streamlit.app/)

**A production-grade Retrieval-Augmented Generation (RAG) chatbot that lets you upload any document and have an intelligent conversation with it — powered by Groq's ultra-fast LLM inference, Pinecone vector database, and HuggingFace embeddings.**

[🚀 Live Demo](https://advanced-rag-chatbot.streamlit.app/) · [📝 Report Bug](https://github.com/anujvish005/advanced-rag-chatbot/issues) · [💡 Request Feature](https://github.com/anujvish005/advanced-rag-chatbot/issues)

</div>

---

## 📌 Table of Contents

- [What is RAG?](#-what-is-rag)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Advanced RAG Techniques](#-advanced-rag-techniques)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [How to Use](#-how-to-use)
- [Deployment](#-deployment-to-streamlit-cloud)
- [Author](#-author)

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI technique that combines two powerful capabilities:

1. **Retrieval** — Searching through your own documents to find the most relevant information
2. **Generation** — Using a Large Language Model (LLM) to generate accurate, context-aware answers

Unlike a standard chatbot that relies on pre-trained knowledge, a RAG chatbot **grounds every answer in your actual documents**. This means:

- ✅ Answers are always based on your uploaded content
- ✅ No hallucinations or made-up facts
- ✅ Every answer cites the exact source and page number
- ✅ Works with any domain — legal, medical, finance, research, HR, and more

**Advanced RAG** goes further by adding techniques like query rewriting, MMR retrieval, conversation memory, and semantic chunking to significantly improve answer quality.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Multi-format Document Support** | Upload PDF, TXT, and DOCX files |
| ✂️ **Smart Recursive Chunking** | Splits documents intelligently with configurable size and overlap to preserve context |
| 🔢 **HuggingFace Embeddings** | Converts text into semantic vectors using sentence-transformers |
| 🗄️ **Pinecone Vector Store** | Stores and searches embeddings in a serverless cloud vector database |
| 🔍 **Dual Retrieval Strategies** | Choose between Semantic Search (accuracy) or MMR — Maximum Marginal Relevance (diversity) |
| 🔄 **Query Rewriting** | Automatically reformulates follow-up questions using conversation history for better retrieval |
| 🤖 **Groq LLM Integration** | Ultra-fast inference using Llama 3.3 70B, Llama 3.1 8B, Mixtral, and Gemma models |
| 💾 **Conversation Memory** | Maintains multi-turn chat history with a sliding window for context-aware follow-ups |
| 📚 **Source Attribution** | Every answer shows exactly which document, page, and chunk was used |
| 🎨 **Beautiful Streamlit UI** | Clean, responsive interface with custom styling, metrics, and chat layout |
| ☁️ **Streamlit Cloud Ready** | One-click deployment to a shareable public URL |

---

## 🏗️ Architecture

The following diagram shows how a user query flows through the entire RAG pipeline:

```
┌─────────────────────────────────────────────────────────┐
│                    USER QUERY                           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│          STEP 1: QUERY REWRITING                        │
│  Uses Groq Llama 3.1 8B + Chat History                  │
│  Converts follow-up questions to standalone queries     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│          STEP 2: EMBEDDING                              │
│  HuggingFace all-MiniLM-L6-v2                           │
│  Converts query text to 384-dim semantic vector         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│          STEP 3: VECTOR RETRIEVAL (Pinecone)            │
│  Semantic Search (cosine similarity)                    │
│  MMR - Max Marginal Relevance (relevance + diversity)   │
│  Returns Top-K most relevant document chunks            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│          STEP 4: CONTEXT ASSEMBLY                       │
│  Combines retrieved chunks with source metadata         │
│  Adds conversation history (last 6 messages)            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│          STEP 5: GENERATION (Groq LLM)                  │
│  Llama 3.3 70B reads context + question                 │
│  Generates grounded answer with chunk citations         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│             ANSWER + SOURCES displayed to USER          │
└─────────────────────────────────────────────────────────┘
```

### Document Indexing Pipeline (runs once on upload)

```
PDF / TXT / DOCX
      │
      ▼
Text Extraction  (PyPDF / python-docx)
      │
      ▼
Recursive Text Splitting  (chunk_size=512, overlap=50)
      │
      ▼
HuggingFace Embedding  (batch encode all chunks)
      │
      ▼
Pinecone Upsert  (stored with source + page metadata)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | [Groq](https://groq.com) — Llama 3.3 70B | Ultra-fast answer generation |
| **Vector Database** | [Pinecone](https://pinecone.io) Serverless | Store and search embeddings |
| **Embeddings** | [HuggingFace](https://sbert.net) sentence-transformers | Convert text to vectors |
| **UI Framework** | [Streamlit](https://streamlit.io) | Web interface |
| **PDF Parsing** | [PyPDF](https://pypdf.readthedocs.io) | Extract text from PDFs |
| **DOCX Parsing** | [python-docx](https://python-docx.readthedocs.io) | Extract text from Word files |
| **Language** | Python 3.10+ | Core programming language |

### Why Groq?
Groq runs open-source LLMs on custom LPU (Language Processing Unit) hardware, delivering **10-100x faster inference** than traditional GPU-based services. This means answers in under 2 seconds even for large contexts.

### Why Pinecone?
Pinecone is a fully managed, serverless vector database. It handles billions of vectors with millisecond query times and zero infrastructure management. The free tier supports 1 index with 100,000 vectors — more than enough for most projects.

---

## 🔬 Advanced RAG Techniques

This project implements several techniques that go beyond basic RAG:

### 1. Recursive Text Splitting
Instead of naively cutting text at fixed character counts, the splitter tries to break on natural boundaries in order: paragraphs → sentences → words → characters. This preserves semantic coherence within each chunk.

### 2. Chunk Overlap
Each chunk shares `overlap` characters with the previous chunk. This ensures that sentences or ideas spanning a chunk boundary are not lost.

### 3. Query Rewriting
When a user asks a follow-up question like *"What about his education?"*, the system uses a fast LLM to rewrite it as a standalone question *"What is Anuj Vishwakarma's educational background?"* before searching. This dramatically improves retrieval accuracy in multi-turn conversations.

### 4. MMR — Maximum Marginal Relevance
Standard semantic search returns the top-K most similar chunks, which can all say the same thing. MMR balances **relevance** (similar to query) and **diversity** (different from already selected chunks), giving a broader and more complete view of the answer.

### 5. Source Attribution
Every retrieved chunk carries metadata — source filename, page number, and similarity score. These are surfaced in the UI so users can verify answers against the original document.

### 6. Sliding Window Memory
The last 6 conversation turns are included in every LLM prompt. This allows natural follow-up questions without losing context from earlier in the conversation.

---

## 📁 Project Structure

```
advanced-rag-chatbot/
│
├── app.py                  # Streamlit UI — sidebar, chat interface, session state
├── rag_engine.py           # Core RAG pipeline — retrieval, rewriting, generation
├── document_processor.py   # Document loader and chunker for PDF / TXT / DOCX
├── requirements.txt        # All Python dependencies
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Streamlit theme and server settings
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com)
- A free [Pinecone API key](https://pinecone.io)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/anujvish005/advanced-rag-chatbot.git
cd advanced-rag-chatbot

# 2. Optional — create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📖 How to Use

**1. Enter API Keys**
Open the sidebar and paste your Groq and Pinecone API keys into the fields provided. Keys are never stored — they exist only in your browser session.

**2. Upload Documents**
Click "Upload PDFs, TXT, or DOCX" and select one or more files. The app supports multiple documents at once.

**3. Process and Index**
Click the "⚡ Process & Index Documents" button. The app will extract text, split it into chunks, embed them using HuggingFace, and store them in Pinecone. You will see a success message showing the number of chunks indexed.

**4. Start Chatting**
Type any question in the chat input. The app retrieves the most relevant chunks, generates a grounded answer using Groq, and displays it along with the source document references.

**5. Ask Follow-up Questions**
Ask follow-ups naturally — the app remembers the full conversation and rewrites your questions for better retrieval accuracy.

---

## ☁️ Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** and select this repository
4. Set main file path to `app.py`
5. Click **Advanced settings → Secrets** and add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
PINECONE_API_KEY = "pcsk_your_key_here"
```

6. Click **Deploy** — you will get a shareable public URL in 2-3 minutes

---

## ⚙️ Configuration Options

All settings are available in the sidebar at runtime — no code changes needed:

| Setting | Default | Description |
|---|---|---|
| LLM Model | `llama-3.3-70b-versatile` | Groq model for answer generation |
| Embedding Model | `all-MiniLM-L6-v2` | HuggingFace model for embeddings |
| Temperature | `0.2` | Lower = more factual answers |
| Top-K Chunks | `5` | Number of chunks retrieved per query |
| Chunk Size | `512` | Characters per chunk |
| Chunk Overlap | `50` | Overlap between adjacent chunks |
| Retrieval Strategy | `Semantic Search` | Semantic Search or MMR |

---

## 👤 Author

**Anuj Vishwakarma**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/anuj-vishwakarma-38571b209)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/anujvish005)

---

## 📄 License

This project is licensed under the MIT License — free to use, modify, and distribute.

---

<div align="center">
If this project helped you, please ⭐ star the repo and share it on LinkedIn!
</div>

