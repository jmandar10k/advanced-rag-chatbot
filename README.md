# 🤖 Advanced RAG Chatbot

An intelligent document-based chatbot powered by **Retrieval-Augmented Generation (RAG)** with hybrid retrieval, multi-query expansion, and real-time streaming responses. Built with LangChain, LLaMA 3.1, and Streamlit.

---

## 🚀 Demo

> 💡 Upload any PDF → Ask questions → Get accurate, context-aware answers in real time.

---

## 🧠 What Makes This Advanced?

Most RAG systems use basic vector similarity search — this one doesn't.

| Feature | Basic RAG | This Project |
|---|---|---|
| Retrieval | Single vector search | Hybrid (FAISS + BM25) |
| Query handling | Single query | MultiQuery expansion |
| Result quality | Raw retrieval | Reranking applied |
| Memory | None | Multi-session conversational memory |
| Response | Batch | Real-time streaming |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3.1 via Groq API |
| Orchestration | LangChain |
| Vector Store | FAISS |
| Keyword Search | BM25 (Hybrid Retrieval) |
| Reranking | Cross-encoder reranker |
| Embeddings | HuggingFace Embeddings |
| Frontend | Streamlit |
| Document Parsing | PyMuPDF / PDFPlumber |

---

## 🔍 System Architecture

```
User Query
    │
    ▼
MultiQuery Expansion (generates multiple query variants)
    │
    ▼
Hybrid Retrieval
├── FAISS (semantic/vector search)
└── BM25  (keyword/lexical search)
    │
    ▼
Reranking (cross-encoder selects most relevant chunks)
    │
    ▼
LLaMA 3.1 via Groq (context-aware answer generation)
    │
    ▼
Streaming Response → Streamlit UI
```

---

## ✨ Key Features

- **Hybrid Retrieval (FAISS + BM25)** — Combines semantic and keyword search for better chunk coverage
- **MultiQuery Expansion** — Automatically generates multiple query variants to improve recall
- **Reranking** — Filters and reorders retrieved chunks by relevance before passing to LLM
- **Conversational Memory** — Maintains context across turns within a session
- **Multi-session Support** — Manage and switch between multiple chat sessions
- **Real-time Streaming** — Responses stream token-by-token for a natural chat feel
- **PDF Ingestion** — Upload any PDF and start querying instantly

---

## 📁 Project Structure

```
advanced-rag-chatbot/
│
├── app.py                  
├── rag_backend.py         
├── requirements.txt        
├── .gitignore
└── README.md
```

---

## 🛠️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/jmandar10k/advanced-rag-chatbot.git
cd advanced-rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
# Create a .env file and add:
GROQ_API_KEY=your_groq_api_key_here

# 4. Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
langchain
langchain-community
langchain-groq
faiss-cpu
rank-bm25
sentence-transformers
streamlit
pymupdf
python-dotenv
```

---

## 💡 How It Works

1. **PDF Ingestion** — Uploaded PDF is parsed, split into chunks, and embedded
2. **Hybrid Index Built** — FAISS index (semantic) and BM25 index (keyword) created simultaneously
3. **Query Expansion** — User query is expanded into multiple variants using LLM
4. **Hybrid Retrieval** — Both indexes queried, results merged
5. **Reranking** — Cross-encoder scores and reorders chunks by relevance
6. **Answer Generation** — Top chunks + conversation history passed to LLaMA 3.1 for answer
7. **Streaming Output** — Response streamed live to Streamlit UI

---

## 📸 Screenshots

<!-- Add screenshots of your app here -->
| Chat Interface | Multi-session View |
|---|---|
| ![Chat](screenshots/chat.png) | ![Sessions](screenshots/sessions.png) |

---

## 🙋‍♂️ Author

**Mandar Joshi**
- 📧 jmandar1322@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/mandar-j-016244200)
- 🐙 [GitHub](https://github.com/jmandar10k

---

## 📄 License

This project is licensed under the MIT License.
