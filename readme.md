# 🧠 AI Document Analysis — Adaptive RAG

An intelligent document Q&A system powered by **Adaptive RAG** (Retrieval-Augmented Generation) with self-reflection, hallucination detection, and web search fallback.

🔗 **Live Demo**: [yusuf-documents.streamlit.app](https://yusuf-documents.streamlit.app)

---

## 🏗️ RAG Architecture

```mermaid
flowchart TD
    A([❓ User Question]) --> B[🔍 Retrieve\nVector Search FAISS]
    B --> C{📋 Grade Documents\nRelevance Check}
    C -- Relevant --> D[✨ Generate Answer\nLlama 3.1 8B]
    C -- Not Relevant --> G[🌐 Web Search\nTavily Fallback]
    D --> E{🔎 Grade Generation}
    E -- ✅ Grounded + Relevant --> F([💬 Final Answer])
    G --> F
    E -- ❌ Hallucination / Off-topic --> H[✍️ Rewrite Question]
    H --> B
    E -- Max iterations reached --> F

    style A fill:#dbeafe,stroke:#93c5fd,color:#1d4ed8
    style F fill:#dcfce7,stroke:#86efac,color:#166534
    style H fill:#fef9c3,stroke:#fde68a,color:#92400e
    style G fill:#f3e8ff,stroke:#d8b4fe,color:#6b21a8
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF Upload** | Upload any PDF and ask questions about it |
| 🔍 **Vector Retrieval** | FAISS vector search with `all-MiniLM-L6-v2` embeddings |
| 🧠 **LLM Generation** | Grounded answers using `llama-3.1-8b-instant` via Groq |
| ✅ **Hallucination Check** | Auto-validates every answer against source documents |
| 🌐 **Web Fallback** | Falls back to Tavily web search if PDF lacks info |
| ✍️ **Self-Reflection** | Rewrites and retries questions up to 3 iterations |
| ⚡ **Parallel Grading** | Hallucination + relevance graders run in parallel |

---

## � Local Setup

```bash
git clone https://github.com/yusuf-analytics/Rag-Document.git
cd Rag-Document
pip install -r requirements.txt
```

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Run the app:
```bash
streamlit run streamlit_app.py
```

---

## ☁️ Streamlit Cloud Deployment

In **Settings → Secrets**, add:
```toml
GROQ_API_KEY = "your_key"
GOOGLE_API_KEY = "your_key"
TAVILY_API_KEY = "your_key"
LANGCHAIN_API_KEY = ""
```

---

## �️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — `llama-3.1-8b-instant` |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector Store | FAISS |
| Orchestration | LangGraph |
| Web Search | Tavily |
| Framework | LangChain + Streamlit |
