import streamlit as st
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor   # ⚡ parallel grader calls
from typing import List, Dict, Any, TypedDict, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path
from datetime import datetime

# Load environment variables (local dev)
load_dotenv()

def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets (Streamlit Cloud) or os.getenv (.env) — works in both environments."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)

# Configure API keys
api_key = _get_secret("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
tavily_api_key = _get_secret("TAVILY_API_KEY")
groq_api_key = _get_secret("GROQ_API_KEY")
langchain_api_key = _get_secret("LANGCHAIN_API_KEY")

# Only enable LangSmith tracing if a real key is provided
if langchain_api_key and langchain_api_key != "your_langchain_api_key_here":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Create directories
os.makedirs("temp_uploads", exist_ok=True)

# ----------------------------
# 🔐 Graph State Definition
# ----------------------------
class GraphState(TypedDict, total=False):
    question: str
    document_context: str
    retrieved_docs: List[str]
    grade: str
    generation: str
    web_search_results: str
    rewritten_question: str
    final_answer: str
    iterations: int
    max_iterations: int
    hallucination_score: str
    answer_score: str

# ----------------------------
# 🤖 LLM Configuration
# ----------------------------
if groq_api_key:
    llm_generator = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=1024,
        groq_api_key=groq_api_key
    )
    llm_grader = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=10,
        groq_api_key=groq_api_key
    )
    llm = llm_generator
else:
    st.error("⚠️ GROQ_API_KEY tidak ditemukan. Tambahkan di Streamlit Cloud → Settings → Secrets, atau di file .env untuk lokal.")
    st.stop()


# ----------------------------
# 🤖 Embedding Configuration
# ----------------------------
class CustomLocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

@st.cache_resource
def get_embeddings():
    return CustomLocalEmbeddings()

embeddings = get_embeddings()

# Web search tool
web_search_tool = TavilySearchResults(
    k=3,
    tavily_api_key=tavily_api_key
) if tavily_api_key else None

# ----------------------------
# 📄 Document Processing
# ----------------------------
class DocumentProcessor:
    def __init__(self):
        # 🧠 STRATEGY 4: Larger overlap keeps sentences together → less fragmented context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=400,
            length_function=len,
        )
        self.vectorstore = None
    
    def process_pdf(self, file_path: str):
        """Process PDF and create vector store"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Combine all text for context
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        return combined_text
    
    def retrieve_docs(self, query: str, k: int = 4) -> List[str]:
        """Retrieve relevant documents — k=4 balances speed vs. coverage"""
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        else:
            return []

# ----------------------------
# 🎯 Prompt Templates (Anti-Hallucination Tuned)
# ----------------------------
GRADING_PROMPT = PromptTemplate.from_template("""
You are a strict relevance grader. Your ONLY job is to output a single word.

Documents:
{documents}

User Question: {question}

Instructions:
- Output ONLY the word "relevant" if the documents contain ANY information related to the question.
- Output ONLY the words "not relevant" if the documents are completely unrelated to the question.
- Do NOT output any other text. No explanation. No punctuation. No extra words.

Grade:""")

# Grounded generation: model must cite before answering (prevents hallucination)
GENERATION_PROMPT = PromptTemplate.from_template("""
You are a document Q&A assistant. Answer ONLY from the context below. Never use outside knowledge.

<context>
{context}
</context>

Question: {question}

Rules:
- If the answer exists in <context>: first write [Quote: "exact text"] then write Answer: <answer>
- If NOT found in <context>: output exactly "Sorry, the information is not found in the uploaded PDF document."
- No guessing. No extra knowledge. No invented facts.

[Quote: "..."]
Answer:""")

HALLUCINATION_GRADER_PROMPT = PromptTemplate.from_template("""
Facts:
{documents}

Answer:
{generation}

Is every claim in the Answer explicitly supported by the Facts? Output ONLY yes or no.
If the Answer says information was not found in the document, output yes.

Score:""")

ANSWER_GRADER_PROMPT = PromptTemplate.from_template("""
Question: {question}
Answer: {generation}

Does the Answer address the Question? Output ONLY yes or no.

Score:""")

QUESTION_REWRITER_PROMPT = PromptTemplate.from_template("""
You are a query optimizer. Rewrite the following question to be more specific and better suited for document retrieval.

Original Question: {question}

Output ONLY the rewritten question. No explanation.

Rewritten Question:""")

WEB_SEARCH_PROMPT = PromptTemplate.from_template("""
You are a research assistant. The main PDF document does not contain relevant information, so use the Web Search Results below to answer.

Web Search Results:
{web_results}

Question: {question}

Instructions:
1. Start your answer with EXACTLY: "*Note: Because the information was not found in the PDF document, this answer is taken from a web search.*"
2. Answer concisely and accurately based ONLY on the Web Search Results above.
3. Do NOT use external knowledge beyond what is provided in the Web Search Results.

Answer:""")

# ----------------------------
# �️ Strategy 5: Post-Processing Filter
# ----------------------------
FALLBACK_PHRASES = [
    "sorry, the information is not found",
    "information is not found in the uploaded pdf",
    "not found in the uploaded pdf document",
    "not explicitly mentioned in the context",
    "not mentioned in the provided context",
    "no information available in the document",
]

def _extract_clean_answer(raw_output: str) -> str:
    """
    Extract the clean Answer from CoT output.
    The CoT prompt produces:
        [Quote: \"...\"]
        Answer: <actual answer>

    This function:
    1. Strips the [Quote: ...] citation prefix.
    2. If \"Answer:\" label is present, extracts only that part.
    3. Detects if the model returned a fallback / not-found statement
       and normalises it to the canonical fallback phrase.
    """
    text = raw_output.strip()

    # Detect fallback — no need to parse CoT structure
    text_lower = text.lower()
    for phrase in FALLBACK_PHRASES:
        if phrase in text_lower:
            return "Sorry, the information is not found in the uploaded PDF document."

    # Extract Answer: section if CoT format is followed
    if "Answer:" in text:
        # Take everything after the last "Answer:" label
        answer_part = text.split("Answer:")[-1].strip()
        return answer_part

    # If model skipped the CoT format entirely, return as-is
    # (still grounded check will catch hallucinations)
    return text

# ----------------------------
# �🔄 LangGraph Nodes (LCEL — no deprecated LLMChain)
# ----------------------------
str_parser = StrOutputParser()

def retrieve_node(state: GraphState):
    question = state["question"]
    documents = st.session_state.doc_processor.retrieve_docs(question)
    return {**state, "retrieved_docs": documents, "iterations": state.get("iterations", 0)}

def grade_documents_node(state: GraphState):
    question = state["question"]
    documents = state["retrieved_docs"]
    docs_text = "\n\n".join(documents)

    # Use fast grader LLM for binary relevance check
    grading_chain = GRADING_PROMPT | llm_grader | str_parser
    result = grading_chain.invoke({"documents": docs_text, "question": question})

    raw_grade = result.strip().lower()
    if "not relevant" in raw_grade:
        grade = "not relevant"
    else:
        grade = "relevant"

    return {**state, "grade": grade, "document_context": docs_text}

def generate_node(state: GraphState):
    question = state["question"]
    documents = state["retrieved_docs"]
    context = "\n\n".join(documents)

    # 🧠 STRATEGY 1: Use stronger generator LLM
    generation_chain = GENERATION_PROMPT | llm_generator | str_parser
    raw_result = generation_chain.invoke({"context": context, "question": question})

    # 🧠 STRATEGY 5: Post-processing — extract clean Answer from CoT output
    # The CoT prompt returns [Quote: "..."] followed by Answer: ...
    # We extract only the Answer part for the final output.
    result = _extract_clean_answer(raw_result)

    return {**state, "generation": result, "iterations": state.get("iterations", 0) + 1}

def grade_generation_node(state: GraphState):
    question = state["question"]
    documents = state["retrieved_docs"]
    generation = state["generation"]
    docs_text = "\n\n".join(documents)

    hallucination_chain = HALLUCINATION_GRADER_PROMPT | llm_grader | str_parser
    answer_chain = ANSWER_GRADER_PROMPT | llm_grader | str_parser

    # ⚡ Run both graders IN PARALLEL — saves ~1-2 seconds per iteration
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_hallucination = executor.submit(
            hallucination_chain.invoke, {"documents": docs_text, "generation": generation}
        )
        f_answer = executor.submit(
            answer_chain.invoke, {"question": question, "generation": generation}
        )
        raw_hallucination = f_hallucination.result()
        raw_answer = f_answer.result()

    # Robust parsing — take first word, normalize to yes/no
    hallucination_score = raw_hallucination.strip().lower().split()[0] if raw_hallucination.strip() else "no"
    answer_score = raw_answer.strip().lower().split()[0] if raw_answer.strip() else "no"
    hallucination_score = "yes" if hallucination_score.startswith("yes") else "no"
    answer_score = "yes" if answer_score.startswith("yes") else "no"

    return {**state, "hallucination_score": hallucination_score, "answer_score": answer_score}

def rewrite_question_node(state: GraphState):
    st.write("✍️ Menulis ulang pertanyaan...")
    question = state["question"]

    # FIX 1: Use LCEL pipe
    rewrite_chain = QUESTION_REWRITER_PROMPT | llm | str_parser
    rewritten_question = rewrite_chain.invoke({"question": question})
    return {**state, "rewritten_question": rewritten_question, "question": rewritten_question}

def web_search_node(state: GraphState):
    question = state["question"]
    if not web_search_tool:
        return {**state, "final_answer": "Pencarian web tidak tersedia. Silakan konfigurasi TAVILY_API_KEY."}

    web_results = web_search_tool.run(question)

    # FIX 1 + FIX 2: Use LCEL pipe and only pass variables that exist in the template
    web_search_chain = WEB_SEARCH_PROMPT | llm | str_parser
    result = web_search_chain.invoke({"web_results": str(web_results), "question": question})

    return {**state, "web_search_results": str(web_results), "final_answer": result}

# ----------------------------
# 🔄 Decision Functions
# ----------------------------
def decide_to_generate(state: GraphState):
    if state["grade"] == "relevant":
        return "generate"
    else:
        return "web_search"

def grade_generation_decision(state: GraphState):
    hallucination_score = state.get("hallucination_score", "no")
    answer_score = state.get("answer_score", "no")
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)

    if iterations >= max_iterations:
        return "end"
    if hallucination_score == "yes" and answer_score == "yes":
        return "end"
    return "rewrite"

# ----------------------------
# 🏗️ Build Graph
# ----------------------------
def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_generation", grade_generation_node)
    workflow.add_node("rewrite_question", rewrite_question_node)
    workflow.add_node("web_search", web_search_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {"generate": "generate", "web_search": "web_search"})
    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges("grade_generation", grade_generation_decision, {"end": END, "rewrite": "rewrite_question"})
    workflow.add_edge("rewrite_question", "retrieve")
    workflow.add_edge("web_search", END)

    return workflow.compile()

# ----------------------------
# 🚀 Streamlit UI — ChatGPT Style (Centered)
# ----------------------------
st.set_page_config(page_title="AI Document Analysis", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f6f8fa; color: #1f2328; }
    [data-testid="collapsedControl"], section[data-testid="stSidebar"] { display: none; }
    .app-header { text-align: center; padding: 2rem 0 0.8rem; }
    .app-header h1 { font-size: 1.9rem; font-weight: 700; color: #1f2328; margin: 0; }
    .app-header p  { color: #57606a; font-size: 0.88rem; margin-top: 0.4rem; }
    .question-bubble {
        background: #dbeafe; border: 1px solid #93c5fd;
        border-radius: 12px; padding: 0.85rem 1.1rem;
        margin: 1.2rem 0 0.4rem; color: #1d4ed8; font-weight: 500;
    }
    .answer-bubble {
        background: #ffffff; border: 1px solid #d0d7de;
        border-left: 3px solid #2563eb; border-radius: 12px;
        padding: 1.1rem 1.4rem; color: #1f2328; line-height: 1.75;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    }
    .info-bar { display: flex; gap: 0.6rem; margin-top: 0.6rem; flex-wrap: wrap; }
    .info-chip {
        background: #f0f2f5; border: 1px solid #d0d7de;
        border-radius: 20px; padding: 0.2rem 0.75rem;
        font-size: 0.75rem; color: #57606a;
    }
    .info-chip.green { background: #dcfce7; border-color: #86efac; color: #166534; }
    .info-chip.red   { background: #fee2e2; border-color: #fca5a5; color: #991b1b; }
    .info-chip.blue  { background: #dbeafe; border-color: #93c5fd; color: #1d4ed8; }
    .stTextArea textarea {
        background-color: #ffffff !important; border: 1px solid #d0d7de !important;
        border-radius: 10px !important; color: #1f2328 !important;
        font-family: "Inter", sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: #2563eb !important; box-shadow: 0 0 0 2px #2563eb22 !important;
    }
    .stButton > button[kind="primary"] {
        background: #2563eb !important; border: none !important;
        color: white !important; border-radius: 8px !important;
        font-weight: 600 !important; width: 100% !important;
        transition: background 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover { background: #1d4ed8 !important; }
    .stButton > button[kind="secondary"] {
        background: #ffffff !important; border: 1px solid #d0d7de !important;
        color: #57606a !important; border-radius: 8px !important; font-size: 0.8rem !important;
    }
    div[data-testid="stFileUploader"] > div {
        background: #ffffff !important; border: 1px dashed #d0d7de !important;
        border-radius: 10px !important;
    }
    .stSpinner > div { color: #2563eb !important; }
    h1, h2, h3, h4 { color: #1f2328 !important; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="app-header">
    <h1>🧠 AI Document Analysis</h1>
    <p>Tanya dokumen PDF kamu — AI dengan self-reflection</p>
</div>
""", unsafe_allow_html=True)

# Session state
if "doc_processor" not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()
if "app_graph" not in st.session_state:
    st.session_state.app_graph = build_graph()
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Top bar: reset buttons
_, c1, c2 = st.columns([4, 1, 1])
with c1:
    if st.button("🗑️ Chat", help="Hapus riwayat chat"):
        st.session_state.chat_history = []
        st.rerun()
with c2:
    if st.button("🔄 Reset", help="Reset semua"):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()

# Chat history
for chat in st.session_state.chat_history:
    q = chat["question"].replace("<", "&lt;").replace(">", "&gt;")
    ans = chat["answer"].replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="question-bubble">👤 {q}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-bubble">🧠 {ans}</div>', unsafe_allow_html=True)
    h = chat.get("hallucination_score", "N/A")
    a = chat.get("answer_score", "N/A")
    web = chat.get("web", False)
    i = chat.get("iterations", 0)
    hc = "green" if h == "yes" else "red"
    ac = "green" if a == "yes" else "red"
    ht = "✓ Tidak halusinasi" if h == "yes" else "⚠ Halusinasi"
    at = "✓ Relevan" if a == "yes" else "⚠ Kurang relevan"
    h_chip = f'<span class="info-chip {hc}">{ht}</span>'
    a_chip = f'<span class="info-chip {ac}">{at}</span>'
    i_chip = f'<span class="info-chip blue">🔄 {i} iterasi</span>'
    w_chip = '<span class="info-chip blue">🌐 Web search</span>' if web else ""
    st.markdown(f'<div class="info-bar">{h_chip}{a_chip}{i_chip}{w_chip}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Active doc badge
if st.session_state.processed_file:
    st.success(f"📄 **{st.session_state.processed_file}** — dokumen aktif")

# File uploader (near prompt)
uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if uploaded_file and st.session_state.processed_file != uploaded_file.name:
    with st.spinner("⚙️ Memproses dokumen..."):
        temp_dir = Path("temp_uploads") / str(uuid.uuid4())
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.document_context = st.session_state.doc_processor.process_pdf(str(file_path))
        st.session_state.processed_file = uploaded_file.name
        st.session_state.chat_history = []
    st.success(f"✅ **{uploaded_file.name}** siap dianalisis!")
    st.rerun()

# Question input
question = st.text_area(
    "Pertanyaan",
    placeholder="Tanya apa saja tentang dokumen ini...",
    label_visibility="collapsed",
    height=80,
    key="question_input"
)
send_clicked = st.button("Kirim ➤", type="primary")

if send_clicked:
    if not st.session_state.processed_file:
        st.warning("📎 Upload dokumen PDF terlebih dahulu.")
    elif not question.strip():
        st.warning("Ketik pertanyaan terlebih dahulu.")
    else:
        initial_state = {
            "question": question,
            "document_context": st.session_state.get("document_context", ""),
            "max_iterations": 3,
            "iterations": 0
        }
        with st.spinner("Saya sedang menganalisis dokumen Anda..."):
            result = st.session_state.app_graph.invoke(initial_state)
        final_answer = result.get("final_answer", result.get("generation", "Tidak ada jawaban yang dihasilkan."))
        st.session_state.chat_history.append({
            "question": question,
            "answer": final_answer,
            "hallucination_score": result.get("hallucination_score", "N/A"),
            "answer_score": result.get("answer_score", "N/A"),
            "iterations": result.get("iterations", 0),
            "web": bool(result.get("web_search_results")),
        })
        st.rerun()
