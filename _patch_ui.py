NEW_UI = '''# ----------------------------
# 🚀 Streamlit UI — ChatGPT Style (Centered)
# ----------------------------
st.set_page_config(page_title="Adaptive RAG", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0f1117; color: #ececec; }
    [data-testid="collapsedControl"], section[data-testid="stSidebar"] { display: none; }
    .app-header { text-align: center; padding: 2rem 0 0.8rem; }
    .app-header h1 { font-size: 1.9rem; font-weight: 700; color: #fff; margin: 0; }
    .app-header p  { color: #8b949e; font-size: 0.88rem; margin-top: 0.4rem; }
    .question-bubble {
        background: #1f6feb22; border: 1px solid #1f6feb55;
        border-radius: 12px; padding: 0.85rem 1.1rem;
        margin: 1.2rem 0 0.4rem; color: #58a6ff; font-weight: 500;
    }
    .answer-bubble {
        background: #161b22; border: 1px solid #30363d;
        border-left: 3px solid #58a6ff; border-radius: 12px;
        padding: 1.1rem 1.4rem; color: #c9d1d9; line-height: 1.75;
    }
    .info-bar { display: flex; gap: 0.6rem; margin-top: 0.6rem; flex-wrap: wrap; }
    .info-chip {
        background: #21262d; border: 1px solid #30363d;
        border-radius: 20px; padding: 0.2rem 0.75rem;
        font-size: 0.75rem; color: #8b949e;
    }
    .info-chip.green { border-color: #238636; color: #3fb950; }
    .info-chip.red   { border-color: #da3633; color: #f85149; }
    .info-chip.blue  { border-color: #1f6feb; color: #58a6ff; }
    .stTextArea textarea {
        background-color: #161b22 !important; border: 1px solid #30363d !important;
        border-radius: 10px !important; color: #c9d1d9 !important;
        font-family: "Inter", sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: #58a6ff !important; box-shadow: 0 0 0 2px #1f6feb33 !important;
    }
    .stButton > button[kind="primary"] {
        background: #238636 !important; border: none !important;
        color: white !important; border-radius: 8px !important;
        font-weight: 600 !important; width: 100% !important;
        transition: background 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover { background: #2ea043 !important; }
    .stButton > button[kind="secondary"] {
        background: transparent !important; border: 1px solid #30363d !important;
        color: #8b949e !important; border-radius: 8px !important; font-size: 0.8rem !important;
    }
    div[data-testid="stFileUploader"] > div {
        background: #161b22 !important; border: 1px dashed #30363d !important;
        border-radius: 10px !important;
    }
    .stSpinner > div { color: #58a6ff !important; }
    .stSuccess { background-color: #1a2e1a !important; border-color: #238636 !important; }
    .stWarning { background-color: #2e1e00 !important; }
    h1, h2, h3, h4 { color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="app-header">
    <h1>🧠 Adaptive RAG</h1>
    <p>Tanya dokumen PDF kamu — AI dengan self-reflection &amp; anti-halusinasi</p>
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
    st.markdown(f\'<div class="question-bubble">👤 {q}</div>\', unsafe_allow_html=True)
    st.markdown(f\'<div class="answer-bubble">🧠 {ans}</div>\', unsafe_allow_html=True)
    h = chat.get("hallucination_score", "N/A")
    a = chat.get("answer_score", "N/A")
    web = chat.get("web", False)
    i = chat.get("iterations", 0)
    hc = "green" if h == "yes" else "red"
    ac = "green" if a == "yes" else "red"
    ht = "✓ Tidak halusinasi" if h == "yes" else "⚠ Halusinasi"
    at = "✓ Relevan" if a == "yes" else "⚠ Kurang relevan"
    h_chip = f\'<span class="info-chip {hc}">{ht}</span>\'
    a_chip = f\'<span class="info-chip {ac}">{at}</span>\'
    i_chip = f\'<span class="info-chip blue">🔄 {i} iterasi</span>\'
    w_chip = \'<span class="info-chip blue">🌐 Web search</span>\' if web else ""
    st.markdown(f\'<div class="info-bar">{h_chip}{a_chip}{i_chip}{w_chip}</div>\', unsafe_allow_html=True)

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
'''

content = open('streamlit_app.py', encoding='utf-8').read()
marker = '# ----------------------------\n# \U0001f680 Streamlit UI'
cut_at = content.find(marker)
if cut_at == -1:
    print("ERROR: marker not found")
else:
    new_content = content[:cut_at] + NEW_UI
    open('streamlit_app.py', 'w', encoding='utf-8').write(new_content)
    print(f"OK — {new_content.count(chr(10))} lines written")
