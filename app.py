import streamlit as st

from helpers.pdf_loader import extract_text_from_pdf
from helpers.source_utils import extract_source_details
from helpers.text_splitter import split_text
from helpers.vector_store import create_faiss_index, search_similar_chunks


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="NeuraDocs",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Session state
# ============================================================
DEFAULT_STATE = {
    "all_chunks": [],
    "faiss_index": None,
    "processed_file_signature": [],
    "chat_history": [],
    "pdf_summary": "",
    "study_notes": "",
    "questions_asked": 0,
}


def clone_default(value):
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


for state_key, default_value in DEFAULT_STATE.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = clone_default(default_value)


def clear_application_data() -> None:
    """Clear all document-related state."""
    st.session_state.all_chunks = []
    st.session_state.faiss_index = None
    st.session_state.processed_file_signature = []
    st.session_state.chat_history = []
    st.session_state.pdf_summary = ""
    st.session_state.study_notes = ""
    st.session_state.questions_asked = 0

    if "pdf_question_input" in st.session_state:
        st.session_state.pdf_question_input = ""


def build_complete_document_text() -> str:
    """Combine all stored chunks into a single document string."""
    return "\n\n".join(st.session_state.all_chunks)


def format_user_friendly_error(error: Exception) -> str:
    """Convert technical exceptions into concise user-friendly messages."""
    error_text = str(error)
    error_lower = error_text.lower()

    if (
        "429" in error_text
        or "resource_exhausted" in error_lower
        or "quota" in error_lower
        or "rate limit" in error_lower
    ):
        return (
            "The current AI provider quota has been reached. "
            "Please try again later or verify that the fallback provider is configured."
        )

    if (
        "api key" in error_lower
        or "google_api_key" in error_lower
        or "groq_api_key" in error_lower
        or "unauthenticated" in error_lower
    ):
        return (
            "An AI API key is missing or invalid. "
            "Please verify the Streamlit Secrets configuration."
        )

    if (
        "503" in error_text
        or "unavailable" in error_lower
        or "high demand" in error_lower
        or "timeout" in error_lower
    ):
        return (
            "The AI service is temporarily unavailable. "
            "Please wait briefly and try again."
        )

    if "huggingface" in error_lower or "couldn't connect" in error_lower:
        return (
            "The embedding model could not be loaded. "
            "Please check the deployment connection and model configuration."
        )

    if "reportlab" in error_lower:
        return (
            "PDF export is unavailable because ReportLab is not installed correctly."
        )

    return f"An unexpected error occurred: {error_text}"


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="side-logo">🧠</div>
            <div>
                <div class="side-title">NeuraDocs</div>
                <div class="side-subtitle">AI PDF Study Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_theme = st.selectbox(
        "Appearance",
        options=["Light", "Dark"],
        index=0,
        key="neuradocs_theme_selector",
        help="Switch between Light and Dark mode.",
    )

    st.markdown(
        '<div class="side-label">DOCUMENTS</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based PDF files.",
        label_visibility="collapsed",
    )

    st.caption(
        "Text-based PDFs work best. Scanned PDFs will require OCR support."
    )

    if st.button(
        "🗑️ Clear workspace",
        use_container_width=True,
        key="clear_workspace_button",
    ):
        clear_application_data()
        st.success("Workspace cleared.")
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div class="side-mini-card">
            <div class="side-mini-title">Quick guide</div>
            <div class="side-mini-text">
                1. Upload PDFs<br>
                2. Wait for indexing<br>
                3. Generate notes or ask questions
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Theme values
# ============================================================
if selected_theme == "Dark":
    theme = {
        "page_bg": "#070b14",
        "page_bg_2": "#0f172a",
        "surface": "#111827",
        "surface_2": "#172033",
        "border": "#2b3a55",
        "text": "#f8fafc",
        "muted": "#a8b3c7",
        "heading": "#ffffff",
        "input": "#0b1220",
        "shadow": "rgba(0, 0, 0, 0.35)",
        "soft_blue": "rgba(59, 130, 246, 0.12)",
    }
else:
    theme = {
        "page_bg": "#f6f8fc",
        "page_bg_2": "#eef2ff",
        "surface": "#ffffff",
        "surface_2": "#f8fafc",
        "border": "#dbe4f0",
        "text": "#172033",
        "muted": "#64748b",
        "heading": "#0f172a",
        "input": "#ffffff",
        "shadow": "rgba(15, 23, 42, 0.09)",
        "soft_blue": "rgba(37, 99, 235, 0.08)",
    }


# ============================================================
# Premium UI CSS
# ============================================================
st.markdown(
    f"""
    <style>
    #MainMenu, footer {{
        visibility: hidden;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {{
        background:
            radial-gradient(circle at 10% 0%, {theme['soft_blue']} 0%, transparent 32%),
            linear-gradient(135deg, {theme['page_bg']} 0%, {theme['page_bg_2']} 100%) !important;
        color: {theme['text']} !important;
    }}

    .block-container {{
        max-width: 1280px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }}

    /* Typography */
    [data-testid="stAppViewContainer"] .main p,
    [data-testid="stAppViewContainer"] .main li,
    [data-testid="stAppViewContainer"] .main label,
    [data-testid="stAppViewContainer"] .main span,
    [data-testid="stAppViewContainer"] .main small {{
        color: {theme['text']} !important;
        -webkit-text-fill-color: {theme['text']} !important;
    }}

    [data-testid="stAppViewContainer"] .main h1,
    [data-testid="stAppViewContainer"] .main h2,
    [data-testid="stAppViewContainer"] .main h3,
    [data-testid="stAppViewContainer"] .main h4 {{
        color: {theme['heading']} !important;
        -webkit-text-fill-color: {theme['heading']} !important;
        letter-spacing: -0.02em;
    }}

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {{
        color: {theme['muted']} !important;
        -webkit-text-fill-color: {theme['muted']} !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background:
            radial-gradient(circle at 20% 0%, rgba(79, 70, 229, 0.28), transparent 32%),
            linear-gradient(180deg, #08111f 0%, #0f172a 100%) !important;
        border-right: 1px solid #243149 !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }}

    .side-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 2px 18px 2px;
    }}

    .side-logo {{
        width: 46px;
        height: 46px;
        border-radius: 15px;
        display: grid;
        place-items: center;
        font-size: 24px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.32);
    }}

    .side-title {{
        color: #ffffff !important;
        font-size: 23px;
        font-weight: 850;
        line-height: 1.1;
    }}

    .side-subtitle {{
        color: #aebbd0 !important;
        font-size: 12px;
        margin-top: 4px;
    }}

    .side-label {{
        color: #8090aa !important;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.15em;
        margin: 18px 0 8px 2px;
    }}

    .side-mini-card {{
        padding: 14px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}

    .side-mini-title {{
        color: #ffffff !important;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 7px;
    }}

    .side-mini-text {{
        color: #b7c2d4 !important;
        font-size: 12px;
        line-height: 1.7;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background: #111b2d !important;
        border: 1px solid #33425d !important;
        border-radius: 11px !important;
        min-height: 42px;
    }}

    section[data-testid="stSidebar"] section[data-testid="stFileUploader"] {{
        background: rgba(255, 255, 255, 0.045) !important;
        border: 1px dashed #43516a !important;
        border-radius: 16px !important;
        padding: 6px !important;
    }}

    section[data-testid="stSidebar"] section[data-testid="stFileUploader"] button {{
        background: #f8fafc !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 750 !important;
    }}

    section[data-testid="stSidebar"] section[data-testid="stFileUploader"] button * {{
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stAlert"] {{
        background: rgba(37, 99, 235, 0.16) !important;
        border: 1px solid rgba(96, 165, 250, 0.25) !important;
    }}

    /* Hero */
    .hero-shell {{
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        padding: 36px 34px;
        margin-bottom: 22px;
        background:
            radial-gradient(circle at 82% 20%, rgba(255,255,255,0.18), transparent 24%),
            linear-gradient(135deg, #172554 0%, #1d4ed8 48%, #6d28d9 100%);
        box-shadow: 0 24px 55px rgba(30, 64, 175, 0.28);
    }}

    .hero-shell::after {{
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -90px;
        bottom: -140px;
        border-radius: 50%;
        background: rgba(255,255,255,0.09);
    }}

    .hero-badge {{
        display: inline-block;
        padding: 7px 12px;
        margin-bottom: 14px;
        border-radius: 999px;
        color: #dbeafe !important;
        background: rgba(255,255,255,0.11);
        border: 1px solid rgba(255,255,255,0.16);
        font-size: 12px;
        font-weight: 750;
        letter-spacing: 0.04em;
    }}

    .hero-title {{
        color: #ffffff !important;
        font-size: 49px;
        font-weight: 900;
        letter-spacing: -0.045em;
        line-height: 1.04;
    }}

    .hero-copy {{
        max-width: 720px;
        color: #dbeafe !important;
        font-size: 17px;
        line-height: 1.7;
        margin-top: 12px;
    }}

    .hero-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 22px;
    }}

    .hero-pill {{
        color: #ffffff !important;
        padding: 8px 11px;
        border-radius: 10px;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.13);
        font-size: 12px;
        font-weight: 700;
    }}

    /* Reusable cards */
    .premium-card {{
        background: {theme['surface']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 12px 30px {theme['shadow']};
        margin-bottom: 14px;
    }}

    .section-kicker {{
        color: #2563eb !important;
        font-size: 11px;
        font-weight: 850;
        letter-spacing: 0.14em;
        margin-bottom: 6px;
    }}

    .section-title {{
        color: {theme['heading']} !important;
        font-size: 25px;
        font-weight: 850;
        letter-spacing: -0.025em;
    }}

    .section-copy {{
        color: {theme['muted']} !important;
        font-size: 14px;
        line-height: 1.65;
        margin-top: 5px;
    }}

    .empty-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 18px;
    }}

    .feature-chip {{
        background: {theme['surface_2']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 14px;
        padding: 16px 13px;
        color: {theme['text']} !important;
        font-weight: 750;
        text-align: center;
        font-size: 13px;
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: {theme['surface']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 18px;
        padding: 17px 18px;
        min-height: 116px;
        box-shadow: 0 10px 25px {theme['shadow']};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    div[data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 15px 34px {theme['shadow']};
    }}

    div[data-testid="stMetricLabel"] p {{
        color: {theme['muted']} !important;
        -webkit-text-fill-color: {theme['muted']} !important;
        font-weight: 700 !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {theme['heading']} !important;
        -webkit-text-fill-color: {theme['heading']} !important;
        font-weight: 900 !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {theme['surface']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 15px;
        padding: 6px;
        box-shadow: 0 8px 22px {theme['shadow']};
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 45px;
        border-radius: 10px;
        color: {theme['muted']} !important;
        font-weight: 750;
        padding: 0 16px;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, #2563eb, #4f46e5) !important;
        color: #ffffff !important;
    }}

    .stTabs [aria-selected="true"] * {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}

    /* Buttons */
    .stButton > button {{
        width: 100%;
        min-height: 47px;
        border: none !important;
        border-radius: 12px !important;
        background: linear-gradient(90deg, #2563eb, #4f46e5) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.22);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(37, 99, 235, 0.3);
    }}

    .stButton > button:disabled {{
        background: #8b98aa !important;
        opacity: 0.65 !important;
        box-shadow: none !important;
    }}

    .stButton > button *,
    .stDownloadButton > button * {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}

    .stDownloadButton > button {{
        width: 100%;
        min-height: 45px;
        border: none !important;
        border-radius: 12px !important;
        background: linear-gradient(90deg, #059669, #0891b2) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
    }}

    /* Inputs */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {{
        background: {theme['input']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 13px !important;
    }}

    div[data-baseweb="input"] input,
    textarea,
    textarea:disabled {{
        background: {theme['input']} !important;
        color: {theme['text']} !important;
        -webkit-text-fill-color: {theme['text']} !important;
        opacity: 1 !important;
    }}

    div[data-baseweb="input"] input::placeholder {{
        color: {theme['muted']} !important;
        -webkit-text-fill-color: {theme['muted']} !important;
        opacity: 1 !important;
    }}

    div[data-baseweb="input"] > div:focus-within {{
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }}

    /* Expanders */
    details[data-testid="stExpander"] {{
        background: {theme['surface']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 15px !important;
        overflow: hidden;
    }}

    details[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary *,
    details[data-testid="stExpander"] p,
    details[data-testid="stExpander"] span,
    details[data-testid="stExpander"] li {{
        color: {theme['text']} !important;
        -webkit-text-fill-color: {theme['text']} !important;
    }}

    /* Alerts */
    [data-testid="stAlert"] {{
        border-radius: 14px !important;
    }}

    [data-testid="stAlert"] * {{
        color: {theme['text']} !important;
        -webkit-text-fill-color: {theme['text']} !important;
    }}

    /* Markdown output */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em {{
        color: {theme['text']} !important;
        -webkit-text-fill-color: {theme['text']} !important;
        line-height: 1.72;
    }}

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {{
        color: {theme['heading']} !important;
        -webkit-text-fill-color: {theme['heading']} !important;
    }}

    /* Chat */
    [data-testid="stChatMessage"] {{
        background: {theme['surface']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 17px !important;
        padding: 8px !important;
        box-shadow: 0 8px 22px {theme['shadow']};
        margin-bottom: 11px;
    }}

    /* Code blocks */
    .block-container pre,
    .block-container pre code,
    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] code,
    [data-testid="stCodeBlock"] span {{
        background: #08111f !important;
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }}

    .block-container pre {{
        border: 1px solid #2b3a55 !important;
        border-radius: 14px !important;
        padding: 18px !important;
        overflow-x: auto !important;
    }}

    hr {{
        border-color: {theme['border']} !important;
    }}

    .footer-card {{
        text-align: center;
        padding: 20px 8px 4px 8px;
        color: {theme['muted']} !important;
        font-size: 12px;
    }}

    .footer-card b {{
        color: {theme['heading']} !important;
    }}

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 99999 !important;
    }}

    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="collapsedControl"] * {{
        color: {theme['heading']} !important;
        fill: {theme['heading']} !important;
        stroke: {theme['heading']} !important;
    }}

    @media (max-width: 820px) {{
        .block-container {{
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }}

        .hero-shell {{
            padding: 28px 20px;
            border-radius: 21px;
        }}

        .hero-title {{
            font-size: 37px;
        }}

        .hero-copy {{
            font-size: 15px;
        }}

        .empty-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Hero
# ============================================================
st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-badge">AI-POWERED DOCUMENT INTELLIGENCE</div>
        <div class="hero-title">Turn PDFs into answers.</div>
        <div class="hero-copy">
            Ask questions, create structured notes, generate summaries,
            and discover the most relevant information from your documents.
        </div>
        <div class="hero-pills">
            <div class="hero-pill">⚡ Semantic Search</div>
            <div class="hero-pill">🧠 RAG Pipeline</div>
            <div class="hero-pill">📚 Study Notes</div>
            <div class="hero-pill">📄 PDF Export</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PDF processing
# ============================================================
if uploaded_files:
    current_file_signature = [
        (uploaded_file.name, uploaded_file.size)
        for uploaded_file in uploaded_files
    ]

    files_have_changed = (
        current_file_signature
        != st.session_state.processed_file_signature
    )

    if files_have_changed:
        with st.spinner(
            "Reading your PDFs and creating the semantic search index..."
        ):
            collected_chunks = []

            for pdf_file in uploaded_files:
                try:
                    pdf_text = extract_text_from_pdf(pdf_file)

                    if not pdf_text or not pdf_text.strip():
                        st.warning(
                            f"No readable text was found in {pdf_file.name}. "
                            "The file may be scanned or image-based."
                        )
                        continue

                    pdf_chunks = split_text(pdf_text)

                    if not pdf_chunks:
                        st.warning(
                            f"No usable text chunks could be created from {pdf_file.name}."
                        )
                        continue

                    for chunk_number, chunk in enumerate(
                        pdf_chunks,
                        start=1,
                    ):
                        formatted_chunk = (
                            f"Source PDF: {pdf_file.name}\n"
                            f"Chunk Number: {chunk_number}\n\n"
                            f"{chunk}"
                        )
                        collected_chunks.append(formatted_chunk)

                except Exception as error:
                    st.error(
                        f"Could not process {pdf_file.name}: "
                        f"{format_user_friendly_error(error)}"
                    )

            if collected_chunks:
                try:
                    from helpers.embedding_model import create_embeddings

                    document_embeddings = create_embeddings(
                        collected_chunks
                    )
                    vector_index = create_faiss_index(
                        document_embeddings
                    )

                    st.session_state.all_chunks = collected_chunks
                    st.session_state.faiss_index = vector_index
                    st.session_state.processed_file_signature = (
                        current_file_signature
                    )
                    st.session_state.chat_history = []
                    st.session_state.pdf_summary = ""
                    st.session_state.study_notes = ""
                    st.session_state.questions_asked = 0

                except Exception as error:
                    st.error(format_user_friendly_error(error))
            else:
                clear_application_data()


# ============================================================
# Derived document state
# ============================================================
documents_are_ready = (
    bool(st.session_state.all_chunks)
    and st.session_state.faiss_index is not None
)

pdf_count = len(uploaded_files) if uploaded_files else 0
chunk_count = len(st.session_state.all_chunks)
embedding_count = (
    st.session_state.faiss_index.ntotal
    if st.session_state.faiss_index is not None
    else 0
)


# ============================================================
# Empty state / status
# ============================================================
if not documents_are_ready:
    st.markdown(
        """
        <div class="premium-card">
            <div class="section-kicker">GET STARTED</div>
            <div class="section-title">Your document workspace is ready.</div>
            <div class="section-copy">
                Upload one or more PDFs from the sidebar. NeuraDocs will extract,
                index, and prepare them for intelligent search.
            </div>
            <div class="empty-grid">
                <div class="feature-chip">💬 Ask Questions</div>
                <div class="feature-chip">📝 Build Summaries</div>
                <div class="feature-chip">📚 Create Notes</div>
                <div class="feature-chip">📥 Export Results</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.success(
        "Documents indexed successfully — your AI workspace is ready."
    )


# ============================================================
# Dashboard
# ============================================================
metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

with metric_col_1:
    st.metric("PDFs", pdf_count)

with metric_col_2:
    st.metric("Chunks", chunk_count)

with metric_col_3:
    st.metric("Embeddings", embedding_count)

with metric_col_4:
    st.metric("Questions", st.session_state.questions_asked)


# ============================================================
# Main workspace tabs
# ============================================================
overview_tab, summary_tab, notes_tab, ask_tab = st.tabs(
    [
        "📁 Overview",
        "📝 Summary",
        "📚 Study Notes",
        "💬 Ask AI",
    ]
)


# ------------------------------------------------------------
# Overview tab
# ------------------------------------------------------------
with overview_tab:
    st.markdown(
        """
        <div class="premium-card">
            <div class="section-kicker">WORKSPACE</div>
            <div class="section-title">Document overview</div>
            <div class="section-copy">
                Review the uploaded files and inspect the indexed content.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if documents_are_ready:
        left_column, right_column = st.columns([1, 1.35])

        with left_column:
            st.markdown("#### Uploaded files")
            for pdf_file in uploaded_files or []:
                st.write(f"📄 {pdf_file.name}")

        with right_column:
            with st.expander(
                "Preview first extracted chunk",
                expanded=True,
            ):
                st.text_area(
                    label="Indexed text preview",
                    value=st.session_state.all_chunks[0],
                    height=245,
                    disabled=True,
                    key="first_extracted_chunk_viewer",
                )
    else:
        st.info("Upload PDFs to populate the workspace.")


# ------------------------------------------------------------
# Summary tab
# ------------------------------------------------------------
with summary_tab:
    st.markdown(
        """
        <div class="premium-card">
            <div class="section-kicker">DOCUMENT INTELLIGENCE</div>
            <div class="section-title">Generate a polished summary</div>
            <div class="section-copy">
                Create a structured overview with key concepts and revision points.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "✨ Generate document summary",
        use_container_width=True,
        disabled=not documents_are_ready,
        key="generate_pdf_summary_button",
    ):
        try:
            with st.spinner(
                "Preparing a detailed summary from your documents..."
            ):
                from helpers.summary_generator import generate_pdf_summary

                generated_summary = generate_pdf_summary(
                    build_complete_document_text()
                )
                st.session_state.pdf_summary = generated_summary

        except Exception as error:
            st.error(format_user_friendly_error(error))

    if st.session_state.pdf_summary:
        st.markdown("### Generated summary")
        st.markdown(st.session_state.pdf_summary)

        try:
            from helpers.pdf_exporter import create_pdf

            summary_pdf = create_pdf(
                "NeuraDocs - PDF Summary",
                st.session_state.pdf_summary,
            )

            st.download_button(
                label="📥 Download summary as PDF",
                data=summary_pdf,
                file_name="NeuraDocs_PDF_Summary.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_summary_pdf_button",
            )

        except Exception as error:
            st.warning(
                "The summary was generated, but the PDF download "
                f"could not be prepared: {error}"
            )
    elif documents_are_ready:
        st.caption(
            "Generate a summary to view and download it here."
        )


# ------------------------------------------------------------
# Study notes tab
# ------------------------------------------------------------
with notes_tab:
    st.markdown(
        """
        <div class="premium-card">
            <div class="section-kicker">STUDY MODE</div>
            <div class="section-title">Create structured study notes</div>
            <div class="section-copy">
                Convert the uploaded material into organized, exam-friendly notes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "📚 Generate study notes",
        use_container_width=True,
        disabled=not documents_are_ready,
        key="generate_study_notes_button",
    ):
        try:
            with st.spinner(
                "Organizing your documents into structured study notes..."
            ):
                from helpers.notes_generator import generate_study_notes

                generated_notes = generate_study_notes(
                    build_complete_document_text()
                )
                st.session_state.study_notes = generated_notes

        except Exception as error:
            st.error(format_user_friendly_error(error))

    if st.session_state.study_notes:
        st.markdown("### Generated study notes")
        st.markdown(st.session_state.study_notes)

        try:
            from helpers.pdf_exporter import create_pdf

            notes_pdf = create_pdf(
                "NeuraDocs - Study Notes",
                st.session_state.study_notes,
            )

            st.download_button(
                label="📥 Download study notes as PDF",
                data=notes_pdf,
                file_name="NeuraDocs_Study_Notes.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_study_notes_pdf_button",
            )

        except Exception as error:
            st.warning(
                "The notes were generated, but the PDF download "
                f"could not be prepared: {error}"
            )
    elif documents_are_ready:
        st.caption(
            "Generate study notes to view and download them here."
        )


# ------------------------------------------------------------
# Ask AI tab
# ------------------------------------------------------------
with ask_tab:
    st.markdown(
        """
        <div class="premium-card">
            <div class="section-kicker">CONVERSATIONAL SEARCH</div>
            <div class="section-title">Ask anything from your PDFs</div>
            <div class="section-copy">
                NeuraDocs retrieves the most relevant chunks before generating
                a context-aware answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form(
        "ask_neuradocs_form",
        clear_on_submit=False,
    ):
        question = st.text_input(
            label="Your question",
            placeholder="Example: Explain the main concept discussed in the document.",
            disabled=not documents_are_ready,
            key="pdf_question_input",
        )

        ask_submitted = st.form_submit_button(
            "🚀 Ask NeuraDocs",
            type="primary",
            use_container_width=True,
            disabled=not documents_are_ready,
        )

    if ask_submitted:
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner(
                    "Retrieving relevant context and generating the answer..."
                ):
                    from helpers.embedding_model import create_query_embedding
                    from helpers.qa_chain import get_answer

                    question_embedding = create_query_embedding(question)

                    retrieval_results = search_similar_chunks(
                        question_embedding=question_embedding,
                        index=st.session_state.faiss_index,
                        chunks=st.session_state.all_chunks,
                        top_k=5,
                    )

                    if not retrieval_results:
                        st.warning(
                            "No sufficiently relevant content was found."
                        )
                    else:
                        retrieved_context = "\n\n".join(
                            result["chunk"]
                            for result in retrieval_results
                        )

                        generated_answer = get_answer(
                            question=question,
                            context=retrieved_context,
                        )

                        st.session_state.questions_asked += 1
                        st.session_state.chat_history.append(
                            {
                                "question": question,
                                "answer": generated_answer,
                                "sources": retrieval_results,
                            }
                        )
                        st.rerun()

            except Exception as error:
                st.error(format_user_friendly_error(error))

    if st.session_state.chat_history:
        chat_heading, chat_clear = st.columns([5, 1])

        with chat_heading:
            st.markdown("### Conversation")

        with chat_clear:
            if st.button(
                "Clear chat",
                use_container_width=True,
                key="clear_chat_button",
            ):
                st.session_state.chat_history = []
                st.rerun()

        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(chat["question"])

            with st.chat_message("assistant"):
                st.markdown(chat["answer"])

                if chat.get("sources"):
                    with st.expander("Sources used"):
                        for source_number, source in enumerate(
                            chat["sources"],
                            start=1,
                        ):
                            source_details = extract_source_details(
                                source["chunk"]
                            )
                            pdf_name = source_details.get(
                                "pdf_name",
                                "Unknown PDF",
                            )
                            chunk_number = source_details.get(
                                "original_chunk_number"
                            )
                            similarity_score = source.get(
                                "score",
                                0.0,
                            )

                            st.markdown(
                                f"**Source {source_number}**"
                            )
                            st.write(f"PDF: {pdf_name}")

                            if chunk_number is not None:
                                st.write(
                                    f"Chunk: {chunk_number}"
                                )

                            st.write(
                                f"Similarity score: "
                                f"{similarity_score:.3f}"
                            )

                            if source_number < len(
                                chat["sources"]
                            ):
                                st.markdown("---")
    elif documents_are_ready:
        st.caption(
            "Your answers and source citations will appear here."
        )


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer-card">
        © 2026 <b>NeuraDocs</b> · Built with Python, Streamlit,
        FAISS, Sentence Transformers and Generative AI
    </div>
    """,
    unsafe_allow_html=True,
)