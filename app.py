import streamlit as st

from helpers.pdf_loader import extract_text_from_pdf
from helpers.source_utils import extract_source_details
from helpers.text_splitter import split_text
from helpers.vector_store import create_faiss_index, search_similar_chunks


# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="NeuraDocs",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# Professional UI Theme
# ==================================================
st.markdown(
    """
    <style>
    /* Hide Streamlit default chrome */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* App background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 55%, #f8fafc 100%);
        color: #0f172a;
    }

    /* Main container */
    .block-container {
        max-width: 1240px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Main content text visibility */
    .block-container,
    .block-container p,
    .block-container li,
    .block-container label,
    .block-container span,
    .block-container div {
        color: #1f2937;
    }

    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4 {
        color: #172554;
    }

    .block-container strong,
    .block-container b {
        color: #0f172a;
    }

    /* Hero */
    .hero-card {
        background: linear-gradient(135deg, #172554 0%, #1d4ed8 55%, #4f46e5 100%);
        border-radius: 24px;
        padding: 34px 28px;
        margin-bottom: 24px;
        box-shadow: 0 18px 45px rgba(30, 64, 175, 0.24);
        text-align: center;
    }

    .hero-title {
        color: #ffffff !important;
        font-size: 46px;
        font-weight: 850;
        line-height: 1.1;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        color: #dbeafe !important;
        font-size: 18px;
        line-height: 1.6;
        margin-bottom: 12px;
    }

    .hero-tech {
        color: #bfdbfe !important;
        font-size: 14px;
        font-weight: 650;
        letter-spacing: 0.2px;
    }

    /* Welcome / feature card */
    .welcome-card {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 24px;
        margin: 8px 0 18px 0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
    }

    .welcome-card h3,
    .welcome-card p,
    .welcome-card li {
        color: #1f2937 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #172033 100%);
        border-right: 1px solid #273449;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #f8fafc !important;
    }

    .sidebar-brand {
        text-align: center;
        padding: 10px 4px 6px 4px;
    }

    .sidebar-brand-title {
        color: #ffffff !important;
        font-size: 27px;
        font-weight: 800;
    }

    .sidebar-brand-subtitle {
        color: #cbd5e1 !important;
        font-size: 13px;
        margin-top: 4px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 48px;
        border: none;
        border-radius: 12px;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: #ffffff !important;
        font-size: 15px;
        font-weight: 700;
        box-shadow: 0 7px 18px rgba(37, 99, 235, 0.22);
        transition: all 0.22s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 11px 24px rgba(37, 99, 235, 0.3);
    }

    .stButton > button:disabled {
        background: #94a3b8;
        color: #e2e8f0 !important;
        box-shadow: none;
    }

    .stButton > button p,
    .stButton > button span,
    .stDownloadButton > button p,
    .stDownloadButton > button span {
        color: #ffffff !important;
    }

    /* Download buttons */
    .stDownloadButton > button {
        width: 100%;
        min-height: 46px;
        border: none;
        border-radius: 12px;
        background: linear-gradient(90deg, #0f766e, #0891b2);
        color: #ffffff !important;
        font-weight: 700;
        transition: all 0.22s ease;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px);
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 18px;
        min-height: 122px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
        transition: all 0.22s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.12);
    }

    div[data-testid="stMetric"] label {
        color: #475569 !important;
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 850;
    }

    /* Input box */
    div[data-baseweb="input"] > div {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="input"] input {
        background: #ffffff !important;
        color: #111827 !important;
        caret-color: #2563eb !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    div[data-baseweb="input"] > div:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18) !important;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 14px;
        padding: 8px;
    }

    /* Expanders */
    details {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 4px;
    }

    details p,
    details span,
    details div,
    details li {
        color: #1f2937 !important;
    }

    /* Text area */
    textarea {
        color: #111827 !important;
        background: #ffffff !important;
    }

    /* Generated Markdown readability */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        color: #1f2937;
        line-height: 1.7;
    }

    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4 {
        color: #172554;
    }

    /* Chat */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 8px;
        margin-bottom: 12px;
        box-shadow: 0 5px 15px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] span {
        color: #1f2937 !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        color: #64748b !important;
        font-size: 13px;
        padding: 18px 0 8px 0;
    }

    .custom-footer b {
        color: #334155 !important;
    }

    /* ==================================================
       FINAL READABILITY FIXES
       Keep the light theme, but force every main-content text
       element to use a clearly visible dark colour.
       ================================================== */

    /* General text inside the main content area */
    [data-testid="stAppViewContainer"] .main p,
    [data-testid="stAppViewContainer"] .main li,
    [data-testid="stAppViewContainer"] .main label,
    [data-testid="stAppViewContainer"] .main small,
    [data-testid="stAppViewContainer"] .main code,
    [data-testid="stAppViewContainer"] .main pre,
    [data-testid="stAppViewContainer"] .main td,
    [data-testid="stAppViewContainer"] .main th {
        color: #111827 !important;
        opacity: 1 !important;
    }

    /* Streamlit Markdown generated from summaries, notes and answers */
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] li,
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] span,
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] strong,
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] em {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] h1,
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] h2,
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] h3,
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] h4 {
        color: #172554 !important;
        -webkit-text-fill-color: #172554 !important;
        opacity: 1 !important;
    }

    /* Question input: white field and dark typed text */
    [data-testid="stAppViewContainer"] .main div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        border: 1px solid #94a3b8 !important;
    }

    [data-testid="stAppViewContainer"] .main div[data-baseweb="input"] input {
        background-color: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #2563eb !important;
        opacity: 1 !important;
    }

    [data-testid="stAppViewContainer"] .main div[data-baseweb="input"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    /* Extracted chunk preview / all text areas, including disabled ones */
    [data-testid="stAppViewContainer"] .main div[data-testid="stTextArea"] textarea,
    [data-testid="stAppViewContainer"] .main div[data-testid="stTextArea"] textarea:disabled,
    [data-testid="stAppViewContainer"] .main textarea,
    [data-testid="stAppViewContainer"] .main textarea:disabled {
        background-color: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        border-color: #94a3b8 !important;
    }

    [data-testid="stAppViewContainer"] .main div[data-testid="stTextArea"] label,
    [data-testid="stAppViewContainer"] .main div[data-testid="stTextArea"] label p {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }

    /* Expanders: dark title and dark content */
    [data-testid="stAppViewContainer"] .main details[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }

    [data-testid="stAppViewContainer"] .main details[data-testid="stExpander"] summary,
    [data-testid="stAppViewContainer"] .main details[data-testid="stExpander"] summary p,
    [data-testid="stAppViewContainer"] .main details[data-testid="stExpander"] summary span,
    [data-testid="stAppViewContainer"] .main details[data-testid="stExpander"] summary svg,
    [data-testid="stAppViewContainer"] .main details[data-testid="stExpander"] div,
    [data-testid="stAppViewContainer"] .main details[data-testid="stExpander"] p,
    [data-testid="stAppViewContainer"] .main details[data-testid="stExpander"] span,
    [data-testid="stAppViewContainer"] .main details[data-testid="stExpander"] li {
        color: #111827 !important;
        fill: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }

    /* Alert messages */
    [data-testid="stAppViewContainer"] .main [data-testid="stAlert"] p,
    [data-testid="stAppViewContainer"] .main [data-testid="stAlert"] div,
    [data-testid="stAppViewContainer"] .main [data-testid="stAlert"] span {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }

    /* Chat cards */
    [data-testid="stAppViewContainer"] .main [data-testid="stChatMessage"] p,
    [data-testid="stAppViewContainer"] .main [data-testid="stChatMessage"] li,
    [data-testid="stAppViewContainer"] .main [data-testid="stChatMessage"] span {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }

    /* Metric labels and values */
    [data-testid="stAppViewContainer"] .main [data-testid="stMetricLabel"] p {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        opacity: 1 !important;
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stMetricValue"] {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
    }

    /* Preserve intentional white text on coloured controls */
    .hero-card .hero-title,
    .hero-card .hero-subtitle,
    .hero-card .hero-tech,
    .stButton > button,
    .stButton > button p,
    .stButton > button span,
    .stDownloadButton > button,
    .stDownloadButton > button p,
    .stDownloadButton > button span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .hero-card {
            padding: 26px 18px;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 35px;
        }

        .hero-subtitle {
            font-size: 15px;
        }

        .hero-tech {
            font-size: 12px;
        }
    }


    /* Code blocks / architecture diagrams inside generated Markdown */
    .block-container pre,
    .block-container pre code,
    .block-container [data-testid="stCodeBlock"],
    .block-container [data-testid="stCodeBlock"] code,
    .block-container [data-testid="stCodeBlock"] span {
        background: #111827 !important;
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        opacity: 1 !important;
    }

    .block-container pre {
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        padding: 18px !important;
        overflow-x: auto !important;
        line-height: 1.65 !important;
    }

    .block-container code:not(pre code) {
        background: #e2e8f0 !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Session State Initialization
# ==================================================
def clone_default(value):
    """Return a fresh copy for mutable default values."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


default_session_values = {
    "all_chunks": [],
    "faiss_index": None,
    "processed_file_signature": [],
    "chat_history": [],
    "pdf_summary": "",
    "study_notes": "",
    "questions_asked": 0,
}

for key, default_value in default_session_values.items():
    if key not in st.session_state:
        st.session_state[key] = clone_default(default_value)


# ==================================================
# Utility Functions
# ==================================================
def clear_application_data() -> None:
    """Clear processed documents and generated content."""
    st.session_state.all_chunks = []
    st.session_state.faiss_index = None
    st.session_state.processed_file_signature = []
    st.session_state.chat_history = []
    st.session_state.pdf_summary = ""
    st.session_state.study_notes = ""
    st.session_state.questions_asked = 0


def format_user_friendly_error(error: Exception) -> str:
    """Convert common technical errors into concise user-facing messages."""
    error_text = str(error)
    error_lower = error_text.lower()

    if "429" in error_text or "resource_exhausted" in error_lower or "quota" in error_lower:
        return (
            "Gemini API quota has been reached. Please try again after the quota "
            "resets or use a project with available quota."
        )

    if "api key" in error_lower or "google_api_key" in error_lower:
        return (
            "The Gemini API key is missing or invalid. Please verify the "
            "GOOGLE_API_KEY configuration."
        )

    if "503" in error_text or "unavailable" in error_lower or "high demand" in error_lower:
        return (
            "Gemini is temporarily unavailable because of high demand. "
            "Please wait briefly and try again."
        )

    if "huggingface" in error_lower or "couldn't connect" in error_lower:
        return (
            "The embedding model could not be downloaded. Please check the "
            "deployment connection and model configuration."
        )

    return f"An unexpected error occurred: {error_text}"


def build_complete_document_text() -> str:
    """Combine all stored chunks into one document string."""
    return "\n\n".join(st.session_state.all_chunks)


# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">🧠 NeuraDocs</div>
            <div class="sidebar-brand-subtitle">
                AI PDF Study Assistant • Version 1.0
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    uploaded_files = st.file_uploader(
        label="📂 Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based PDF files.",
    )

    st.markdown("---")

    st.info(
        "Upload PDFs to ask questions, create summaries, generate study "
        "notes, and download the results."
    )

    if st.button(
        "🗑️ Clear Processed Data",
        use_container_width=True,
        key="clear_processed_data_button",
    ):
        clear_application_data()
        st.success("Processed data cleared successfully.")
        st.rerun()


# ==================================================
# Hero Section
# ==================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🧠 NeuraDocs</div>
        <div class="hero-subtitle">
            AI-powered PDF question answering, intelligent summarization,
            and structured study notes in one place.
        </div>
        <div class="hero-tech">
            Powered by Gemini AI • FAISS • RAG • Sentence Transformers
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# PDF Processing
# ==================================================
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
            "📄 NeuraDocs is reading your PDFs and building the search index..."
        ):
            collected_chunks = []

            for pdf_file in uploaded_files:
                try:
                    pdf_text = extract_text_from_pdf(pdf_file)

                    if not pdf_text or not pdf_text.strip():
                        st.warning(
                            f"No readable text was found in {pdf_file.name}. "
                            "The PDF may be scanned or image-based."
                        )
                        continue

                    pdf_chunks = split_text(pdf_text)

                    if not pdf_chunks:
                        st.warning(
                            f"No text chunks could be created from {pdf_file.name}."
                        )
                        continue

                    for chunk_number, chunk in enumerate(pdf_chunks, start=1):
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

                    document_embeddings = create_embeddings(collected_chunks)
                    vector_index = create_faiss_index(document_embeddings)

                    st.session_state.all_chunks = collected_chunks
                    st.session_state.faiss_index = vector_index
                    st.session_state.processed_file_signature = current_file_signature

                    # Clear outputs generated for previous files.
                    st.session_state.chat_history = []
                    st.session_state.pdf_summary = ""
                    st.session_state.study_notes = ""
                    st.session_state.questions_asked = 0

                except Exception as error:
                    st.error(format_user_friendly_error(error))
            else:
                clear_application_data()


# ==================================================
# Document Ready Check
# ==================================================
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


# ==================================================
# Empty State
# ==================================================
if not documents_are_ready:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 Welcome to NeuraDocs</h3>
            <p>Upload one or more PDF files from the sidebar to begin.</p>
            <ul>
                <li>Ask questions directly from your PDFs</li>
                <li>Generate detailed document summaries</li>
                <li>Create structured study notes</li>
                <li>Download generated content as PDF</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# Dashboard
# ==================================================
st.subheader("📊 Dashboard")

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

with metric_col_1:
    st.metric("📄 PDFs", pdf_count)

with metric_col_2:
    st.metric("🧩 Chunks", chunk_count)

with metric_col_3:
    st.metric("🧠 Embeddings", embedding_count)

with metric_col_4:
    st.metric("💬 Questions", st.session_state.questions_asked)


# ==================================================
# Processed Document Information
# ==================================================
if documents_are_ready:
    st.success(
        "✅ Your documents are ready. You can now ask questions, "
        "generate a summary, or create study notes."
    )

    st.subheader("📁 Uploaded Files")

    for pdf_file in uploaded_files or []:
        st.write(f"📄 {pdf_file.name}")

    with st.expander("🔎 View First Extracted Chunk"):
        st.text_area(
            label="First Chunk",
            value=st.session_state.all_chunks[0],
            height=220,
            disabled=True,
            key="first_extracted_chunk_viewer",
        )


# ==================================================
# PDF Summary Generator
# ==================================================
st.markdown("---")
st.subheader("📝 PDF Summary")

if st.button(
    "📄 Generate Summary",
    use_container_width=True,
    disabled=not documents_are_ready,
    key="generate_pdf_summary_button",
):
    try:
        with st.spinner("🧠 NeuraDocs is preparing a detailed summary..."):
            from helpers.summary_generator import generate_pdf_summary

            generated_summary = generate_pdf_summary(
                build_complete_document_text()
            )
            st.session_state.pdf_summary = generated_summary

    except Exception as error:
        st.error(format_user_friendly_error(error))

if st.session_state.pdf_summary:
    st.markdown("### 📄 Generated Summary")
    st.markdown(st.session_state.pdf_summary)

    try:
        from helpers.pdf_exporter import create_pdf

        summary_pdf = create_pdf(
            "NeuraDocs - PDF Summary",
            st.session_state.pdf_summary,
        )

        st.download_button(
            label="📥 Download Summary as PDF",
            data=summary_pdf,
            file_name="NeuraDocs_PDF_Summary.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_summary_pdf_button",
        )
    except Exception as error:
        st.warning(
            "The summary was generated, but the PDF download could not be "
            f"prepared: {error}"
        )


# ==================================================
# Study Notes Generator
# ==================================================
st.markdown("---")
st.subheader("📚 AI Study Notes")

if st.button(
    "📚 Generate Study Notes",
    use_container_width=True,
    disabled=not documents_are_ready,
    key="generate_study_notes_button",
):
    try:
        with st.spinner("🧠 NeuraDocs is creating structured study notes..."):
            from helpers.notes_generator import generate_study_notes

            generated_notes = generate_study_notes(
                build_complete_document_text()
            )
            st.session_state.study_notes = generated_notes

    except Exception as error:
        st.error(format_user_friendly_error(error))

if st.session_state.study_notes:
    st.markdown("### 📝 Generated Study Notes")
    st.markdown(st.session_state.study_notes)

    try:
        from helpers.pdf_exporter import create_pdf

        notes_pdf = create_pdf(
            "NeuraDocs - Study Notes",
            st.session_state.study_notes,
        )

        st.download_button(
            label="📥 Download Study Notes as PDF",
            data=notes_pdf,
            file_name="NeuraDocs_Study_Notes.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_study_notes_pdf_button",
        )
    except Exception as error:
        st.warning(
            "The study notes were generated, but the PDF download could not "
            f"be prepared: {error}"
        )


# ==================================================
# Question Answering Section
# ==================================================
st.markdown("---")
st.subheader("💬 Ask a Question From Your PDFs")

question = st.text_input(
    label="Enter your question",
    placeholder="Example: What is the main concept explained in this PDF?",
    disabled=not documents_are_ready,
    key="pdf_question_input",
)

if st.button(
    "🚀 Ask NeuraDocs",
    type="primary",
    use_container_width=True,
    disabled=not documents_are_ready,
    key="ask_neuradocs_button",
):
    if not question or not question.strip():
        st.warning("Please enter a question before asking NeuraDocs.")
    else:
        try:
            with st.spinner("🧠 NeuraDocs is analyzing the relevant content..."):
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
                        "No sufficiently relevant content was found for this question."
                    )
                else:
                    retrieved_context = "\n\n".join(
                        result["chunk"] for result in retrieval_results
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


# ==================================================
# ChatGPT-Style Conversation
# ==================================================
if st.session_state.chat_history:
    st.markdown("---")

    heading_column, clear_chat_column = st.columns([4, 1])

    with heading_column:
        st.subheader("💬 NeuraDocs Conversation")

    with clear_chat_column:
        if st.button(
            "🗑️ Clear Chat",
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
                with st.expander("📚 Sources Used"):
                    for source_number, source in enumerate(
                        chat["sources"],
                        start=1,
                    ):
                        source_details = extract_source_details(source["chunk"])

                        pdf_name = source_details.get("pdf_name", "Unknown PDF")
                        chunk_number = source_details.get(
                            "original_chunk_number"
                        )
                        similarity_score = source.get("score", 0.0)

                        st.markdown(f"#### Source {source_number}")
                        st.write(f"**PDF:** {pdf_name}")

                        if chunk_number is not None:
                            st.write(f"**Chunk:** {chunk_number}")

                        st.write(
                            f"**Similarity Score:** {similarity_score:.3f}"
                        )

                        if source_number < len(chat["sources"]):
                            st.markdown("---")


# ==================================================
# Footer
# ==================================================
st.markdown("---")
st.markdown(
    """
    <div class="custom-footer">
        © 2026 <b>NeuraDocs</b><br>
        Built with <b>Python</b> • <b>Streamlit</b> •
        <b>FAISS</b> • <b>Gemini AI</b>
    </div>
    """,
    unsafe_allow_html=True,
)