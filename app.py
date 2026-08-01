import streamlit as st

from helpers.pdf_loader import extract_text_from_pdf
from helpers.text_splitter import split_text
from helpers.vector_store import (
    create_faiss_index,
    search_similar_chunks,
)
from helpers.source_utils import extract_source_details


# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide",
)
st.markdown(
    """
    <style>
    /* Hide Streamlit default chrome */
    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }

    :root {
        color-scheme: light;
    }

    html,
    body,
    [data-testid="stAppViewContainer"],
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 100%
        ) !important;
        color: #172033 !important;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1e3a8a !important;
        text-align: center;
        margin-bottom: 8px;
    }

    .sub-title {
        text-align: center;
        color: #64748b !important;
        font-size: 17px;
        line-height: 1.7;
        margin-bottom: 10px;
    }

    .tech-line {
        text-align: center;
        color: #475569 !important;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 30px;
    }

    /* Main-page text: force readable dark text on light background */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] li,
    [data-testid="stMainBlockContainer"] label,
    [data-testid="stMainBlockContainer"] span:not(button span),
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em,
    [data-testid="stMarkdownContainer"] code {
        color: #172033 !important;
    }

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6,
    .block-container h1,
    .block-container h2,
    .block-container h3 {
        color: #102a56 !important;
    }

    [data-testid="stMarkdownContainer"] a {
        color: #2563eb !important;
    }

    [data-testid="stMarkdownContainer"] blockquote {
        color: #334155 !important;
        border-left-color: #64748b !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #172033 100%
        ) !important;
        border-right: 1px solid #243047;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #f8fafc !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: #ffffff !important;
        font-size: 15px;
        font-weight: 650;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.22);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.30);
    }

    .stButton > button *,
    .stDownloadButton > button * {
        color: #ffffff !important;
    }

    .stDownloadButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #0f766e, #0891b2);
        color: #ffffff !important;
        font-weight: 650;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        min-height: 130px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        transition: all 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.14);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label * {
        color: #475569 !important;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #0f172a !important;
        font-weight: 800;
    }

    /* Inputs */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] {
        border-radius: 12px;
        border-color: #cbd5e1;
        background: #ffffff !important;
    }

    input,
    textarea {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.30);
        border-radius: 14px;
        padding: 8px;
    }

    /* Expanders and chat messages */
    details,
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
    }

    details {
        padding: 5px;
    }

    div[data-testid="stChatMessage"] {
        padding: 8px;
        margin-bottom: 12px;
    }

    details *,
    div[data-testid="stChatMessage"] * {
        color: #172033 !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Horizontal separators */
    hr {
        border-color: #cbd5e1 !important;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        color: #64748b !important;
        font-size: 14px;
        padding-top: 25px;
        padding-bottom: 15px;
    }

    .custom-footer b {
        color: #334155 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# Session State Initialization
# ==================================================
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
        st.session_state[key] = default_value


# ==================================================
# Clear Application Data
# ==================================================
def clear_application_data() -> None:
    """
    Clear processed PDF data and generated content.
    """

    st.session_state.all_chunks = []
    st.session_state.faiss_index = None
    st.session_state.processed_file_signature = []
    st.session_state.chat_history = []
    st.session_state.pdf_summary = ""
    st.session_state.study_notes = ""
    st.session_state.questions_asked = 0


# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    st.title("🤖 AI PDF Assistant")

    st.markdown("---")

    uploaded_files = st.file_uploader(
        label="📂 Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.markdown("---")

    st.info(
        "Upload one or more text-based PDF files and use AI "
        "to ask questions, generate summaries, and create notes."
    )

    if st.button(
        "🗑️ Clear Processed Data",
        use_container_width=True,
    ):
        clear_application_data()
        st.success("Processed data cleared successfully.")
        st.rerun()


# ==================================================
# Main Heading
# ==================================================
st.markdown(
    """
    <div class="main-title">
        🤖 AI PDF Assistant
    </div>

    <div class="sub-title">
        Upload PDF documents and use AI for question answering,
        intelligent summarization, and structured study notes.
    </div>

    <div class="tech-line">
        Powered by Gemini AI • FAISS • RAG • Sentence Transformers
    </div>
    """,
    unsafe_allow_html=True,
)
# ==================================================
# PDF Processing
# ==================================================
if uploaded_files:
    current_file_signature = [
        (
            uploaded_file.name,
            uploaded_file.size,
        )
        for uploaded_file in uploaded_files
    ]

    files_have_changed = (
        current_file_signature
        != st.session_state.processed_file_signature
    )

    if files_have_changed:
        with st.spinner(
            "Extracting PDF text and preparing the document index..."
        ):
            collected_chunks = []

            for pdf_file in uploaded_files:
                try:
                    pdf_text = extract_text_from_pdf(pdf_file)

                    if not pdf_text or not pdf_text.strip():
                        st.warning(
                            f"No readable text was found in "
                            f"{pdf_file.name}. The PDF may be scanned "
                            "or image-based."
                        )
                        continue

                    pdf_chunks = split_text(pdf_text)

                    if not pdf_chunks:
                        st.warning(
                            f"No text chunks could be created from "
                            f"{pdf_file.name}."
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
                        f"An error occurred while processing "
                        f"{pdf_file.name}: {error}"
                    )

            if collected_chunks:
                try:
                    # Lazy import to improve initial page loading.
                    from helpers.embedding_model import (
                        create_embeddings,
                    )

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

                    # Clear previously generated content
                    st.session_state.chat_history = []
                    st.session_state.pdf_summary = ""
                    st.session_state.study_notes = ""

                except Exception as error:
                    st.error(
                        "An error occurred while creating document "
                        f"embeddings or the FAISS index: {error}"
                    )

            else:
                clear_application_data()


# ==================================================
# Document Ready Check
# ==================================================
documents_are_ready = (
    bool(st.session_state.all_chunks)
    and st.session_state.faiss_index is not None
)


# ==================================================
# Processed Document Information
# ==================================================
if documents_are_ready:
    st.success("✅ Documents indexed successfully.")

    st.subheader("📁 Uploaded Files")

    if uploaded_files:
        for pdf_file in uploaded_files:
            st.write(f"📄 {pdf_file.name}")

    metric_column_1, metric_column_2 = st.columns(2)

    with metric_column_1:
        st.metric(
            label="Total Chunks",
            value=len(st.session_state.all_chunks),
        )

    with metric_column_2:
        st.metric(
            label="FAISS Vectors Stored",
            value=st.session_state.faiss_index.ntotal,
        )

    with st.expander("🔎 View First Extracted Chunk"):
        st.text_area(
            label="First Chunk",
            value=st.session_state.all_chunks[0],
            height=220,
            disabled=True,
        )
st.markdown("---")

st.subheader("📊 Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📄 PDFs",
        len(uploaded_files)
    )

with col2:
    st.metric(
        "📑 Chunks",
        len(st.session_state.all_chunks)
    )

with col3:
    st.metric(
        "🧠 Embeddings",
        (
            st.session_state.faiss_index.ntotal
            if st.session_state.faiss_index is not None
            else 0
        ),
    )

with col4:
    st.metric(
        "💬 Questions",
        st.session_state.questions_asked
    )


# ==================================================
# PDF Summary Generator
# ==================================================
st.markdown("---")

st.subheader("📝 PDF Summary")

if st.button(
    "✨ Generate PDF Summary",
    use_container_width=True,
    disabled=not documents_are_ready,
):
    try:
        with st.spinner(
            "Generating a detailed PDF summary..."
        ):
            from helpers.summary_generator import (
                generate_pdf_summary,
            )

            complete_document_text = "\n\n".join(
                st.session_state.all_chunks
            )

            generated_summary = generate_pdf_summary(
                complete_document_text
            )

            st.session_state.pdf_summary = generated_summary

    except Exception as error:
        st.error(
            "An error occurred while generating the PDF summary: "
            f"{error}"
        )

if st.session_state.pdf_summary:
    st.markdown("### 📄 Generated Summary")

    st.markdown(
        st.session_state.pdf_summary
    )

    from helpers.pdf_exporter import create_pdf

    summary_pdf = create_pdf(
        "PDF Summary",
        st.session_state.pdf_summary,
    )

    st.download_button(
        label="📥 Download Summary as PDF",
        data=summary_pdf,
        file_name="PDF_Summary.pdf",
        mime="application/pdf",
        use_container_width=True,
        key="download_summary_pdf",
    )


# ==================================================
# Study Notes Generator
# ==================================================
st.markdown("---")

st.subheader("📚 AI Study Notes")

if st.button(
    "📝 Generate Study Notes",
    use_container_width=True,
    disabled=not documents_are_ready,
):
    try:
        with st.spinner(
            "Generating structured study notes..."
        ):
            from helpers.notes_generator import (
                generate_study_notes,
            )

            complete_document_text = "\n\n".join(
                st.session_state.all_chunks
            )

            generated_notes = generate_study_notes(
                complete_document_text
            )

            st.session_state.study_notes = generated_notes

    except Exception as error:
        st.error(
            "An error occurred while generating study notes: "
            f"{error}"
        )

if st.session_state.study_notes:
    st.markdown("### 📝 Generated Study Notes")

    st.markdown(
        st.session_state.study_notes
    )

    from helpers.pdf_exporter import create_pdf

    study_notes_pdf = create_pdf(
        "Study Notes",
        st.session_state.study_notes,
    )

    st.download_button(
        label="📥 Download Study Notes as PDF",
        data=study_notes_pdf,
        file_name="Study_Notes.pdf",
        mime="application/pdf",
        use_container_width=True,
        key="download_notes_pdf",
    )


# ==================================================
# Question Answering Section
# ==================================================
st.markdown("---")

st.subheader("💬 Ask a Question From the PDF")

question = st.text_input(
    label="Enter your question",
    placeholder="Ask anything from your uploaded PDF...",
)


# ==================================================
# Generate AI Answer
# ==================================================
if st.button(
    "✨ Ask AI",
    type="primary",
):
    if not uploaded_files:
        st.warning(
            "Please upload at least one PDF file."
        )

    elif not st.session_state.all_chunks:
        st.warning(
            "No readable content was extracted from the uploaded PDF."
        )

    elif st.session_state.faiss_index is None:
        st.warning(
            "The document search index is unavailable. "
            "Please upload the PDF again."
        )

    elif not question or not question.strip():
        st.warning(
            "Please enter a question."
        )

    else:
        try:
            with st.spinner("🤖 AI is thinking..."):
                from helpers.embedding_model import (
                    create_query_embedding,
                )
                from helpers.qa_chain import get_answer

                question_embedding = create_query_embedding(
                    question
                )

                retrieval_results = search_similar_chunks(
                    question_embedding=question_embedding,
                    index=st.session_state.faiss_index,
                    chunks=st.session_state.all_chunks,
                    top_k=5,
                )

                if not retrieval_results:
                    st.warning(
                        "No relevant content was found for this question."
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
            st.error(
                "An error occurred while generating the answer: "
                f"{error}"
            )


# ==================================================
# ChatGPT-Style Conversation
# ==================================================
if st.session_state.chat_history:
    st.markdown("---")

    heading_column, clear_chat_column = st.columns(
        [4, 1]
    )

    with heading_column:
        st.subheader("💬 AI Conversation")

    with clear_chat_column:
        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
        ):
            st.session_state.chat_history = []
            st.rerun()

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])

        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

            if chat["sources"]:
                with st.expander("📚 Sources Used"):
                    for source_number, source in enumerate(
                        chat["sources"],
                        start=1,
                    ):
                        source_details = extract_source_details(
                            source["chunk"]
                        )

                        pdf_name = source_details["pdf_name"]

                        chunk_number = source_details[
                            "original_chunk_number"
                        ]

                        similarity_score = source["score"]

                        st.markdown(
                            f"### Source {source_number}"
                        )

                        st.write(
                            f"**PDF:** {pdf_name}"
                        )

                        if chunk_number is not None:
                            st.write(
                                f"**Chunk:** {chunk_number}"
                            )

                        st.write(
                            f"**Similarity:** "
                            f"{similarity_score:.3f}"
                        )

                        st.markdown("---")
                        st.markdown("---")

st.markdown("---")

st.markdown(
    """
    <div class="custom-footer">
        © 2026 AI PDF Assistant<br>
        Built with <b>Python</b> • <b>Streamlit</b> •
        <b>FAISS</b> • <b>Gemini AI</b>
    </div>
    """,
    unsafe_allow_html=True,
)