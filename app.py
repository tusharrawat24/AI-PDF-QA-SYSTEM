import json
import re
import time

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
    page_icon="ðŸ§ ",
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
    "quiz_data": [],
    "quiz_answers": {},
    "quiz_submitted": False,
    "flashcards": [],
    "important_questions": "",
    "study_pack": {},
    "last_similarity_average": 0.0,
    "bookmarks": [],
    "ai_tools_output": {},
    "document_search_results": [],
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
    st.session_state.quiz_data = []
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.flashcards = []
    st.session_state.important_questions = ""
    st.session_state.study_pack = {}
    st.session_state.last_similarity_average = 0.0
    st.session_state.bookmarks = []
    st.session_state.ai_tools_output = {}
    st.session_state.document_search_results = []

    if "pdf_question_input" in st.session_state:
        st.session_state.pdf_question_input = ""


def build_complete_document_text() -> str:
    """Combine all stored chunks into a single document string."""
    return "\n\n".join(st.session_state.all_chunks)


def extract_json_payload(raw_text: str):
    """Extract and decode JSON from an AI response."""
    if not raw_text or not raw_text.strip():
        raise ValueError("The AI provider returned an empty response.")

    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [p for p in (cleaned.find("{"), cleaned.find("[")) if p >= 0]
        if not starts:
            raise ValueError("No JSON payload was found in the AI response.")
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end <= start:
            raise ValueError("The JSON payload was incomplete.")
        return json.loads(cleaned[start:end + 1])


def generate_quiz(document_text: str, question_count: int = 10):
    from helpers.gemini_client import generate_content
    prompt = f"""
Create exactly {question_count} MCQs using ONLY the PDF content below.
Return ONLY valid JSON as a list of objects with keys:
question, options (exactly 4), answer_index (0-3), explanation.
Avoid duplicate questions and mix difficulty levels.

PDF Content:
{document_text}
"""
    payload = extract_json_payload(generate_content(prompt))
    if not isinstance(payload, list):
        raise ValueError("Quiz response must be a JSON list.")
    quiz = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        options = item.get("options", [])
        answer_index = item.get("answer_index")
        explanation = str(item.get("explanation", "")).strip()
        if q and isinstance(options, list) and len(options) == 4 and isinstance(answer_index, int) and answer_index in range(4):
            quiz.append({"question": q, "options": [str(x).strip() for x in options], "answer_index": answer_index, "explanation": explanation})
    if not quiz:
        raise ValueError("No valid quiz questions were generated.")
    return quiz


def generate_flashcards(document_text: str, card_count: int = 12):
    from helpers.gemini_client import generate_content
    prompt = f"""
Create exactly {card_count} revision flashcards using ONLY the PDF content.
Return ONLY valid JSON: [{{"front":"Question or term","back":"Concise answer"}}].
Avoid duplicates and cover the most important concepts.

PDF Content:
{document_text}
"""
    payload = extract_json_payload(generate_content(prompt))
    if not isinstance(payload, list):
        raise ValueError("Flashcard response must be a JSON list.")
    cards = []
    for item in payload:
        if isinstance(item, dict):
            front = str(item.get("front", "")).strip()
            back = str(item.get("back", "")).strip()
            if front and back:
                cards.append({"front": front, "back": back})
    if not cards:
        raise ValueError("No valid flashcards were generated.")
    return cards


def generate_important_questions(document_text: str) -> str:
    from helpers.gemini_client import generate_content
    prompt = f"""
Using ONLY the PDF content, generate clean Markdown sections:
## 2-Mark Questions (8)
## 5-Mark Questions (6)
## 10-Mark Questions (4)
## Viva Questions (10 with one-line answers)

PDF Content:
{document_text}
"""
    return generate_content(prompt).strip()


def generate_study_pack(document_text: str):
    from helpers.gemini_client import generate_content
    prompt = f"""
Create a complete study pack using ONLY the PDF content.
Return ONLY valid JSON with keys: overview, revision_notes (8 strings),
key_terms (8 term/definition objects), flashcards (8 front/back objects),
quiz (5 objects containing question, options[4], answer_index, explanation),
and viva_questions (8 question/answer objects).

PDF Content:
{document_text}
"""
    payload = extract_json_payload(generate_content(prompt))
    if not isinstance(payload, dict):
        raise ValueError("Study pack response must be a JSON object.")
    return payload


def estimate_document_words() -> int:
    return len(build_complete_document_text().split())


def generate_ai_tool_output(document_text: str, tool_name: str) -> str:
    """Generate specialized learning content from the uploaded PDFs."""
    from helpers.gemini_client import generate_content

    tool_prompts = {
        "Table of Contents": """
Create a clean hierarchical table of contents from the PDF.
Use Markdown headings and numbered chapters/sections.
Do not invent sections that are not supported by the document.
""",
        "Mind Map": """
Create a text-based mind map of the PDF's major concepts.
Use this format:
Main Topic
â”œâ”€â”€ Branch
â”‚   â”œâ”€â”€ Subtopic
â”‚   â””â”€â”€ Subtopic
â””â”€â”€ Branch
Return the mind map inside one Markdown code block and add a short explanation below it.
""",
        "Concept Map": """
Create a structured concept map showing important concepts and their relationships.
Use arrows such as: Concept A â†’ leads to â†’ Concept B.
Use only information from the PDF.
""",
        "Timeline": """
Create a chronological timeline from the PDF.
If the PDF contains no meaningful dates or sequence of events, clearly say that a timeline is not applicable.
""",
        "Formula Sheet": """
Extract and organize every important formula, equation, symbol, and short explanation from the PDF.
Do not create formulas that are not present.
""",
        "Code Examples": """
Extract programming concepts from the PDF and create helpful code examples only where supported by the document.
If the PDF is not about programming, clearly state that code examples are not applicable.
""",
        "Diagram Ideas": """
Suggest clear educational diagrams that can be drawn from the PDF content.
For each diagram provide: title, components, connections, and what it explains.
Do not claim that an actual image has been generated.
""",
    }

    instruction = tool_prompts[tool_name]

    prompt = f"""
You are an academic document assistant.

{instruction}

Use ONLY the PDF content below. Keep the output accurate, structured,
and suitable for university students.

PDF Content:
{document_text}
"""

    return generate_content(prompt).strip()


def build_chat_export_text() -> str:
    """Build a plain-text transcript of the current conversation."""
    lines = ["NeuraDocs Conversation Export", "=" * 32, ""]

    for index, chat in enumerate(st.session_state.chat_history, start=1):
        lines.append(f"Question {index}:")
        lines.append(chat.get("question", ""))
        lines.append("")
        lines.append("Answer:")
        lines.append(chat.get("answer", ""))
        lines.append("")
        lines.append("-" * 32)
        lines.append("")

    return "\\n".join(lines)


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
            <div class="side-logo">ðŸ§ </div>
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
        "ðŸ—‘ï¸ Clear workspace",
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

    /* ULTRA PREMIUM VISUAL LAYER */
    .stApp::before,
    .stApp::after {{
        content: "";
        position: fixed;
        width: 420px;
        height: 420px;
        border-radius: 50%;
        filter: blur(95px);
        opacity: 0.18;
        pointer-events: none;
        z-index: 0;
        animation: floatGlow 12s ease-in-out infinite alternate;
    }}

    .stApp::before {{
        top: -130px;
        left: -130px;
        background: #3b82f6;
    }}

    .stApp::after {{
        right: -150px;
        bottom: -150px;
        background: #8b5cf6;
        animation-delay: 2s;
    }}

    @keyframes floatGlow {{
        from {{ transform: translate3d(0, 0, 0) scale(1); }}
        to {{ transform: translate3d(35px, 24px, 0) scale(1.08); }}
    }}

    .block-container {{
        position: relative;
        z-index: 1;
    }}

    .hero-shell {{
        padding: 0 !important;
        border: 1px solid rgba(255,255,255,0.14);
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        box-shadow: 0 28px 70px rgba(30,64,175,0.28), inset 0 1px 0 rgba(255,255,255,0.18);
    }}

    .hero-grid {{
        display: grid;
        grid-template-columns: 1.25fr 0.75fr;
        gap: 24px;
        align-items: center;
        padding: 40px;
        min-height: 360px;
    }}

    .hero-left {{
        position: relative;
        z-index: 2;
    }}

    .hero-right {{
        position: relative;
        min-height: 280px;
        display: grid;
        place-items: center;
    }}

    .hero-highlight {{
        background: linear-gradient(90deg, #ffffff 0%, #bfdbfe 35%, #ddd6fe 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
        -webkit-text-fill-color: transparent !important;
    }}

    .hero-ai-card {{
        position: relative;
        z-index: 2;
        width: min(100%, 330px);
        padding: 20px;
        border-radius: 22px;
        background: rgba(9,18,37,0.58);
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 20px 50px rgba(2,6,23,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        animation: cardFloat 5.5s ease-in-out infinite;
    }}

    @keyframes cardFloat {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-8px); }}
    }}

    .hero-ai-top {{
        display: flex;
        align-items: center;
        gap: 9px;
    }}

    .hero-ai-dot {{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 18px #34d399;
    }}

    .hero-ai-label {{
        color: #c7d2fe !important;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.12em;
    }}

    .hero-ai-line {{
        height: 1px;
        margin: 15px 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.28), transparent);
    }}

    .hero-ai-stat {{
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 11px 0;
    }}

    .hero-ai-stat span {{
        color: #93c5fd !important;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    .hero-ai-stat b {{
        color: #ffffff !important;
        font-size: 14px;
    }}

    .orb {{
        position: absolute;
        border-radius: 50%;
        filter: blur(2px);
    }}

    .orb-one {{
        width: 170px;
        height: 170px;
        top: 5px;
        right: 10px;
        background: radial-gradient(circle, rgba(96,165,250,0.45), transparent 70%);
        animation: orbitOne 7s ease-in-out infinite alternate;
    }}

    .orb-two {{
        width: 130px;
        height: 130px;
        left: 0;
        bottom: 0;
        background: radial-gradient(circle, rgba(167,139,250,0.42), transparent 70%);
        animation: orbitTwo 8s ease-in-out infinite alternate;
    }}

    @keyframes orbitOne {{
        from {{ transform: translate(0, 0) scale(1); }}
        to {{ transform: translate(-18px, 18px) scale(1.08); }}
    }}

    @keyframes orbitTwo {{
        from {{ transform: translate(0, 0) scale(1); }}
        to {{ transform: translate(20px, -10px) scale(1.12); }}
    }}

    .top-feature-strip {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 16px 0 22px;
    }}

    .top-feature-item {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 16px;
        border-radius: 16px;
        background: {theme['surface']} !important;
        border: 1px solid {theme['border']} !important;
        box-shadow: 0 10px 26px {theme['shadow']};
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .top-feature-item:hover {{
        transform: translateY(-3px);
        border-color: #60a5fa !important;
    }}

    .top-feature-icon {{
        font-size: 22px;
    }}

    .top-feature-item b {{
        display: block;
        color: {theme['heading']} !important;
        font-size: 13px;
    }}

    .top-feature-item small {{
        color: {theme['muted']} !important;
        font-size: 11px;
    }}

    .premium-card,
    div[data-testid="stMetric"],
    details[data-testid="stExpander"],
    [data-testid="stChatMessage"] {{
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        animation: fadeUp 0.45s ease both;
    }}

    .premium-card {{
        position: relative;
        overflow: hidden;
    }}

    .premium-card::before {{
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: linear-gradient(180deg, #2563eb, #7c3aed);
    }}

    div[data-testid="stMetric"] {{
        overflow: hidden;
        position: relative;
    }}

    div[data-testid="stMetric"]::after {{
        content: "";
        position: absolute;
        width: 95px;
        height: 95px;
        right: -38px;
        top: -38px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59,130,246,0.16), transparent 70%);
    }}

    .stTabs [data-baseweb="tab-list"] {{
        position: sticky;
        top: 0.55rem;
        z-index: 20;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        overflow-x: auto;
        scrollbar-width: none;
    }}

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
        display: none;
    }}

    .stTabs [data-baseweb="tab"] {{
        transition: all 0.2s ease;
        white-space: nowrap;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(59,130,246,0.08);
    }}

    section[data-testid="stSidebar"] {{
        box-shadow: 18px 0 45px rgba(2,6,23,0.22);
    }}

    .side-brand {{
        padding: 10px 0 22px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }}

    .stButton > button,
    .stDownloadButton > button {{
        position: relative;
        overflow: hidden;
    }}

    .stButton > button::after,
    .stDownloadButton > button::after {{
        content: "";
        position: absolute;
        top: 0;
        left: -110%;
        width: 55%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
        transform: skewX(-20deg);
        transition: left 0.55s ease;
    }}

    .stButton > button:hover::after,
    .stDownloadButton > button:hover::after {{
        left: 140%;
    }}

    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    [data-testid="stChatMessage"] {{
        border-radius: 20px !important;
        padding: 10px 12px !important;
    }}

    @media (max-width: 900px) {{
        .hero-grid {{
            grid-template-columns: 1fr;
            padding: 28px 22px;
        }}

        .hero-right {{
            min-height: 220px;
        }}

        .top-feature-strip {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}

    @media (max-width: 560px) {{
        .hero-title {{
            font-size: 34px;
        }}

        .top-feature-strip {{
            grid-template-columns: 1fr;
        }}

        .hero-ai-card {{
            width: 100%;
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
        <div class="hero-grid">
            <div class="hero-left">
                <div class="hero-badge">NEXT-GEN AI PDF WORKSPACE</div>
                <div class="hero-title">
                    Understand documents.<br>
                    <span class="hero-highlight">Study smarter.</span>
                </div>
                <div class="hero-copy">
                    Turn complex PDFs into answers, notes, quizzes, flashcards,
                    mind maps, and revision packs â€” all from one intelligent workspace.
                </div>
                <div class="hero-pills">
                    <div class="hero-pill">âš¡ Semantic Search</div>
                    <div class="hero-pill">ðŸ§  RAG Intelligence</div>
                    <div class="hero-pill">ðŸŽ“ Study Mode</div>
                    <div class="hero-pill">ðŸ“„ Smart Export</div>
                </div>
            </div>
            <div class="hero-right">
                <div class="orb orb-one"></div>
                <div class="orb orb-two"></div>
                <div class="hero-ai-card">
                    <div class="hero-ai-top">
                        <span class="hero-ai-dot"></span>
                        <span class="hero-ai-label">AI WORKSPACE ACTIVE</span>
                    </div>
                    <div class="hero-ai-line"></div>
                    <div class="hero-ai-stat">
                        <span>Documents</span>
                        <b>Ready to analyze</b>
                    </div>
                    <div class="hero-ai-stat">
                        <span>Retrieval</span>
                        <b>Semantic + Contextual</b>
                    </div>
                    <div class="hero-ai-stat">
                        <span>Outputs</span>
                        <b>Answers Â· Notes Â· Quiz</b>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="top-feature-strip">
        <div class="top-feature-item">
            <span class="top-feature-icon">ðŸ“‚</span>
            <div><b>Upload</b><small>Multiple PDFs</small></div>
        </div>
        <div class="top-feature-item">
            <span class="top-feature-icon">ðŸ§©</span>
            <div><b>Understand</b><small>Embeddings + FAISS</small></div>
        </div>
        <div class="top-feature-item">
            <span class="top-feature-icon">ðŸ’¬</span>
            <div><b>Interact</b><small>Context-aware answers</small></div>
        </div>
        <div class="top-feature-item">
            <span class="top-feature-icon">ðŸŽ“</span>
            <div><b>Learn</b><small>Notes, quiz, flashcards</small></div>
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
                    st.session_state.quiz_data = []
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.flashcards = []
                    st.session_state.important_questions = ""
                    st.session_state.study_pack = {}
                    st.session_state.last_similarity_average = 0.0
                    st.session_state.bookmarks = []
                    st.session_state.ai_tools_output = {}
                    st.session_state.document_search_results = []

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
                <div class="feature-chip">ðŸ’¬ Ask Questions</div>
                <div class="feature-chip">ðŸ“ Build Summaries</div>
                <div class="feature-chip">ðŸ“š Create Notes</div>
                <div class="feature-chip">ðŸ“¥ Export Results</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.success(
        "Documents indexed successfully â€” your AI workspace is ready."
    )


# ============================================================
# Dashboard
# ============================================================
document_words = estimate_document_words() if documents_are_ready else 0
estimated_reading_minutes = (
    max(1, round(document_words / 220))
    if document_words
    else 0
)

similarity_value = (
    st.session_state.last_similarity_average * 100
    if st.session_state.last_similarity_average
    else 0
)

workspace_status = "Ready" if documents_are_ready else "Waiting"
workspace_status_icon = "â—" if documents_are_ready else "â—‹"
index_progress = 100 if documents_are_ready else 8

learning_assets = (
    len(st.session_state.quiz_data)
    + len(st.session_state.flashcards)
    + len(st.session_state.bookmarks)
)

st.markdown(
    f"""
    <style>
    .dash-shell {{
        margin: 8px 0 24px 0;
    }}

    .dash-top {{
        display: grid;
        grid-template-columns: 1.5fr 0.5fr;
        gap: 14px;
        margin-bottom: 14px;
    }}

    .dash-welcome {{
        position: relative;
        overflow: hidden;
        min-height: 170px;
        padding: 25px;
        border-radius: 23px;
        background:
            radial-gradient(circle at 85% 10%, rgba(255,255,255,0.16), transparent 28%),
            linear-gradient(135deg, #111c44 0%, #1d4ed8 52%, #6d28d9 100%);
        border: 1px solid rgba(255,255,255,0.13);
        box-shadow: 0 22px 52px rgba(30,64,175,0.23);
    }}

    .dash-welcome::after {{
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        right: -55px;
        bottom: -95px;
        border-radius: 50%;
        background: rgba(255,255,255,0.09);
    }}

    .dash-eyebrow {{
        color: #bfdbfe !important;
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 0.16em;
        margin-bottom: 10px;
    }}

    .dash-title {{
        color: #ffffff !important;
        font-size: 29px;
        font-weight: 900;
        letter-spacing: -0.035em;
        line-height: 1.12;
    }}

    .dash-subtitle {{
        color: #dbeafe !important;
        margin-top: 9px;
        max-width: 620px;
        font-size: 13px;
        line-height: 1.65;
    }}

    .dash-progress-wrap {{
        margin-top: 18px;
        max-width: 520px;
    }}

    .dash-progress-meta {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 7px;
        color: #dbeafe !important;
        font-size: 11px;
        font-weight: 700;
    }}

    .dash-progress {{
        height: 8px;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
    }}

    .dash-progress > span {{
        display: block;
        width: {index_progress}%;
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #60a5fa, #c4b5fd);
        box-shadow: 0 0 18px rgba(147,197,253,0.65);
    }}

    .status-card {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 170px;
        border-radius: 23px;
        background: {theme['surface']} !important;
        border: 1px solid {theme['border']} !important;
        box-shadow: 0 16px 38px {theme['shadow']};
        text-align: center;
        padding: 18px;
    }}

    .status-orb {{
        width: 76px;
        height: 76px;
        display: grid;
        place-items: center;
        border-radius: 50%;
        margin-bottom: 12px;
        background: conic-gradient(
            #22c55e {similarity_value if similarity_value else 82}%,
            {theme['border']} 0
        );
        box-shadow: 0 12px 28px rgba(34,197,94,0.18);
        position: relative;
    }}

    .status-orb::after {{
        content: "";
        position: absolute;
        inset: 8px;
        border-radius: 50%;
        background: {theme['surface']};
    }}

    .status-orb span {{
        position: relative;
        z-index: 2;
        color: #22c55e !important;
        font-size: 24px;
        font-weight: 900;
    }}

    .status-label {{
        color: {theme['muted']} !important;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.12em;
    }}

    .status-value {{
        color: {theme['heading']} !important;
        font-size: 20px;
        font-weight: 900;
        margin-top: 4px;
    }}

    .metric-grid-pro {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 13px;
        margin-bottom: 14px;
    }}

    .metric-pro {{
        position: relative;
        overflow: hidden;
        min-height: 128px;
        padding: 18px;
        border-radius: 19px;
        background: {theme['surface']} !important;
        border: 1px solid {theme['border']} !important;
        box-shadow: 0 12px 30px {theme['shadow']};
        transition: transform 0.22s ease, border-color 0.22s ease;
    }}

    .metric-pro:hover {{
        transform: translateY(-4px);
        border-color: #60a5fa !important;
    }}

    .metric-pro::after {{
        content: "";
        position: absolute;
        width: 88px;
        height: 88px;
        right: -35px;
        top: -35px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59,130,246,0.16), transparent 70%);
    }}

    .metric-icon-pro {{
        width: 37px;
        height: 37px;
        display: grid;
        place-items: center;
        border-radius: 11px;
        background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(124,58,237,0.15));
        font-size: 18px;
        margin-bottom: 13px;
    }}

    .metric-name-pro {{
        color: {theme['muted']} !important;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.04em;
    }}

    .metric-value-pro {{
        color: {theme['heading']} !important;
        font-size: 27px;
        font-weight: 900;
        margin-top: 3px;
        letter-spacing: -0.035em;
    }}

    .metric-note-pro {{
        color: {theme['muted']} !important;
        font-size: 10px;
        margin-top: 5px;
    }}

    .insight-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 13px;
    }}

    .insight-card {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 17px 19px;
        border-radius: 17px;
        background: {theme['surface']} !important;
        border: 1px solid {theme['border']} !important;
        box-shadow: 0 10px 26px {theme['shadow']};
    }}

    .insight-left b {{
        display: block;
        color: {theme['heading']} !important;
        font-size: 13px;
    }}

    .insight-left small {{
        display: block;
        color: {theme['muted']} !important;
        margin-top: 4px;
        font-size: 10px;
    }}

    .insight-number {{
        color: #2563eb !important;
        font-size: 22px;
        font-weight: 900;
        white-space: nowrap;
    }}

    @media (max-width: 900px) {{
        .dash-top {{
            grid-template-columns: 1fr;
        }}

        .metric-grid-pro {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}

    @media (max-width: 560px) {{
        .metric-grid-pro,
        .insight-grid {{
            grid-template-columns: 1fr;
        }}

        .dash-title {{
            font-size: 24px;
        }}
    }}
    </style>

    <div class="dash-shell">
        <div class="dash-top">
            <div class="dash-welcome">
                <div class="dash-eyebrow">SMART WORKSPACE DASHBOARD</div>
                <div class="dash-title">Everything you need,<br>at one glance.</div>
                <div class="dash-subtitle">
                    Your live document intelligence overview â€” files, indexing,
                    learning assets, and retrieval performance.
                </div>
                <div class="dash-progress-wrap">
                    <div class="dash-progress-meta">
                        <span>Workspace preparation</span>
                        <span>{index_progress}%</span>
                    </div>
                    <div class="dash-progress"><span></span></div>
                </div>
            </div>

            <div class="status-card">
                <div class="status-orb">
                    <span>{workspace_status_icon}</span>
                </div>
                <div class="status-label">WORKSPACE STATUS</div>
                <div class="status-value">{workspace_status}</div>
            </div>
        </div>

        <div class="metric-grid-pro">
            <div class="metric-pro">
                <div class="metric-icon-pro">ðŸ“„</div>
                <div class="metric-name-pro">PDF DOCUMENTS</div>
                <div class="metric-value-pro">{pdf_count}</div>
                <div class="metric-note-pro">Files in current workspace</div>
            </div>

            <div class="metric-pro">
                <div class="metric-icon-pro">ðŸ§©</div>
                <div class="metric-name-pro">TEXT CHUNKS</div>
                <div class="metric-value-pro">{chunk_count}</div>
                <div class="metric-note-pro">Searchable knowledge units</div>
            </div>

            <div class="metric-pro">
                <div class="metric-icon-pro">ðŸ§ </div>
                <div class="metric-name-pro">VECTOR EMBEDDINGS</div>
                <div class="metric-value-pro">{embedding_count}</div>
                <div class="metric-note-pro">Indexed semantic vectors</div>
            </div>

            <div class="metric-pro">
                <div class="metric-icon-pro">ðŸ’¬</div>
                <div class="metric-name-pro">AI QUESTIONS</div>
                <div class="metric-value-pro">{st.session_state.questions_asked}</div>
                <div class="metric-note-pro">Questions answered this session</div>
            </div>
        </div>

        <div class="insight-grid">
            <div class="insight-card">
                <div class="insight-left">
                    <b>Estimated reading time</b>
                    <small>Manual reading time saved with AI</small>
                </div>
                <div class="insight-number">{estimated_reading_minutes} min</div>
            </div>

            <div class="insight-card">
                <div class="insight-left">
                    <b>Learning assets created</b>
                    <small>Quiz items, flashcards, and bookmarks</small>
                </div>
                <div class="insight-number">{learning_assets}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Main workspace tabs
# ============================================================
(
    overview_tab,
    summary_tab,
    notes_tab,
    ask_tab,
    quiz_tab,
    flashcards_tab,
    questions_tab,
    study_mode_tab,
    search_tab,
    tools_tab,
    bookmarks_tab,
    analytics_tab,
) = st.tabs(
    [
        "ðŸ“ Overview",
        "ðŸ“ Summary",
        "ðŸ“š Study Notes",
        "ðŸ’¬ Ask AI",
        "ðŸ§  Quiz",
        "ðŸƒ Flashcards",
        "â“ Important Questions",
        "ðŸŽ“ Study Mode",
        "ðŸ” Smart Search",
        "ðŸ› ï¸ AI Tools",
        "â­ Bookmarks",
        "ðŸ“ˆ Analytics",
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
                st.write(f"ðŸ“„ {pdf_file.name}")

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
        "âœ¨ Generate document summary",
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
                label="ðŸ“¥ Download summary as PDF",
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
        "ðŸ“š Generate study notes",
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
                label="ðŸ“¥ Download study notes as PDF",
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
            "ðŸš€ Ask NeuraDocs",
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

                        valid_scores = [float(result.get("score", 0.0)) for result in retrieval_results]
                        if valid_scores:
                            st.session_state.last_similarity_average = sum(valid_scores) / len(valid_scores)

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

        chat_export_text = build_chat_export_text()

        export_col_1, export_col_2 = st.columns(2)
        with export_col_1:
            st.download_button(
                "ðŸ“¥ Export full chat as text",
                data=chat_export_text,
                file_name="NeuraDocs_Conversation.txt",
                mime="text/plain",
                use_container_width=True,
                key="export_full_chat_text",
            )
        with export_col_2:
            try:
                from helpers.pdf_exporter import create_pdf

                chat_pdf = create_pdf(
                    "NeuraDocs - Conversation",
                    chat_export_text,
                )
                st.download_button(
                    "ðŸ“„ Export full chat as PDF",
                    data=chat_pdf,
                    file_name="NeuraDocs_Conversation.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="export_full_chat_pdf",
                )
            except Exception:
                st.caption("PDF chat export is unavailable.")

        for chat_index, chat in enumerate(list(st.session_state.chat_history)):
            with st.chat_message("user"):
                st.markdown(chat["question"])

            with st.chat_message("assistant"):
                st.markdown(chat["answer"])

                action_col_1, action_col_2 = st.columns(2)

                with action_col_1:
                    bookmark_exists = any(
                        saved.get("question") == chat.get("question")
                        and saved.get("answer") == chat.get("answer")
                        for saved in st.session_state.bookmarks
                    )

                    if st.button(
                        "âœ… Bookmarked" if bookmark_exists else "â­ Bookmark",
                        key=f"bookmark_chat_{chat_index}",
                        use_container_width=True,
                        disabled=bookmark_exists,
                    ):
                        st.session_state.bookmarks.append(
                            {
                                "question": chat.get("question", ""),
                                "answer": chat.get("answer", ""),
                                "sources": chat.get("sources", []),
                            }
                        )
                        st.rerun()

                with action_col_2:
                    if st.button(
                        "ðŸ—‘ï¸ Delete answer",
                        key=f"delete_chat_{chat_index}",
                        use_container_width=True,
                    ):
                        st.session_state.chat_history.pop(chat_index)
                        st.rerun()

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


# ------------------------------------------------------------
# Quiz tab
# ------------------------------------------------------------
with quiz_tab:
    st.markdown("""<div class="premium-card"><div class="section-kicker">SELF ASSESSMENT</div><div class="section-title">AI Quiz Generator</div><div class="section-copy">Generate MCQs directly from uploaded PDFs and check your score instantly.</div></div>""", unsafe_allow_html=True)
    quiz_count = st.slider("Number of questions", 5, 15, 10, disabled=not documents_are_ready, key="quiz_question_count")
    if st.button("ðŸ§  Generate quiz", use_container_width=True, disabled=not documents_are_ready, key="generate_quiz_button"):
        try:
            with st.spinner("Creating a balanced quiz from your PDFs..."):
                st.session_state.quiz_data = generate_quiz(build_complete_document_text(), quiz_count)
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
        except Exception as error:
            st.error(format_user_friendly_error(error))
    if st.session_state.quiz_data:
        with st.form("quiz_answer_form"):
            for index, item in enumerate(st.session_state.quiz_data):
                st.markdown(f"#### {index + 1}. {item['question']}")
                st.session_state.quiz_answers[index] = st.radio("Choose an answer", list(range(4)), format_func=lambda option_index, options=item["options"]: options[option_index], index=None, key=f"quiz_answer_{index}", label_visibility="collapsed")
            quiz_submitted = st.form_submit_button("âœ… Submit quiz", use_container_width=True)
        if quiz_submitted:
            st.session_state.quiz_submitted = True
        if st.session_state.quiz_submitted:
            score = 0
            for index, item in enumerate(st.session_state.quiz_data):
                selected = st.session_state.quiz_answers.get(index)
                correct = item["answer_index"]
                if selected == correct:
                    score += 1
                    st.success(f"Question {index + 1}: Correct â€” {item['options'][correct]}")
                else:
                    st.error(f"Question {index + 1}: Correct answer â€” {item['options'][correct]}")
                if item.get("explanation"):
                    st.caption(item["explanation"])
            total = len(st.session_state.quiz_data)
            st.metric("Quiz score", f"{score}/{total}", delta=f"{(score / total) * 100:.0f}%")
    elif documents_are_ready:
        st.caption("Generate a quiz to begin your self-assessment.")


# ------------------------------------------------------------
# Flashcards tab
# ------------------------------------------------------------
with flashcards_tab:
    st.markdown("""<div class="premium-card"><div class="section-kicker">ACTIVE RECALL</div><div class="section-title">AI Flashcards</div><div class="section-copy">Build quick revision cards from the most important document concepts.</div></div>""", unsafe_allow_html=True)
    flashcard_count = st.slider("Number of flashcards", 6, 20, 12, disabled=not documents_are_ready, key="flashcard_count")
    if st.button("ðŸƒ Generate flashcards", use_container_width=True, disabled=not documents_are_ready, key="generate_flashcards_button"):
        try:
            with st.spinner("Creating revision flashcards..."):
                st.session_state.flashcards = generate_flashcards(build_complete_document_text(), flashcard_count)
        except Exception as error:
            st.error(format_user_friendly_error(error))
    if st.session_state.flashcards:
        for index, card in enumerate(st.session_state.flashcards, 1):
            with st.expander(f"Card {index}: {card['front']}"):
                st.markdown(card["back"])
    elif documents_are_ready:
        st.caption("Generate flashcards for active recall practice.")


# ------------------------------------------------------------
# Important questions tab
# ------------------------------------------------------------
with questions_tab:
    st.markdown("""<div class="premium-card"><div class="section-kicker">EXAM PREPARATION</div><div class="section-title">Important Questions Generator</div><div class="section-copy">Create 2-mark, 5-mark, 10-mark, and viva questions from your PDFs.</div></div>""", unsafe_allow_html=True)
    if st.button("â“ Generate important questions", use_container_width=True, disabled=not documents_are_ready, key="generate_important_questions_button"):
        try:
            with st.spinner("Preparing exam and viva questions..."):
                st.session_state.important_questions = generate_important_questions(build_complete_document_text())
        except Exception as error:
            st.error(format_user_friendly_error(error))
    if st.session_state.important_questions:
        st.markdown(st.session_state.important_questions)
        st.download_button("ðŸ“¥ Download questions as text", st.session_state.important_questions, "NeuraDocs_Important_Questions.txt", "text/plain", use_container_width=True, key="download_important_questions_button")
    elif documents_are_ready:
        st.caption("Generate question sets for exams and viva preparation.")


# ------------------------------------------------------------
# Study Mode tab
# ------------------------------------------------------------
with study_mode_tab:
    st.markdown("""<div class="premium-card"><div class="section-kicker">ALL-IN-ONE LEARNING</div><div class="section-title">AI Study Mode</div><div class="section-copy">Generate overview, revision notes, key terms, flashcards, quiz, and viva questions in one request.</div></div>""", unsafe_allow_html=True)
    if st.button("ðŸŽ“ Build complete study pack", use_container_width=True, disabled=not documents_are_ready, key="generate_study_pack_button"):
        try:
            with st.spinner("Building your complete AI study pack..."):
                started = time.perf_counter()
                st.session_state.study_pack = generate_study_pack(build_complete_document_text())
                st.success(f"Study pack created in {time.perf_counter() - started:.1f} seconds.")
        except Exception as error:
            st.error(format_user_friendly_error(error))
    pack = st.session_state.study_pack
    if pack:
        st.markdown("### Overview")
        st.write(pack.get("overview", ""))
        st.markdown("### Revision notes")
        for point in pack.get("revision_notes", []):
            st.markdown(f"- {point}")
        st.markdown("### Key terms")
        for item in pack.get("key_terms", []):
            if isinstance(item, dict):
                st.markdown(f"**{item.get('term', 'Term')}** â€” {item.get('definition', '')}")
        st.markdown("### Flashcards")
        for index, card in enumerate(pack.get("flashcards", []), 1):
            if isinstance(card, dict):
                with st.expander(f"{index}. {card.get('front', 'Flashcard')}"):
                    st.write(card.get("back", ""))
        st.markdown("### Mini quiz")
        for index, item in enumerate(pack.get("quiz", []), 1):
            if isinstance(item, dict):
                st.markdown(f"**{index}. {item.get('question', '')}**")
                options = item.get("options", [])
                answer_index = item.get("answer_index", 0)
                for option_index, option in enumerate(options):
                    st.write(f"{'âœ…' if option_index == answer_index else 'â€¢'} {option}")
                if item.get("explanation"):
                    st.caption(item["explanation"])
        st.markdown("### Viva questions")
        for index, item in enumerate(pack.get("viva_questions", []), 1):
            if isinstance(item, dict):
                st.markdown(f"**{index}. {item.get('question', '')}**")
                st.write(item.get("answer", ""))
    elif documents_are_ready:
        st.caption("Build one complete pack for fast revision and viva practice.")



# ------------------------------------------------------------
# Smart Search tab
# ------------------------------------------------------------
with search_tab:
    st.markdown(
        """
        <div class="premium-card">
            <div class="section-kicker">IN-DOCUMENT DISCOVERY</div>
            <div class="section-title">Smart Search</div>
            <div class="section-copy">
                Search the indexed chunks and inspect the closest matching passages.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    search_query = st.text_input(
        "Search inside uploaded PDFs",
        placeholder="Example: operating system, neural network, conclusion...",
        disabled=not documents_are_ready,
        key="smart_search_query",
    )

    if st.button(
        "ðŸ” Search documents",
        use_container_width=True,
        disabled=not documents_are_ready,
        key="smart_search_button",
    ):
        if not search_query.strip():
            st.warning("Enter a search query.")
        else:
            try:
                from helpers.embedding_model import create_query_embedding

                search_embedding = create_query_embedding(search_query)
                st.session_state.document_search_results = (
                    search_similar_chunks(
                        question_embedding=search_embedding,
                        index=st.session_state.faiss_index,
                        chunks=st.session_state.all_chunks,
                        top_k=10,
                    )
                )
            except Exception as error:
                st.error(format_user_friendly_error(error))

    for result_index, result in enumerate(
        st.session_state.document_search_results,
        start=1,
    ):
        details = extract_source_details(result["chunk"])
        pdf_name = details.get("pdf_name", "Unknown PDF")
        chunk_number = details.get("original_chunk_number", "â€”")
        score = float(result.get("score", 0.0))

        with st.expander(
            f"{result_index}. {pdf_name} Â· Chunk {chunk_number} Â· "
            f"Similarity {score * 100:.1f}%"
        ):
            st.code(result["chunk"], language=None)


# ------------------------------------------------------------
# AI Tools tab
# ------------------------------------------------------------
with tools_tab:
    st.markdown(
        """
        <div class="premium-card">
            <div class="section-kicker">SPECIALIZED GENERATORS</div>
            <div class="section-title">AI Learning Tools</div>
            <div class="section-copy">
                Generate document structure, mind maps, timelines, formulas,
                concept maps, code examples, and diagram plans.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_tool = st.selectbox(
        "Choose a tool",
        [
            "Table of Contents",
            "Mind Map",
            "Concept Map",
            "Timeline",
            "Formula Sheet",
            "Code Examples",
            "Diagram Ideas",
        ],
        disabled=not documents_are_ready,
        key="selected_ai_tool",
    )

    if st.button(
        f"âœ¨ Generate {selected_tool}",
        use_container_width=True,
        disabled=not documents_are_ready,
        key="generate_selected_ai_tool",
    ):
        try:
            with st.spinner(f"Generating {selected_tool.lower()}..."):
                st.session_state.ai_tools_output[selected_tool] = (
                    generate_ai_tool_output(
                        build_complete_document_text(),
                        selected_tool,
                    )
                )
        except Exception as error:
            st.error(format_user_friendly_error(error))

    if selected_tool in st.session_state.ai_tools_output:
        tool_output = st.session_state.ai_tools_output[selected_tool]
        st.markdown(tool_output)

        st.download_button(
            f"ðŸ“¥ Download {selected_tool}",
            data=tool_output,
            file_name=f"NeuraDocs_{selected_tool.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"download_{selected_tool}",
        )


# ------------------------------------------------------------
# Bookmarks tab
# ------------------------------------------------------------
with bookmarks_tab:
    st.markdown(
        """
        <div class="premium-card">
            <div class="section-kicker">SAVED KNOWLEDGE</div>
            <div class="section-title">Bookmarked Answers</div>
            <div class="section-copy">
                Save important AI answers for quick revision and later export.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.bookmarks:
        bookmark_export_parts = []

        for bookmark_index, bookmark in enumerate(
            list(st.session_state.bookmarks)
        ):
            with st.expander(
                f"{bookmark_index + 1}. {bookmark.get('question', 'Saved answer')}",
                expanded=bookmark_index == 0,
            ):
                st.markdown(bookmark.get("answer", ""))

                if st.button(
                    "Remove bookmark",
                    key=f"remove_bookmark_{bookmark_index}",
                ):
                    st.session_state.bookmarks.pop(bookmark_index)
                    st.rerun()

            bookmark_export_parts.append(
                f"Question: {bookmark.get('question', '')}\\n"
                f"Answer: {bookmark.get('answer', '')}\\n"
                + "-" * 40
            )

        bookmark_export_text = "\\n\\n".join(bookmark_export_parts)

        st.download_button(
            "ðŸ“¥ Export all bookmarks",
            data=bookmark_export_text,
            file_name="NeuraDocs_Bookmarks.txt",
            mime="text/plain",
            use_container_width=True,
            key="export_all_bookmarks",
        )
    else:
        st.info("Bookmark useful answers from the Ask AI tab.")


# ------------------------------------------------------------
# Analytics tab
# ------------------------------------------------------------
with analytics_tab:
    st.markdown(
        """
        <div class="premium-card">
            <div class="section-kicker">WORKSPACE INSIGHTS</div>
            <div class="section-title">Analytics Dashboard</div>
            <div class="section-copy">
                Review document size, engagement, generated learning assets,
                and current retrieval quality.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    analytics_data = {
        "PDFs": pdf_count,
        "Chunks": chunk_count,
        "Questions": st.session_state.questions_asked,
        "Bookmarks": len(st.session_state.bookmarks),
        "Quiz Items": len(st.session_state.quiz_data),
        "Flashcards": len(st.session_state.flashcards),
    }

    st.bar_chart(analytics_data)

    stat_col_1, stat_col_2, stat_col_3 = st.columns(3)

    with stat_col_1:
        st.metric("Document words", document_words)

    with stat_col_2:
        st.metric(
            "Estimated reading",
            f"{estimated_reading_minutes} min",
        )

    with stat_col_3:
        st.metric(
            "Average similarity",
            (
                f"{st.session_state.last_similarity_average * 100:.1f}%"
                if st.session_state.last_similarity_average
                else "â€”"
            ),
        )

    st.caption(
        "Analytics are session-based. A persistent user library and cross-session "
        "history require a database and authentication layer."
    )


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer-card">
        Â© 2026 <b>NeuraDocs</b> Â· Built with Python, Streamlit,
        FAISS, Sentence Transformers and Generative AI
    </div>
    """,
    unsafe_allow_html=True,
)
