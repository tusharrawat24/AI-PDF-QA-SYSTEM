# 🧠 NeuraDocs

> **AI-Powered PDF Question Answering, Summarization & Study Assistant using Retrieval-Augmented Generation (RAG), FAISS and Google Gemini AI.**

🌐 **Live Demo:** https://neuradocs.streamlit.app/

---

## 📖 Overview

NeuraDocs is an AI-powered PDF Assistant that allows users to upload one or multiple PDF documents and interact with them using natural language.

Instead of manually reading hundreds of pages, users can ask questions, generate summaries, create study notes, and export the generated content.

The application combines **Retrieval-Augmented Generation (RAG)** with **Google Gemini AI** to provide context-aware responses from uploaded PDFs.

---

# ✨ Features

- 📂 Upload multiple PDF files
- 🤖 AI-powered Question Answering
- 📝 Generate detailed PDF Summary
- 📚 Generate AI Study Notes
- 🔍 Semantic Search using FAISS
- 🧠 Sentence Transformer Embeddings
- 💬 ChatGPT-style Conversation
- 📄 Source PDF Tracking
- 📊 Dashboard with PDF Statistics
- ⬇ Download Summary as PDF
- ⬇ Download Study Notes as PDF
- ⚡ Fast document retrieval using RAG

---

# 🏗 Project Architecture

```text
                 📂 Upload PDF
                        │
                        ▼
              📄 Extract PDF Text
                        │
                        ▼
               ✂ Split into Chunks
                        │
                        ▼
          🧠 Create Embeddings
                        │
                        ▼
          📚 Store in FAISS Index
                        │
                        ▼
              ❓ User Question
                        │
                        ▼
          🔎 Similarity Search
                        │
                        ▼
        📄 Retrieve Relevant Chunks
                        │
                        ▼
            🤖 Google Gemini AI
                        │
                        ▼
             💡 Final AI Response
```

---

# 🚀 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend |
| Streamlit | Web Interface |
| Google Gemini AI | Large Language Model |
| Sentence Transformers | Embedding Generation |
| FAISS | Vector Database |
| PyPDF | PDF Text Extraction |
| ReportLab | PDF Export |
| GitHub | Version Control |
| Streamlit Cloud | Deployment |

---

# 📁 Project Structure

```text
NeuraDocs/
│
├── app.py
│
├── helpers/
│   ├── embedding_model.py
│   ├── gemini_client.py
│   ├── notes_generator.py
│   ├── pdf_exporter.py
│   ├── pdf_loader.py
│   ├── qa_chain.py
│   ├── source_utils.py
│   ├── summary_generator.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── requirements.txt
├── README.md
└── .streamlit/
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/your-username/NeuraDocs.git
```

Move inside the project

```bash
cd NeuraDocs
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📸 Screenshots

> Add screenshots of:

- Home Page
- PDF Upload
- AI Chat
- Summary
- Study Notes
- Dashboard

---

# 🔮 Future Scope

- 🎙 Voice Input
- 📱 Android Application
- 🌍 Multi-language Support
- 📄 OCR for Scanned PDFs
- 🧠 Flashcard Generator
- 📝 Quiz Generator
- ☁ Cloud Database
- 👥 User Authentication

---

# ⭐ Why NeuraDocs?

Unlike traditional PDF readers, NeuraDocs understands the meaning of your documents using **Retrieval-Augmented Generation (RAG)** and **semantic search**.

It provides accurate answers from your uploaded PDFs instead of generating generic AI responses.

---

# 📜 License

This project is developed for educational and learning purposes.

---

## 👨‍💻 Developed By

**Tushar Rawat**

⭐ If you like this project, don't forget to give it a **Star** on GitHub!