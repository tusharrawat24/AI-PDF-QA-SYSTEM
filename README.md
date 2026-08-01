# AI PDF Assistant

AI PDF Assistant is a Retrieval-Augmented Generation application that allows users to upload PDF documents, ask questions, generate summaries, create study notes, and export generated content as PDF files.

## Features

- Upload one or more PDF documents
- Extract and split PDF text into chunks
- Generate embeddings using Sentence Transformers
- Store and search vectors using FAISS
- Ask questions using Gemini AI
- Generate PDF summaries
- Generate structured study notes
- View source PDF, chunk number, and similarity score
- Download summaries and notes as PDF
- Track PDFs, chunks, embeddings, and questions through a dashboard

## Technology Stack

- Python
- Streamlit
- Google Gemini API
- FAISS
- Sentence Transformers
- PyPDF
- ReportLab

## Project Architecture

```text
PDF Upload
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Embedding Generation
   ↓
FAISS Vector Index
   ↓
User Question
   ↓
Similarity Search
   ↓
Relevant Chunks
   ↓
Gemini AI
   ↓
Final Answer    