import re


def extract_source_details(chunk_text):
    pdf_match = re.search(r"Source PDF:\s*(.+)", chunk_text)
    chunk_match = re.search(r"Chunk Number:\s*(\d+)", chunk_text)

    pdf_name = pdf_match.group(1).strip() if pdf_match else "Unknown PDF"
    chunk_number = int(chunk_match.group(1)) if chunk_match else None

    return {
        "pdf_name": pdf_name,
        "original_chunk_number": chunk_number
    }