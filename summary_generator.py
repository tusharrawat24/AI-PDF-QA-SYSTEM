"""
PDF summary generation using the Gemini API.
"""

from helpers.gemini_client import generate_content


def generate_pdf_summary(document_text: str) -> str:
    """
    Generate a detailed summary of the uploaded PDF.
    """

    if not document_text or not document_text.strip():
        raise ValueError("The PDF text cannot be empty.")

    prompt = f"""
You are an AI document summarization assistant.

Create a detailed and well-structured summary of the PDF content below.

Instructions:
1. Start with a short overview of the document.
2. Identify and explain the main topics.
3. Include important definitions, concepts, arguments, and conclusions.
4. Use headings and bullet points for readability.
5. Preserve important technical terms.
6. Do not invent information that is not present in the document.
7. End with a section titled "Key Takeaways".
8. Keep the summary detailed but avoid unnecessary repetition.

PDF Content:
{document_text}

Detailed Summary:
"""

    try:
        summary_text = generate_content(prompt)

        if not summary_text:
            return "No summary was received from Gemini."

        return summary_text

    except Exception as error:
        raise RuntimeError(
            f"Gemini could not generate the PDF summary: {error}"
        ) from error
