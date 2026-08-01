"""
Question answering using the Gemini API.
"""

from helpers.gemini_client import generate_content


def get_answer(question: str, context: str) -> str:
    """
    Generate an answer using only the retrieved PDF context.
    """

    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    if not context or not context.strip():
        raise ValueError("The PDF context cannot be empty.")

    prompt = f"""
You are an AI PDF Question Answering Assistant.

Answer the user's question using only the PDF context below.

Instructions:
1. Use only information available in the supplied context.
2. You may combine related facts from multiple context sections.
3. Provide a clear and concise answer.
4. Do not add unsupported facts.
5. If the context contains related information but not a direct definition,
   explain the concept using the available related details.
6. Only respond with:
   "I could not find this information in the uploaded PDF."
   when the context contains no relevant information.

PDF Context:
{context}

User Question:
{question}

Answer:
"""

    try:
        answer_text = generate_content(prompt)

        if not answer_text:
            return "No answer was received from Gemini."

        return answer_text

    except Exception as error:
        raise RuntimeError(
            f"Gemini could not generate an answer: {error}"
        ) from error
