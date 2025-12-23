import google.generativeai as genai
import os

MAX_PROMPT_CHARS = 12000  # cost + quota guardrail

def configure_llm():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_insights(summary_text):
    # ---------- Guardrail 1: prompt size ----------
    if len(summary_text) > MAX_PROMPT_CHARS:
        return "⚠️ Dataset summary is too large. Please reduce columns or rows."

    try:
        model = genai.GenerativeModel("models/gemini-flash-latest")

        prompt = f"""
        You are a senior business data analyst.
        Do NOT invent data.
        Base insights strictly on the summary.

        Summary:
        {summary_text}

        Provide:
        1. Dataset overview
        2. Data quality issues
        3. Key patterns
        4. Business meaning
        5. Actionable recommendations
        """

        response = model.generate_content(prompt)

        # ---------- Guardrail 2: empty response ----------
        if not response or not hasattr(response, "text") or not response.text:
            return "⚠️ LLM did not return a response. Please try again later."

        return response.text

    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"


