# complaints_analyser.py
# Robust Citizen Complaint Analyzer using Local LLM (Mistral via Ollama)

import subprocess
import json
import re

# -------------------------------------------------
# 1. LOCAL LLM CALL (OLLAMA)
# -------------------------------------------------
def call_llm(prompt: str) -> str:
    """
    Calls local Mistral LLM via Ollama.
    Returns raw text output from the model.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt,
            capture_output=True,
            timeout=60,
            encoding="utf-8",
            errors="ignore"
        )
        return result.stdout.strip()
    except Exception as e:
        return f"LLM_ERROR: {e}"

# -------------------------------------------------
# 2. COMPLAINT SUMMARIZATION (LLM)
# -------------------------------------------------
def summarize_complaint(text: str, max_words: int = 20) -> str:
    """
    Generates a concise summary using LLM.
    """
    prompt = f"""
Summarize the following citizen complaint in under {max_words} words.
Return only the summary sentence.

Complaint:
{text}
"""
    summary = call_llm(prompt)
    return summary

# -------------------------------------------------
# 3. COMPLAINT CLASSIFICATION (LLM)
# -------------------------------------------------
def classify_complaint(text: str) -> str:
    """
    Classifies complaint using LLM into predefined categories.
    """
    prompt = f"""
You are a municipal complaint classifier.

Categories:
Garbage, Road, Traffic, Water, Sewage, Electricity, Pollution, Other

Return ONLY JSON in this format:
{{"category": "..."}} 

Complaint:
{text}
"""
    response = call_llm(prompt)

    try:
        json_text = re.search(r"\{.*\}", response, re.DOTALL).group()
        data = json.loads(json_text)
        return data.get("category", "Other")
    except Exception:
        return "Other"

# -------------------------------------------------
# 4. DEPARTMENT ROUTING
# -------------------------------------------------
DEPARTMENT_MAP = {
    "Garbage": "Municipal Corporation",
    "Road": "Public Works Department",
    "Traffic": "Traffic Police",
    "Water": "Water Supply Board",
    "Sewage": "Water & Sewerage Board",
    "Electricity": "Electricity Board",
    "Pollution": "Pollution Control Board",
    "Other": "General Administration"
}

def route_complaint(category: str) -> str:
    """
    Maps category to department.
    """
    return DEPARTMENT_MAP.get(category, "General Administration")

# -------------------------------------------------
# 5. FULL PIPELINE
# -------------------------------------------------
def process_complaint(text: str) -> dict:
    """
    Complete complaint analysis pipeline.
    """
    if not text or len(text.strip()) < 5:
        raise ValueError("Complaint text too short")

    summary = summarize_complaint(text)
    category = classify_complaint(text)
    department = route_complaint(category)

    return {
        "original_text": text,
        "summary": summary,
        "predicted_category": category,
        "routed_department": department
    }

# -------------------------------------------------
# 6. TEST RUN
# -------------------------------------------------
if __name__ == "__main__":
    sample_complaint = (
        "Garbage has not been collected for several days and "
        "the area has become unhygienic and foul smelling."
    )

    result = process_complaint(sample_complaint)

    print("\n--- Complaint Analysis Result ---")
    for key, value in result.items():
        print(f"{key}: {value}")
