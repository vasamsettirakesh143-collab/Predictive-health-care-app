"""
genai_explainer.py
Turns a numeric ML risk score + contributing factors into a personalised,
human-readable explanation.

If a GROQ_API_KEY environment variable is set on Render, this calls the
real Groq API (OpenAI-compatible chat completions) for a genuinely
generated explanation (the "GenAI" part of the project). If no key is
set, it falls back to a rule-based template so the app still works out
of the box with zero configuration.
"""
import os
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _template_explanation(probability, factors, patient):
    level = "high" if probability >= 0.66 else ("moderate" if probability >= 0.33 else "low")
    top = ", ".join(f[0] for f in factors[:3])
    lines = [
        f"Based on the submitted health data, the estimated Type-2 diabetes risk is {level} "
        f"({probability*100:.1f}% probability).",
        f"The factors contributing most to this result are: {top}.",
        "",
        "Suggested next steps:",
    ]
    if patient["activity_level"] < 1:
        lines.append("- Increase physical activity to at least 150 minutes/week of moderate exercise.")
    if patient["bmi"] >= 27:
        lines.append("- Work with a healthcare provider on a gradual, sustainable weight-management plan.")
    if patient["glucose"] >= 110:
        lines.append("- Schedule a follow-up fasting glucose test within the next 3 months.")
    if patient["smoking"] == 1:
        lines.append("- Consider a smoking-cessation program; smoking compounds metabolic risk.")
    if len(lines) == 4:
        lines.append("- Maintain current healthy habits and repeat screening annually.")
    lines.append("")
    lines.append("Note: This is an automated, advisory estimate and does not replace professional medical diagnosis.")
    return "\n".join(lines)


def generate_explanation(probability, factors, patient):
    """
    probability: float 0-1, model's predicted risk probability
    factors: list of (feature_name, importance) sorted by importance desc
    patient: dict of the raw submitted patient values
    """
    if not GROQ_API_KEY:
        return _template_explanation(probability, factors, patient), "template"

    top_factors_str = ", ".join(f"{name} (importance {imp:.2f})" for name, imp in factors[:4])
    prompt = (
        "You are a careful, plain-language health assistant. A machine learning model has "
        f"estimated a {probability*100:.1f}% probability of Type-2 diabetes risk for a patient "
        f"with these values: {patient}. The top contributing factors were: {top_factors_str}. "
        "Write a short (under 150 words), warm, clear explanation of this risk level for the "
        "patient, followed by 2-3 concrete, prioritized lifestyle recommendations. "
        "End with one sentence reminding them this is advisory and not a medical diagnosis."
    )
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return text.strip(), "llm"
    except Exception as e:
        # Never let an API failure break the app -- fall back gracefully.
        fallback = _template_explanation(probability, factors, patient)
        return fallback + f"\n\n[Note: live GenAI call failed ({e}); showing template explanation.]", "template_fallback"
