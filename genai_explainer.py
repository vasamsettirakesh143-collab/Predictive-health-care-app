"""
genai_explainer.py
Turns a numeric ML risk score + contributing factors into a personalised,
human-readable explanation.

If an ANTHROPIC_API_KEY environment variable is set on Render, this calls
the real Claude API for a genuinely generated explanation (the "GenAI"
part of the project). If no key is set, it falls back to a rule-based
template so the app still works out of the box with zero configuration.
"""
import os
import requests

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


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
    if not ANTHROPIC_API_KEY:
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
            ANTHROPIC_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(block.get("text", "") for block in data.get("content", []))
        return text.strip(), "llm"
    except Exception as e:
        # Never let an API failure break the app -- fall back gracefully.
        fallback = _template_explanation(probability, factors, patient)
        return fallback + f"\n\n[Note: live GenAI call failed ({e}); showing template explanation.]", "template_fallback"
