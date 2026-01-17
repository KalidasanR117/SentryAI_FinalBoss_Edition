import subprocess
import json
import textwrap


def _build_prompt(events, mode):
    """
    Convert structured events into an LLM-friendly prompt.
    """

    if not events:
        return (
            "No significant events were detected during surveillance. "
            "The environment appeared normal throughout the observation period."
        )

    event_lines = []
    for e in events:
        line = (
            f"- Time {e.get('start_time', '?')}s to {e.get('end_time', '?')}s | "
            f"Type: {e.get('type')} | "
            f"Severity: {e.get('final')} | "
            f"Confidence: {e.get('confidence', 'N/A')}"
        )
        event_lines.append(line)

    event_block = "\n".join(event_lines)

    prompt = f"""
You are an AI system generating an executive surveillance summary.

Context:
- Mode: {mode}
- The system detected multiple human activity events.
- Events are based on pose-based analysis (LIVE) or video-based transformer analysis (OFFLINE).

Detected Events:
{event_block}

Instructions:
- Write a professional executive summary.
- Mention how many events occurred.
- Highlight danger-level incidents clearly.
- Keep tone formal and suitable for an official surveillance report.
- Do NOT invent events.
- Do NOT use bullet points.
- Keep it concise (6–8 sentences).
"""

    return textwrap.dedent(prompt).strip()


def generate_llm_summary(events, mode="LIVE", model="mistral"):
    """
    Generate a summary using local Ollama (Mistral).
    Falls back safely if LLM fails.
    """

    prompt = _build_prompt(events, mode)

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            errors="ignore", 
            encoding="utf-8",      
            timeout=60
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        summary = result.stdout.strip()

        # Basic sanity check
        if len(summary) < 50:
            raise ValueError("LLM output too short")

        return summary

    except Exception as e:
        print("[LLM] Summary generation failed:", e)

        # ---------- SAFE FALLBACK ----------
        danger_count = len([e for e in events if e.get("final") == "danger"])
        total = len(events)

        return (
            f"This surveillance session detected {total} significant events. "
            f"Among them, {danger_count} were classified as danger-level incidents. "
            f"Events were identified using automated AI-based activity analysis. "
            f"This summary was generated using a fallback mechanism due to "
            f"local language model unavailability."
        )
