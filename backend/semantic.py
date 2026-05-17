"""
semantic.py — Semantic Analysis Layer
Runs in parallel with the main pipeline on complete utterances.
Produces structured JSON for: dashboard, context injection, and post-call report.
"""

import json
import google.generativeai as genai

SEMANTIC_PROMPT = """You are a real-time call analysis engine for an Indian banking customer support line.
Analyze the customer's LATEST utterance in the context of the conversation so far.

Return a JSON object with these exact fields:
{
  "intent": "string — primary intent (balance_inquiry, loan_query, card_block, complaint, greeting, farewell, transfer_query, account_opening, kyc_update, emi_query, fd_query, upi_issue, general_question)",
  "sentiment": "float from -1.0 (very negative/angry) to 1.0 (very positive/happy)",
  "urgency_level": "low | medium | high | critical",
  "compliance_flag": "boolean — true if utterance references sensitive data like account numbers, Aadhaar, PAN, card numbers",
  "escalation_recommended": "boolean — true if customer sounds frustrated/angry or asks for manager/supervisor",
  "one_line_summary": "string — one sentence summarizing what the customer said",
  "recommended_tone": "empathetic | professional | reassuring | apologetic | cheerful"
}

Rules:
- Analyze ONLY the customer's utterance, not the bot's response.
- A frustrated customer who says "this is ridiculous" should get sentiment < -0.5 and escalation_recommended=true.
- A happy customer saying "thank you so much" should get sentiment > 0.5.
- Set compliance_flag=true if ANY sensitive data pattern appears (numbers that look like account/card/Aadhaar).
- Return ONLY valid JSON. No markdown. No explanation."""


async def analyze_utterance(
    utterance: str,
    api_key: str,
    conversation_summary: str = "",
) -> dict:
    """
    Run semantic analysis on a complete utterance.
    conversation_summary provides prior context for multi-turn awareness.
    Returns structured JSON dict.
    """
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=SEMANTIC_PROMPT,
        generation_config={"response_mime_type": "application/json"},
    )

    prompt = f'Customer utterance: "{utterance}"'
    if conversation_summary:
        prompt = f"Conversation so far: {conversation_summary}\n\n{prompt}"

    response = await model.generate_content_async(prompt)

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        # Fallback if Gemini returns malformed JSON
        return {
            "intent": "unknown",
            "sentiment": 0.0,
            "urgency_level": "low",
            "compliance_flag": False,
            "escalation_recommended": False,
            "one_line_summary": utterance[:100],
            "recommended_tone": "professional",
        }
