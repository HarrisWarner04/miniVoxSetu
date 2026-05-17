"""
pii.py — PII Redaction Layer
Redacts sensitive patterns from transcript text before sending to external LLM APIs.
Production upgrade path: Microsoft Presidio (ML-based detection).
"""

import re
from typing import Tuple

# Pattern order matters — specific patterns before general ones.
_PATTERNS = [
    # Credit/Debit card: 16 digits optionally grouped by 4 — BEFORE Aadhaar
    (re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[CARD_NO]"),

    # Aadhaar: 12 digits optionally grouped as 4-4-4
    (re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b"), "[AADHAAR]"),

    # PAN card: ABCDE1234F format
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE), "[PAN]"),

    # Indian mobile: optional +91/0 prefix, 10 digits starting with 6-9
    # Allows one optional space or hyphen in the middle (e.g. 98765 43210)
    (re.compile(r"(?<!\d)(?:\+91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)"), "[PHONE]"),

    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),

    # IFSC code: 4 letters + 0 + 6 alphanumeric
    (re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE), "[IFSC]"),

    # Bank account numbers: 8–18 digits (generic, comes last)
    (re.compile(r"\b\d{8,18}\b"), "[ACCOUNT_NO]"),
]


def redact_pii(text: str) -> Tuple[str, list]:
    """
    Replace PII patterns in text with typed placeholder tokens.
    Returns (redacted_text, list_of_token_types_found).
    findings list is for compliance logging only — never log actual values.
    """
    if not text or not text.strip():
        return text, []

    redacted = text
    findings = []

    for pattern, token in _PATTERNS:
        matches = pattern.findall(redacted)
        if matches:
            findings.extend([token.strip("[]")] * len(matches))
            redacted = pattern.sub(token, redacted)

    return redacted, findings


if __name__ == "__main__":
    tests = [
        "My phone number is 9876543210",
        "Call me on +91 98765 43210",
        "My Aadhaar is 1234 5678 9012",
        "PAN card ABCDE1234F",
        "Card number 4111 1111 1111 1111",
        "Email me at customer@neobank.in",
        "Account number 00123456789",
        "IFSC is HDFC0001234",
        "I have no sensitive data here",
        "Number 9876543210, aadhaar 1234 5678 9012, email foo@bar.com",
    ]
    print("=" * 55)
    for t in tests:
        out, found = redact_pii(t)
        print(f"IN : {t}")
        print(f"OUT: {out}  |  {found or 'clean'}")
        print()
