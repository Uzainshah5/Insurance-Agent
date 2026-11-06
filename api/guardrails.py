"""
guardrails.py — Security & Safety Guardrails for the RAG Assistant

This module enforces ChatGPT / enterprise-grade safety and security policies:
  1. Input Validation: guards against empty input or DoS (excessive character lengths).
  2. Prompt Injection Defense: detects attempts to override system prompts or jailbreak.
  3. System Prompt Protection: prevents leaking internal prompts or instructions.
  4. Harmful Content Filtering: blocks requests for exploits, malware, or violence.
  5. PII & Secret Redaction: masks credit cards, SSNs, and API keys.

All functions are kept simple, clean, and beginner-friendly with regular expressions.
"""

import re

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

MAX_QUESTION_LENGTH = 2000  # Characters (prevents token exhaustion attacks)
MIN_QUESTION_LENGTH = 1

# Standard ChatGPT-style refusal message
REFUSAL_MESSAGE = (
    "I cannot fulfill this request. My role is to safely assist with "
    "questions regarding Insurellm, its products, employees, and contracts, "
    "in accordance with security and safety policies."
)


# ---------------------------------------------------------------------------
# 1. Input Length & Structure Validation
# ---------------------------------------------------------------------------

def check_input_length(question: str) -> tuple[bool, str | None]:
    """Ensure the user's question is within safe character limits."""
    cleaned = question.strip()
    if len(cleaned) < MIN_QUESTION_LENGTH:
        return False, "Please enter a valid question."
    if len(cleaned) > MAX_QUESTION_LENGTH:
        return False, f"Question is too long (maximum {MAX_QUESTION_LENGTH} characters allowed)."
    return True, None


# ---------------------------------------------------------------------------
# 2. Prompt Injection & Jailbreak Detection
# ---------------------------------------------------------------------------

# Patterns commonly used to hijack or bypass LLM system instructions
INJECTION_PATTERNS = [
    # Ignore previous instructions
    r"(?i)\b(ignore|disregard|forget|override|bypass)\s+(all\s+)?(previous|prior|above|existing|system)\s+(instructions|prompts|rules|commands|constraints)\b",
    
    # Reveal system prompt or developer instructions
    r"(?i)\b(reveal|show|display|print|repeat|output|tell\s+me)\s+(your\s+)?(system\s+prompt|initial\s+prompt|developer\s+instructions|system\s+instructions|hidden\s+rules|internal\s+prompt)\b",
    
    # Jailbreak personas (DAN, Developer Mode, AIM, etc.)
    r"(?i)\b(dan\s+mode|jailbreak|developer\s+mode|unfiltered\s+mode|evil\s*gpt|aim\s+mode|chaos\s+mode)\b",
    
    # Roleplay as an unrestricted or unfiltered agent
    r"(?i)\byou\s+are\s+now\s+(an\s+unfiltered|an\s+unrestricted|a\s+rogue|free\s+of\s+rules)\b",
    
    # Prompt leak through formatting tricks ("start from line 1 of your prompt")
    r"(?i)\b(repeat\s+the\s+words\s+above|print\s+everything\s+above|what\s+are\s+the\s+first\s+\d+\s+words\s+of\s+your\s+prompt)\b",
]

def check_prompt_injection(question: str) -> tuple[bool, str | None]:
    """Detect common prompt injection, jailbreak, and system-prompt extraction attempts."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, question):
            return False, REFUSAL_MESSAGE
    return True, None


# ---------------------------------------------------------------------------
# 3. Harmful & Malicious Content Filter
# ---------------------------------------------------------------------------

# Patterns for malicious cybersecurity activities and dangerous content
HARMFUL_PATTERNS = [
    # Exploits, malware, ransomware, keyloggers
    r"(?i)\b(create|write|generate|build)\s+(a\s+)?(malware|ransomware|keylogger|trojan|rootkit|spyware|reverse\s+shell)\b",
    
    # Cyberattacks (DDoS, brute-force scripts, SQL injection payloads for exploitation)
    r"(?i)\b(ddos\s+attack|sql\s+injection\s+payload|exploit\s+vulnerability|bypass\s+firewall|crack\s+password)\b",
    
    # Weapons / explosives
    r"(?i)\b(how\s+to\s+make|synthesize|manufacture)\s+(a\s+)?(bomb|explosive|weapon|poison)\b",
]

def check_harmful_content(question: str) -> tuple[bool, str | None]:
    """Detect requests for malicious, harmful, or destructive content."""
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, question):
            return False, REFUSAL_MESSAGE
    return True, None


# ---------------------------------------------------------------------------
# 4. PII & Secret Redaction (Masking)
# ---------------------------------------------------------------------------

PII_PATTERNS = [
    # Credit Card numbers (16 digits with optional dashes or spaces)
    (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[REDACTED_CREDIT_CARD]"),
    
    # US Social Security Numbers (XXX-XX-XXXX)
    (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
    
    # OpenAI / OpenRouter style API keys (sk-...)
    (r"\bsk-[a-zA-Z0-9_\-]{20,}\b", "[REDACTED_API_KEY]"),
]

def redact_sensitive_data(text: str) -> str:
    """Mask any accidentally leaked or entered credit cards, SSNs, or API keys."""
    redacted_text = text
    for pattern, replacement in PII_PATTERNS:
        redacted_text = re.sub(pattern, replacement, redacted_text)
    return redacted_text


# ---------------------------------------------------------------------------
# 5. Master Guardrail Checkers
# ---------------------------------------------------------------------------

def run_input_guardrails(question: str) -> tuple[bool, str | None]:
    """
    Run all input guardrails before sending the question to the RAG pipeline.

    Returns:
      (True, None)            if the question is safe and approved.
      (False, refusal_reason) if the question violates security/safety policies.
    """
    # 1. Length check
    valid_length, length_error = check_input_length(question)
    if not valid_length:
        return False, length_error

    # 2. Prompt injection / jailbreak check
    safe_injection, injection_error = check_prompt_injection(question)
    if not safe_injection:
        return False, injection_error

    # 3. Harmful content check
    safe_content, content_error = check_harmful_content(question)
    if not safe_content:
        return False, content_error

    # All checks passed
    return True, None


def run_output_guardrails(answer: str) -> str:
    """
    Run output guardrails on the LLM's answer before sending to the frontend.
    Redacts any sensitive tokens, API keys, or personal identity numbers.
    """
    return redact_sensitive_data(answer)

