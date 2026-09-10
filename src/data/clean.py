import re
import string

def clean_text(text: str, remove_handles: bool = False, remove_urls: bool = True) -> str:
    """
    Clean and normalize customer support tweet text.
    - Normalizes extra whitespace and newlines.
    - Optionally replaces URLs with [URL].
    - Optionally removes Twitter user handles (@username).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text.strip()

    if remove_urls:
        # Standard HTTP/HTTPS and www URLs
        cleaned = re.sub(r"https?://\S+|www\.\S+", "[URL]", cleaned)

    if remove_handles:
        cleaned = re.sub(r"@\w+", "", cleaned)

    # Collapse multiple whitespaces/newlines into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned

def extract_brand_mentions(text: str) -> list:
    """Extract all @mentions from tweet text."""
    if not isinstance(text, str):
        return []
    return re.findall(r"@(\w+)", text)
