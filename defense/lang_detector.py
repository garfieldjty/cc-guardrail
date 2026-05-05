import re

# Matches any CJK Unified Ideograph (covers all Chinese characters)
_CHINESE_CHARS = re.compile(r"[一-鿿]")


def detect_classical_chinese(text: str, threshold: float = 0.3) -> tuple[bool, float]:
    """
    Returns (is_flagged, confidence).
    Flags text whose Chinese character density exceeds threshold.
    Regex-only — no langdetect, which is non-deterministic and misclassifies
    classical Chinese as Korean/Japanese on the same input across runs.
    """
    if not text or not text.strip():
        return False, 0.0
    chinese_chars = len(_CHINESE_CHARS.findall(text))
    if chinese_chars < 5:
        return False, 0.0
    density = chinese_chars / max(len(text), 1)
    if density >= threshold:
        return True, round(density, 2)
    return False, round(density, 2)
