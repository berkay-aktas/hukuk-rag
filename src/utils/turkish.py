"""Turkish language utilities shared across retrieval and evaluation."""

import re

# Turkish locale-aware lowercasing: İ → i, I → ı (dotless)
TURKISH_LOWER_MAP = str.maketrans("İIÖÜÇŞĞ", "iıöüçşğ")

TURKISH_STOPWORDS = frozenset({
    "bir", "bu", "da", "de", "ve", "ile", "için", "olan", "olarak", "gibi",
    "daha", "en", "çok", "her", "kadar", "sonra", "önce", "ise", "ya",
    "ne", "nasıl", "neden", "nerede", "kim", "hangi", "o", "şu", "ben",
    "sen", "biz", "siz", "onlar", "mi", "mu", "mü", "mı", "dir", "dır",
    "dur", "dür", "tir", "tır", "tur", "tür", "ki", "ama", "ancak",
    "fakat", "lakin", "veya", "yahut", "hem", "üzere", "göre", "karşı",
    "arasında", "tarafından", "dolayı", "halde", "rağmen", "itibaren",
    "değil", "var", "yok", "eden", "eder", "etti", "oldu",
    "olur", "olmuş", "iken", "olup", "buna", "şöyle",
    "böyle", "öyle", "aynı", "bazı", "birçok", "diğer", "başka",
})

# Pre-compiled patterns for normalization
_RE_PUNCT = re.compile(r'[^\w\s]')
_RE_MULTI_SPACE = re.compile(r'\s+')


def turkish_lower(text: str) -> str:
    """Lowercase text with Turkish locale rules (İ→i, I→ı)."""
    return text.translate(TURKISH_LOWER_MAP).lower()


def normalize_turkish(text: str) -> str:
    """Normalize Turkish text: lowercase, remove punctuation, collapse whitespace."""
    text = turkish_lower(text)
    text = _RE_PUNCT.sub(' ', text)
    text = _RE_MULTI_SPACE.sub(' ', text).strip()
    return text


def turkish_tokenize(text: str) -> list[str]:
    """Tokenize Turkish text with proper lowercasing and stopword removal.

    Args:
        text: Input text.

    Returns:
        List of tokens with stopwords and single-character tokens removed.
    """
    text = normalize_turkish(text)
    return [t for t in text.split() if t not in TURKISH_STOPWORDS and len(t) > 1]


def content_tokens(text: str) -> set[str]:
    """Extract content-bearing token set for retrieval relevance scoring.

    Applies normalization, removes stopwords and tokens with fewer than
    3 characters. Returns a set for efficient overlap computation.

    Args:
        text: Input text.

    Returns:
        Set of content tokens.
    """
    norm = normalize_turkish(text)
    return {t for t in norm.split() if t not in TURKISH_STOPWORDS and len(t) > 2}
