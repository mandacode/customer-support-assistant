import re
import unicodedata

# Function words that carry no meaning on their own (Polish + English).
STOPWORDS = {
    "a", "i", "w", "we", "z", "ze", "na", "do", "od", "o", "u", "po", "za", "dla", "oraz", "lub", "the",
    "and", "or", "of", "for", "to", "in", "on", "with",
}

# Polish inflects word endings ("opony", "opon", "oponami"), so two words match
# when they share a beginning as long as the shorter one (capped at STEM_LENGTH)
# and at least MIN_STEM letters. Shorter words must match exactly.
STEM_LENGTH = 5
MIN_STEM = 4


def stems(text: str) -> list[str]:
    words = re.findall(r"\w+", fold(text))
    return [w for w in words if w not in STEMS_SKIP and (len(w) > 2 or w.isdigit())]


def is_grounded(phrase: str, source: str, min_ratio: float = 1.0) -> bool:
    """True if enough of the phrase's meaningful words occur in the source text.

    Deterministic, no LLM: used to reject names or facts the model made up.
    """
    phrase_stems = stems(phrase)
    if not phrase_stems:
        return False
    source_words = set(re.findall(r"\w+", fold(source)))
    found = sum(1 for word in phrase_stems if any(_same_word(word, other) for other in source_words))
    return found / len(phrase_stems) >= min_ratio


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    needed = min(len(a), len(b), STEM_LENGTH)
    return needed >= MIN_STEM and a[:needed] == b[:needed]


def fold(text: str) -> str:
    """Lowercase without diacritics, so "kół" and "kol" compare equal."""
    text = text.lower().replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


STEMS_SKIP = {fold(word) for word in STOPWORDS}
