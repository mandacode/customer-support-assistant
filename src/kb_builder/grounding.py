import bisect
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
    return 1 - len(ungrounded_words(phrase, source)) / len(phrase_stems) >= min_ratio


def ungrounded_words(text: str, source: str) -> list[str]:
    """Meaningful words of `text` that have no matching word in `source`."""
    source_words = set(re.findall(r"\w+", fold(source)))
    return [word for word in stems(text) if not any(_same_word(word, other) for other in source_words)]


def dropped_sentences(source: str, output: str, min_words: int = 6, min_coverage: float = 0.8) -> list[str]:
    """Source sentences whose content is missing from `output`.

    A sentence counts as kept when at least `min_coverage` of its meaningful
    words occur close together in the output (within a window about twice the
    sentence's length), so its words scattered over other paragraphs do not
    hide a dropped sentence. Sentences with fewer than `min_words` meaningful
    words (buttons, navigation, short headings) are ignored.
    """
    output_tokens = _tokens(output)
    dropped = []
    for sentence in _sentences(source):
        words = list(dict.fromkeys(stems(sentence)))
        if len(words) < min_words:
            continue
        if _best_window_coverage(words, output_tokens, span=2 * len(words) + 4) < min_coverage:
            dropped.append(sentence)
    return dropped


def _sentences(text: str) -> list[str]:
    parts = []
    for line in text.splitlines():
        parts += [part.strip() for part in re.split(r"(?<=[.!?;:])\s+", line) if part.strip()]
    return parts


def _best_window_coverage(words: list[str], tokens: list[str], span: int) -> float:
    """Largest share of `words` found within any `span` consecutive tokens."""
    positions = [[i for i, token in enumerate(tokens) if _same_word(word, token)] for word in words]
    starts = sorted({i for found in positions for i in found})
    best = 0
    # The best window can always be shifted to start at a matched word.
    for start in starts:
        found = sum(1 for at in positions if _any_in_range(at, start, start + span))
        best = max(best, found)
        if best == len(words):
            break
    return best / len(words)


def _any_in_range(sorted_positions: list[int], low: int, high: int) -> bool:
    index = bisect.bisect_left(sorted_positions, low)
    return index < len(sorted_positions) and sorted_positions[index] < high


def misplaced_numbers(
    output: str,
    source: str,
    output_window: int = 4,
    source_window: int = 6,
    min_word_ratio: float = 0.85,
) -> list[str]:
    """Numbers whose surroundings in `output` do not match any place in `source`.

    For every number in the output, its neighbours on the same line (other
    numbers and meaningful words within `output_window` tokens) must occur near
    one occurrence of that number in the source: all neighbouring numbers, and
    at least `min_word_ratio` of the words that occur in the source at all (words
    absent from the source are the word-coverage check's job). Numbers written
    next to each other ("600 227 769", `18" | 220`) must also stand next to each
    other, in the same order, in the source. This catches facts that exist in
    the source but were attached to the wrong thing, e.g. a price moved to
    another row of a price list. Returns a description per misplaced number.
    """
    source_tokens = _tokens(source)
    source_words = set(source_tokens)
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(source_tokens):
        if token.isdigit():
            positions.setdefault(token, []).append(index)

    problems = []
    for line in output.splitlines():
        tokens = _tokens(line)
        for index, token in enumerate(tokens):
            if not token.isdigit():
                continue
            around = tokens[max(0, index - output_window) : index] + tokens[index + 1 : index + 1 + output_window]
            near_numbers = [t for t in around if t.isdigit()]
            near_words = [
                t for t in around
                if not t.isdigit() and t not in STEMS_SKIP and len(t) > 2
                and any(_same_word(t, other) for other in source_words)
            ]
            if not any(
                _matches_near(source_tokens, at, source_window, near_numbers, near_words, min_word_ratio)
                for at in positions.get(token, [])
            ):
                problems.append(f"{token} ({' '.join(around)})")

        for run in _number_runs(tokens):
            if not _contains_run(source_tokens, run):
                problems.append(f"order {' '.join(run)}")
    return problems


def _number_runs(tokens: list[str]) -> list[list[str]]:
    """Groups of two or more numbers standing directly next to each other."""
    runs, current = [], []
    for token in tokens + [""]:
        if token.isdigit():
            current.append(token)
            continue
        if len(current) > 1:
            runs.append(current)
        current = []
    return runs


def _contains_run(sequence: list[str], run: list[str]) -> bool:
    return any(sequence[i : i + len(run)] == run for i in range(len(sequence) - len(run) + 1))


def _matches_near(
    tokens: list[str], at: int, window: int, near_numbers: list[str], near_words: list[str], min_ratio: float
) -> bool:
    nearby = tokens[max(0, at - window) : at] + tokens[at + 1 : at + 1 + window]
    if any(number not in nearby for number in near_numbers):
        return False
    if not near_words:
        return True
    found = sum(1 for word in near_words if any(_same_word(word, other) for other in nearby))
    return found / len(near_words) >= min_ratio


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", fold(text))


def numbers(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def emails(text: str) -> set[str]:
    return {e.lower() for e in re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)}


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
