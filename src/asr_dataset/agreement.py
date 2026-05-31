"""Word-level agreement between Gemini text and the second ASR, via a
Levenshtein/SequenceMatcher alignment. Returns a 0..1 score = fraction of
reference words that survive alignment (1 - WER, floored at 0).

Includes Persian-specific normalization so trivial orthographic differences
(Arabic vs Persian yeh/kaf, diacritics, ZWNJ, digits) don't count as errors.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

# Arabic -> Persian canonicalization
_TRANS = {
    "\u064a": "\u06cc",  # Arabic yeh -> Persian yeh
    "\u0649": "\u06cc",  # alef maqsura -> Persian yeh
    "\u0643": "\u06a9",  # Arabic kaf -> Persian kaf
    "\u0629": "\u0647",  # teh marbuta -> heh
    "\u200c": " ",       # ZWNJ -> space
}
_DIACRITICS = re.compile(r"[\u064b-\u0652\u0670]")  # harakat
_AR_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_FA_DIGITS = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_PUNCT = re.compile(r"[^\w\u0600-\u06FF\s]")


def normalize_fa(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for a, b in _TRANS.items():
        text = text.replace(a, b)
    text = text.translate(_AR_DIGITS).translate(_FA_DIGITS)
    text = _DIACRITICS.sub("", text)
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def persian_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    fa = sum(1 for c in letters if "\u0600" <= c <= "\u06FF")
    return fa / len(letters)


def agreement_score(reference: str, hypothesis: Optional[str]) -> Optional[float]:
    """1 - WER style agreement on normalized words. None if no hypothesis."""
    if hypothesis is None:
        return None
    ref = normalize_fa(reference).split()
    hyp = normalize_fa(hypothesis).split()
    if not ref:
        return None
    if not hyp:
        return 0.0
    sm = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched = sum(blk.size for blk in sm.get_matching_blocks())
    return float(matched / len(ref))


def word_disagreements(reference: str, hypothesis: Optional[str]) -> list[str]:
    """Reference words not matched by the hypothesis — candidates for review /
    new-vocabulary mining."""
    if hypothesis is None:
        return []
    ref = normalize_fa(reference).split()
    hyp = normalize_fa(hypothesis).split()
    sm = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    out: list[str] = []
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag in ("replace", "delete"):
            out.extend(ref[i1:i2])
    return out
