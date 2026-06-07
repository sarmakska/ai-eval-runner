"""Token-level F1 overlap, the standard span-overlap metric from QA evals.

``token_f1`` measures how much the prediction and the reference share at the
word level, independent of order. It is the harmonic mean of token precision
(how much of the prediction is on target) and token recall (how much of the
reference is recovered), counting repeated tokens up to their multiplicity. It
is more forgiving than ``exact_match`` for free-form answers and, unlike
``rouge_l``, ignores word order entirely, so a reordered but correct answer is
not penalised.

The score is 0.0 when either side is empty and 1.0 only when the two token
multisets match exactly. Comparison is lower-cased so casing does not move the
score; pass ``case_sensitive=True`` to keep it.
"""
from __future__ import annotations

from collections import Counter


def token_f1(prediction: str, expected: str, *, case_sensitive: bool = False) -> float:
    """Return the token-level F1 overlap of ``prediction`` against ``expected``.

    Tokens are whitespace-split words. Shared tokens are counted up to the
    smaller of their two multiplicities, so a prediction cannot inflate its
    score by repeating a correct word. Returns a float in the range 0.0 to 1.0.
    """
    if not case_sensitive:
        prediction = prediction.lower()
        expected = expected.lower()
    pred_tokens = prediction.split()
    exp_tokens = expected.split()
    if not pred_tokens or not exp_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(exp_tokens)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0

    precision = shared / len(pred_tokens)
    recall = shared / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)
