"""Token-level F1 overlap scorer."""
import math

from aieval.scorers import token_f1


def test_exact_overlap_scores_one():
    assert token_f1("the cat sat", "the cat sat") == 1.0


def test_disjoint_scores_zero():
    assert token_f1("alpha beta", "gamma delta") == 0.0


def test_empty_sides_score_zero():
    assert token_f1("", "anything") == 0.0
    assert token_f1("anything", "") == 0.0
    assert token_f1("", "") == 0.0


def test_order_independent():
    # rouge_l would penalise reordering; token F1 does not.
    assert token_f1("cat sat the", "the cat sat") == 1.0


def test_partial_overlap_is_harmonic_mean_of_precision_and_recall():
    # prediction has 3 tokens, reference 2, two tokens shared.
    # precision 2/3, recall 2/2 -> F1 = 2*pr/(p+r) = 0.8
    score = token_f1("the cat extra", "the cat")
    assert math.isclose(score, 0.8)


def test_repeated_tokens_counted_by_multiplicity():
    # A repeated correct word cannot inflate the score beyond its reference count.
    # pred has "the the cat" (3 tokens), ref "the cat" (2 tokens).
    # shared "the" counts once (min multiplicity), "cat" once -> shared = 2.
    # precision 2/3, recall 2/2 -> 0.8
    assert math.isclose(token_f1("the the cat", "the cat"), 0.8)


def test_case_insensitive_by_default_and_sensitive_on_request():
    assert token_f1("The Cat", "the cat") == 1.0
    assert token_f1("The Cat", "the cat", case_sensitive=True) == 0.0
