"""Lightweight ROUGE-L. For full ROUGE use the `rouge-score` package."""
from typing import Sequence


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[n][m]


def rouge_l(prediction: str, expected: str) -> float:
    pred_tokens = prediction.lower().split()
    exp_tokens = expected.lower().split()
    if not pred_tokens or not exp_tokens:
        return 0.0
    lcs = _lcs(pred_tokens, exp_tokens)
    if lcs == 0:
        return 0.0
    p = lcs / len(pred_tokens)
    r = lcs / len(exp_tokens)
    return 2 * p * r / (p + r)
