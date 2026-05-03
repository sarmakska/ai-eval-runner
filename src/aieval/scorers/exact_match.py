def exact_match(prediction: str, expected: str) -> float:
    return 1.0 if prediction.strip() == expected.strip() else 0.0
