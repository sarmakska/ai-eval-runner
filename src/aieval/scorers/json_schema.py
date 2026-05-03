import json


def json_valid(prediction: str, _expected: str) -> float:
    try:
        json.loads(prediction)
        return 1.0
    except Exception:
        return 0.0
