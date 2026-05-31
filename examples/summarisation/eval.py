"""Example: summarisation eval."""
from aieval import dataset, run, scorer
from aieval.scorers import rouge_l


@scorer
def length_under_120_words(prediction: str, _expected: str) -> float:
    return 1.0 if len(prediction.split()) <= 120 else 0.0


if __name__ == "__main__":
    run(
        name="summarisation",
        dataset=dataset.jsonl("examples/summarisation/dataset.jsonl"),
        scorers=[rouge_l, length_under_120_words],
        provider="sarmalink",
        model="smart",
    )
