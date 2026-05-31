"""Example: summarisation eval.

Scores each summary three ways: ROUGE-L overlap against a reference, a length
budget, and an LLM judge graded against a rubric. The dataset is registered so
its content version is recorded across runs.
"""
from aieval import dataset, run, scorer
from aieval.core.dataset import DatasetRegistry
from aieval.scorers import llm_judge, rouge_l


@scorer
def length_under_120_words(prediction: str, _expected: str) -> float:
    return 1.0 if len(prediction.split()) <= 120 else 0.0


faithful = llm_judge(
    rubric="Reward summaries that are faithful to the source and omit nothing important.",
    provider="sarmalink",
    model="smart",
    name="faithfulness",
)


if __name__ == "__main__":
    rows = list(dataset.jsonl("examples/summarisation/dataset.jsonl"))
    DatasetRegistry().register("summarisation", rows)
    run(
        name="summarisation",
        dataset=rows,
        scorers=[rouge_l, length_under_120_words, faithful],
        provider="sarmalink",
        model="smart",
    )
