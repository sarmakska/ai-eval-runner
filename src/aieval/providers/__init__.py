from .sarmalink import sarmalink_completion
from .openai import openai_completion


def get_provider(name: str):
    if name == "openai":
        return openai_completion
    return sarmalink_completion
