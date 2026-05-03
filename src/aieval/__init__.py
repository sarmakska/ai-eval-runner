"""ai-eval-runner — evals as code."""
from .core.runner import run
from .core.scorer import scorer
from .core import dataset

__version__ = "1.0.0"
__all__ = ["run", "scorer", "dataset"]
