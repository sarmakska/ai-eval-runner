"""ai-eval-runner. Evals as code."""
from .core import dataset
from .core.pairwise import pairwise
from .core.regression import compare
from .core.runner import run
from .core.scorer import scorer

__version__ = "1.1.0"
__all__ = ["run", "scorer", "dataset", "compare", "pairwise"]
