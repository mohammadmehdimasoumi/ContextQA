"""Small shared helpers: exceptions, logging, device selection, softmax."""

from __future__ import annotations

import logging

import numpy as np


class ContextQAError(Exception):
    """Base class for all project-specific errors."""


class InvalidInputError(ContextQAError, ValueError):
    """Raised when the question or context is empty, wrong type, or too long."""


class UnknownModelError(ContextQAError, KeyError):
    """Raised when a model key is not registered in config.MODELS."""


class ModelLoadError(ContextQAError, RuntimeError):
    """Raised when a model/tokenizer cannot be loaded or is unsuitable."""


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once with a compact format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_device(preferred: str | None = None) -> str:
    """Return 'cuda' if available (unless overridden), else 'cpu'."""
    if preferred:
        return preferred
    import torch  # lazy: keeps pure-numpy code and tests torch-free

    return "cuda" if torch.cuda.is_available() else "cpu"


def softmax(values) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    x = np.asarray(values, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
