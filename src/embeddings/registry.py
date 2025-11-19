from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

from .base import EmbeddingResult

LoaderFn = Callable[..., EmbeddingResult]

_REGISTRY: Dict[str, LoaderFn] = {}


def register(name: str) -> Callable[[LoaderFn], LoaderFn]:
    """Decorator used to register embedding loader functions."""

    def decorator(func: LoaderFn) -> LoaderFn:
        if name in _REGISTRY:
            raise ValueError(f"Embedding loader '{name}' already registered.")
        _REGISTRY[name] = func
        return func

    return decorator


def get(name: str) -> LoaderFn:
    """Retrieve a loader by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown embedding loader '{name}'.")
    return _REGISTRY[name]


def load(name: str, **kwargs) -> EmbeddingResult:
    """Convenience wrapper to load an embedding by registry name."""
    loader = get(name)
    return loader(**kwargs)


def list_embeddings() -> Iterable[str]:
    """Return registered embedding names."""
    return sorted(_REGISTRY.keys())

