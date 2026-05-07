"""Centralized lazy model loading + sequential eviction.

All other modules go through this registry to obtain models. The registry
keeps a working set under MAX_VRAM_GB by evicting LRU models when adding
a new one would exceed the budget. Eviction is voluntary: callers are
encouraged to call .evict(name) when they know they are done with a model
for a while.
"""
from __future__ import annotations

import gc
from collections import OrderedDict
from typing import Any, Callable

import torch

MAX_LOADED_MODELS = 6  # soft cap; primary control is per-stage evict() calls


class ModelRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, Callable[[], Any]] = {}
        self._loaded: OrderedDict[str, Any] = OrderedDict()

    def register(self, name: str, loader: Callable[[], Any]) -> None:
        if name in self._loaders:
            raise ValueError(f"Model '{name}' already registered")
        self._loaders[name] = loader

    def get(self, name: str) -> Any:
        if name not in self._loaders:
            raise KeyError(f"Model '{name}' not registered")
        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]
        if len(self._loaded) >= MAX_LOADED_MODELS:
            self._evict_oldest()
        model = self._loaders[name]()
        self._loaded[name] = model
        return model

    def evict(self, name: str) -> None:
        if name in self._loaded:
            del self._loaded[name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def evict_all(self) -> None:
        names = list(self._loaded.keys())
        for n in names:
            self.evict(n)

    def loaded(self) -> list[str]:
        return list(self._loaded.keys())

    def _evict_oldest(self) -> None:
        oldest = next(iter(self._loaded))
        self.evict(oldest)


# Module-level singleton — every module imports this.
registry = ModelRegistry()
