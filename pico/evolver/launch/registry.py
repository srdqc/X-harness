"""Bench registry: name -> ``module:function`` building a BenchBundle."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional, Union

BENCHES: dict[str, str] = {
    "appworld": "benchmarks.appworld.evolve.entry:build",
}


def _module_belongs_to(module: ModuleType, root: Path) -> bool:
    locations = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(Path(module_file))
    module_path = getattr(module, "__path__", None)
    if module_path:
        locations.extend(Path(path) for path in module_path)
    if not locations:
        return False
    for location in locations:
        try:
            location.resolve().relative_to(root)
        except ValueError:
            return False
    return True


def _evict_conflicting_package(package: str, root: Path) -> None:
    cached = {
        name: module
        for name, module in sys.modules.items()
        if (name == package or name.startswith(f"{package}.")) and isinstance(module, ModuleType)
    }
    if cached and not all(_module_belongs_to(module, root) for module in cached.values()):
        for name in cached:
            del sys.modules[name]


def load_bench(name: str, repo_root: Optional[Union[str, Path]] = None) -> Callable:
    """Import the bench plugin registered under ``name``.

    Bench plugins live in the subject repo (repo-root ``benchmarks/``), not in
    the installed Pico package, so ``repo_root`` — the subject checkout — is
    put first on ``sys.path`` before importing. Omitting it only works when
    the checkout root is already importable (e.g. cwd is the checkout).
    """
    target = BENCHES.get(name)
    if target is None:
        raise ValueError(
            f"unknown bench {name!r}; registered: {sorted(BENCHES)} (add yours to pico.evolver.launch.registry.BENCHES)"
        )
    mod_name, _, fn_name = target.partition(":")
    if repo_root is not None:
        resolved_root = Path(repo_root).resolve()
        root = str(resolved_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        subject_entry = resolved_root.joinpath(*mod_name.split(".")).with_suffix(".py")
        if subject_entry.is_file():
            _evict_conflicting_package(target.partition(".")[0], resolved_root)
    module = import_module(mod_name)
    if repo_root is not None and subject_entry.is_file() and not _module_belongs_to(module, resolved_root):
        raise ImportError(f"bench {name!r} resolved outside repo_root {resolved_root}: {module.__file__}")
    return getattr(module, fn_name)


__all__ = ["BENCHES", "load_bench"]
