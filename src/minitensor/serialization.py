"""Portable model state serialization for MiniTensor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import StateDictError
from .nn import Module
from .tensor import Tensor

_METADATA_KEY = "__minitensor_metadata__"


def model_structure(module: Module) -> dict[str, Any]:
    """Return parameter names, shapes, dtypes, and module class names."""
    return {
        "module": type(module).__name__,
        "parameters": {
            name: {"shape": list(parameter.shape), "dtype": parameter.dtype.str}
            for name, parameter in module.named_parameters()
        },
        "modules": [
            {"name": name, "type": type(child).__name__}
            for name, child in module.named_modules()
        ],
    }


def save_state_dict(module: Module, path: str | Path) -> None:
    """Save a module state dictionary and self-describing metadata."""
    destination = Path(path)
    state = module.state_dict()
    metadata = model_structure(module)
    arrays = {name: value.data for name, value in state.items()}
    arrays[_METADATA_KEY] = np.asarray(json.dumps(metadata), dtype=np.str_)
    np.savez_compressed(destination, **arrays)


def load_state_dict(
    module: Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a state dictionary, validating saved metadata before assignment."""
    source = Path(path)
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise StateDictError(f"Could not read state_dict file {source}: {error}") from error
    with archive:
        if _METADATA_KEY not in archive:
            raise StateDictError(f"State_dict file {source} has no MiniTensor metadata.")
        try:
            metadata = json.loads(str(archive[_METADATA_KEY].item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise StateDictError(f"State_dict file {source} has invalid metadata.") from error
        expected_structure = model_structure(module)
        if metadata.get("parameters") != expected_structure["parameters"]:
            raise StateDictError(
                "Incompatible model structure: saved parameter shapes or dtypes differ."
            )
        state = {
            name: Tensor(archive[name], copy=True)
            for name in archive.files
            if name != _METADATA_KEY
        }
    module.load_state_dict(state, strict=strict)
    return metadata


__all__ = ["load_state_dict", "model_structure", "save_state_dict"]
