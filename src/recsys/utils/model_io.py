from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_VERSION = 1


def _prepare_output_path(
    path: str | Path,
) -> Path:
    """Normalize an output path and create its parent directory."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_path


def _validate_existing_file(
    path: str | Path,
) -> Path:
    """Validate that an artifact path points to an existing file."""

    artifact_path = Path(path)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {artifact_path}")

    if not artifact_path.is_file():
        raise ValueError(f"Artifact path is not a file: {artifact_path}")

    return artifact_path


def _fsync_directory(
    directory: Path,
) -> None:
    """
    Best-effort fsync of a directory after an atomic replacement.

    This improves durability of the directory-entry update on
    POSIX filesystems. Some platforms/filesystems do not support
    directory fsync, so failure is intentionally ignored.
    """

    try:
        descriptor = os.open(
            directory,
            os.O_RDONLY,
        )

    except OSError:
        return

    try:
        os.fsync(descriptor)

    except OSError:
        pass

    finally:
        os.close(descriptor)


def _atomic_torch_save(
    value: object,
    path: str | Path,
) -> None:
    """
    Persist a torch artifact using same-directory atomic replacement.

    The complete artifact is first written and fsynced to a temporary
    file. Only after the write succeeds is the destination replaced
    with os.replace().

    Therefore a failed write cannot leave a partially serialized
    destination file.
    """

    output_path = _prepare_output_path(path)

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )

    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            temporary_fd,
            "wb",
        ) as file:
            torch.save(
                value,
                file,
            )

            file.flush()

            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            output_path,
        )

        _fsync_directory(output_path.parent)

    except Exception:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)

        raise


def _validate_checkpoint(
    checkpoint: object,
) -> dict[str, Any]:
    """Validate the minimum structure of a model checkpoint."""

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError("Model checkpoint must be a dictionary.")

    required_keys = {
        "checkpoint_version",
        "model_state_dict",
        "model_config",
        "metrics",
        "training_config",
    }

    missing_keys = required_keys - set(checkpoint)

    if missing_keys:
        raise ValueError(
            f"Model checkpoint is missing required keys: {sorted(missing_keys)}"
        )

    checkpoint_version = checkpoint["checkpoint_version"]

    if not isinstance(
        checkpoint_version,
        int,
    ):
        raise TypeError("checkpoint['checkpoint_version'] must be an integer.")

    if checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported checkpoint version: "
            f"expected={CHECKPOINT_VERSION}, "
            f"actual={checkpoint_version}."
        )

    model_state_dict = checkpoint["model_state_dict"]

    model_config = checkpoint["model_config"]

    metrics = checkpoint["metrics"]

    training_config = checkpoint["training_config"]

    if not isinstance(
        model_state_dict,
        dict,
    ):
        raise TypeError("checkpoint['model_state_dict'] must be a dictionary.")

    if not isinstance(
        model_config,
        dict,
    ):
        raise TypeError("checkpoint['model_config'] must be a dictionary.")

    if not isinstance(
        metrics,
        dict,
    ):
        raise TypeError("checkpoint['metrics'] must be a dictionary.")

    if not isinstance(
        training_config,
        dict,
    ):
        raise TypeError("checkpoint['training_config'] must be a dictionary.")

    return checkpoint


def build_model_checkpoint(
    model: nn.Module,
    model_config: dict[str, Any],
    metrics: dict[str, float] | None = None,
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a portable model-checkpoint payload without writing it.

    Keeping checkpoint construction separate from persistence is useful
    for staged release publication and validation.
    """

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError("model must be a torch.nn.Module.")

    if not isinstance(
        model_config,
        dict,
    ):
        raise TypeError("model_config must be a dictionary.")

    if metrics is not None and not isinstance(
        metrics,
        dict,
    ):
        raise TypeError("metrics must be a dictionary or None.")

    if training_config is not None and not isinstance(
        training_config,
        dict,
    ):
        raise TypeError("training_config must be a dictionary or None.")

    state_dict = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }

    checkpoint = {
        "checkpoint_version": (CHECKPOINT_VERSION),
        "model_state_dict": (state_dict),
        "model_config": dict(model_config),
        "metrics": dict(metrics or {}),
        "training_config": dict(training_config or {}),
    }

    return _validate_checkpoint(checkpoint)


def save_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    model_config: dict[str, Any],
    metrics: dict[str, float] | None = None,
    training_config: dict[str, Any] | None = None,
) -> None:
    """
    Atomically persist model weights and experiment metadata.

    Model tensors are moved to CPU before serialization so checkpoints
    remain portable across CPU, CUDA, and MPS environments.

    The destination is replaced only after complete serialization
    succeeds.
    """

    checkpoint = build_model_checkpoint(
        model=model,
        model_config=model_config,
        metrics=metrics,
        training_config=training_config,
    )

    _atomic_torch_save(
        checkpoint,
        path,
    )


def load_model_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    Load and validate a persisted model checkpoint.

    `weights_only=True` is used because project checkpoints contain
    tensors and standard metadata rather than arbitrary serialized
    Python objects.
    """

    checkpoint_path = _validate_existing_file(path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    return _validate_checkpoint(checkpoint)


def restore_model_from_checkpoint(
    model: nn.Module,
    checkpoint: dict[str, Any],
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    """
    Restore model weights from an already-loaded checkpoint.

    This avoids reading the checkpoint from disk a second time when
    model configuration has already been inspected.
    """

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError("model must be a torch.nn.Module.")

    checkpoint = _validate_checkpoint(checkpoint)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    model = model.to(device)

    model.eval()

    return model


def save_tensor(
    tensor: torch.Tensor,
    path: str | Path,
) -> None:
    """
    Atomically persist a tensor artifact on CPU.

    The existing destination remains untouched if serialization fails.
    """

    if not isinstance(
        tensor,
        torch.Tensor,
    ):
        raise TypeError("tensor must be a torch.Tensor.")

    cpu_tensor = tensor.detach().cpu()

    _atomic_torch_save(
        cpu_tensor,
        path,
    )


def load_tensor(
    path: str | Path,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Load and validate a persisted tensor artifact."""

    tensor_path = _validate_existing_file(path)

    tensor = torch.load(
        tensor_path,
        map_location=(device if device is not None else "cpu"),
        weights_only=True,
    )

    if not isinstance(
        tensor,
        torch.Tensor,
    ):
        raise TypeError(f"Expected a tensor artifact: {tensor_path}")

    return tensor
