"""Focused regression tests for pre-release artifact and CLI validation fixes."""

from __future__ import annotations

import sys

import pytest

import scripts.train as train_script
from recsys.utils.artifacts import validate_sha256


def test_validate_sha256_accepts_lowercase_hex() -> None:
    digest = "a" * 64
    assert validate_sha256(digest, "test_digest") == digest


@pytest.mark.parametrize(
    "invalid_digest",
    [
        " " * 64,
        "a" * 31 + " " + "a" * 32,
        "a" * 63 + "\t",
        "a" * 63 + "g",
        "A" * 64,
        "a" * 63,
    ],
)
def test_validate_sha256_rejects_malformed_digest(
    invalid_digest: str,
) -> None:
    with pytest.raises(ValueError):
        validate_sha256(invalid_digest, "test_digest")


def test_validate_sha256_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        validate_sha256(None, "test_digest")


def test_train_cli_accepts_complete_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--models", "all"])
    assert train_script.parse_args().models == "all"


@pytest.mark.parametrize("model", ["neumf", "two-tower"])
def test_train_cli_rejects_unsupported_partial_release(
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--models", model])

    with pytest.raises(SystemExit) as exc_info:
        train_script.parse_args()

    assert exc_info.value.code == 2
