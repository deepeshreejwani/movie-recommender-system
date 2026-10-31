from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def load_ratings(
    path: str | Path,
) -> pl.DataFrame:
    """Load MovieLens ratings with memory-efficient data types."""

    return pl.read_csv(
        path,
        schema_overrides={
            "userId": pl.Int32,
            "movieId": pl.Int32,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        },
    )


def load_movies(
    path: str | Path,
) -> pl.DataFrame:
    """Load MovieLens movie metadata."""

    return pl.read_csv(
        path,
        schema_overrides={
            "movieId": pl.Int32,
            "title": pl.String,
            "genres": pl.String,
        },
    )


def save_parquet(
    dataframe: pl.DataFrame,
    path: str | Path,
) -> None:
    """Persist a dataframe as compressed Parquet."""

    if not isinstance(
        dataframe,
        pl.DataFrame,
    ):
        raise TypeError("dataframe must be a Polars DataFrame.")

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.write_parquet(
        output_path,
        compression="zstd",
    )


def load_parquet(
    path: str | Path,
) -> pl.DataFrame:
    """Load a processed Parquet dataset."""

    artifact_path = Path(path)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Parquet artifact does not exist: {artifact_path}")

    if not artifact_path.is_file():
        raise ValueError(f"Parquet artifact path is not a file: {artifact_path}")

    return pl.read_parquet(artifact_path)


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate a JSON object."""

    artifact_path = Path(path)

    if not artifact_path.exists():
        raise FileNotFoundError(f"JSON artifact does not exist: {artifact_path}")

    if not artifact_path.is_file():
        raise ValueError(f"JSON artifact path is not a file: {artifact_path}")

    with artifact_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(f"Expected JSON object in: {artifact_path}")

    return data


def save_json(
    data: dict[str, Any],
    path: str | Path,
) -> None:
    """
    Persist a dictionary as deterministic formatted JSON.

    Keys are sorted so equivalent metadata produces stable
    serialized output.
    """

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError("data must be a dictionary.")

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            sort_keys=True,
        )

        file.write("\n")
