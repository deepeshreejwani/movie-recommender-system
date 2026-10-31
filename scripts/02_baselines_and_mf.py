"""Train and validate classical MovieLens recommendation models.

Run from the repository root:
    uv run python scripts/02_baselines_and_mf.py --run-id baseline-001

Reads only training and validation data, never the held-out test split.
Saves run-specific reports and training histories without overwriting existing
reports or canonical model checkpoints.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import polars as pl
import torch

from recsys.config import (
    ARTIFACT_MANIFEST_PATH,
    BPR_EMBEDDING_DIM,
    BPR_EPOCHS,
    BPR_LEARNING_RATE,
    BPR_WEIGHT_DECAY,
    COLD_ITEM_INDEX,
    EVAL_BATCH_SIZE,
    EVAL_KS,
    ITEM_MAPPING_PATH,
    MF_EMBEDDING_DIM,
    MF_EPOCHS,
    MF_LEARNING_RATE,
    MF_WEIGHT_DECAY,
    MODEL_DIR,
    PREPROCESSING_PROTOCOL_VERSION,
    RANDOM_SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_PATH,
    USER_MAPPING_PATH,
    VALIDATION_PATH,
)
from recsys.data.loader import load_json, load_parquet, save_json
from recsys.data.sampling import UniformNegativeSampler
from recsys.evaluation.evaluator import (
    build_seen_items_csr,
    build_target_array,
    evaluate_model,
)
from recsys.models.baseline import PopularityRecommender, RandomRecommender
from recsys.models.matrix_factorization import MatrixFactorization
from recsys.training import train_bpr_epoch, train_pointwise_epoch
from recsys.utils.artifacts import (
    canonical_id_mapping_sha256,
    canonical_json_sha256,
    sha256_file,
)
from recsys.utils.helpers import get_torch_device, set_seed


def display(title: str, value: object) -> None:
    print(f"\n{'=' * 16} {title} {'=' * 16}")
    print(value)


def validation_identities(
    manifest: dict,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
) -> tuple[dict, dict]:
    """Compute validation-only data and experiment identities."""
    if manifest.get("preprocessing_protocol_version") != PREPROCESSING_PROTOCOL_VERSION:
        raise RuntimeError(
            "Processed preprocessing protocol does not match configuration."
        )

    processed_artifacts = manifest.get("processed_artifacts")
    if not isinstance(processed_artifacts, dict):
        raise TypeError("artifact_manifest.processed_artifacts must be a dictionary.")

    paths = {
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "user_mapping": USER_MAPPING_PATH,
        "item_mapping": ITEM_MAPPING_PATH,
    }
    artifact_hashes = {}
    for name, path in paths.items():
        metadata = processed_artifacts.get(name)
        if not isinstance(metadata, dict):
            raise TypeError(f"Missing processed artifact metadata for '{name}'.")
        expected = metadata.get("sha256")
        if not isinstance(expected, str):
            raise TypeError(f"Missing SHA-256 for '{name}'.")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Processed artifact fingerprint mismatch for '{name}'.")
        artifact_hashes[name] = actual

    mapping_hashes = manifest.get("semantic_mapping_hashes")
    if not isinstance(mapping_hashes, dict):
        raise TypeError(
            "artifact_manifest.semantic_mapping_hashes must be a dictionary."
        )

    user_hash = canonical_id_mapping_sha256(user_mapping, "userId", "user_idx")
    item_hash = canonical_id_mapping_sha256(item_mapping, "movieId", "item_idx")
    if user_hash != mapping_hashes.get("user_mapping_sha256"):
        raise RuntimeError("User mapping semantic fingerprint mismatch.")
    if item_hash != mapping_hashes.get("item_mapping_sha256"):
        raise RuntimeError("Item mapping semantic fingerprint mismatch.")

    data_payload = {
        "identity_schema_version": 1,
        "preprocessing_protocol_version": PREPROCESSING_PROTOCOL_VERSION,
        "artifact_hashes": artifact_hashes,
        "semantic_mapping_hashes": {
            "user_mapping_sha256": user_hash,
            "item_mapping_sha256": item_hash,
        },
    }
    data_fingerprint = canonical_json_sha256(data_payload)
    data_identity = {
        "validation_data_id": f"validation-data-{data_fingerprint[:20]}",
        "validation_data_fingerprint": data_fingerprint,
        "identity": data_payload,
    }
    selection_config = {
        "experiment_family": "baseline_and_matrix_factorization",
        "evaluation_split": "validation",
        "evaluation_ks": list(EVAL_KS),
        "evaluation_batch_size": EVAL_BATCH_SIZE,
        "training_batch_size": TRAIN_BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "cold_item_index": COLD_ITEM_INDEX,
        "pointwise_mf": {
            "embedding_dim": MF_EMBEDDING_DIM,
            "epochs": MF_EPOCHS,
            "learning_rate": MF_LEARNING_RATE,
            "weight_decay": MF_WEIGHT_DECAY,
        },
        "bpr_mf": {
            "embedding_dim": BPR_EMBEDDING_DIM,
            "epochs": BPR_EPOCHS,
            "learning_rate": BPR_LEARNING_RATE,
            "weight_decay": BPR_WEIGHT_DECAY,
        },
        "model_selection": {
            "candidate_models": ["Pointwise MF", "BPR MF"],
            "primary_metric": "ndcg@10",
            "sort_order": "descending",
        },
    }
    experiment_payload = {
        "identity_schema_version": 1,
        "validation_data_fingerprint": data_fingerprint,
        "selection_config": selection_config,
    }
    experiment_fingerprint = canonical_json_sha256(experiment_payload)
    experiment_identity = {
        "validation_experiment_id": f"validation-mf-{experiment_fingerprint[:20]}",
        "validation_experiment_fingerprint": experiment_fingerprint,
        "identity": experiment_payload,
    }
    assert "test" not in artifact_hashes
    return data_identity, experiment_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique reproduction run ID")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id) is None:
        parser.error(
            "--run-id must contain 1-64 letters, digits, hyphens or underscores"
        )

    output_dir = MODEL_DIR.parent / "reproductions" / args.run_id
    results_path = output_dir / "baseline_mf_results.json"
    history_path = output_dir / "baseline_mf_training_history.json"
    for path in (results_path, history_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {path}")

    set_seed(RANDOM_SEED)
    device = get_torch_device()
    print(
        f"Device: {device} | Train batch: {TRAIN_BATCH_SIZE:,} "
        f"| Eval batch: {EVAL_BATCH_SIZE:,}"
    )
    print(f"Evaluation K: {EVAL_KS} | Output directory: {output_dir}")
    print("Selection uses validation only; test split is not loaded.")

    train = load_parquet(TRAIN_PATH)
    validation = load_parquet(VALIDATION_PATH)
    user_mapping = load_parquet(USER_MAPPING_PATH)
    item_mapping = load_parquet(ITEM_MAPPING_PATH)
    n_users, n_items = user_mapping.height, item_mapping.height

    cold_targets = validation.filter(pl.col("item_idx") == COLD_ITEM_INDEX).height
    warm_targets = validation.height - cold_targets
    summary = pl.DataFrame(
        [
            {
                "dataset": "train",
                "interactions": train.height,
                "users": train["user_idx"].n_unique(),
                "known_items": train["item_idx"].n_unique(),
                "cold_targets": 0,
                "cold_pct": 0.0,
            },
            {
                "dataset": "validation",
                "interactions": validation.height,
                "users": validation["user_idx"].n_unique(),
                "known_items": validation.filter(pl.col("item_idx") >= 0)[
                    "item_idx"
                ].n_unique(),
                "cold_targets": cold_targets,
                "cold_pct": cold_targets / validation.height * 100,
            },
        ]
    ).with_columns(pl.col("cold_pct").round(3))
    display("Training and validation", summary)

    seen_items = build_seen_items_csr(train, n_users=n_users)
    targets = build_target_array(validation, n_users=n_users)
    assert targets.shape == (n_users,)
    print(
        f"Users: {n_users:,} | Items: {n_items:,} | Seen: {seen_items.indices.size:,}"
    )
    print(
        f"Validation targets: {targets.size:,} "
        f"| Warm: {warm_targets:,} | Cold: {cold_targets:,}"
    )

    popularity = PopularityRecommender(n_items=n_items, device=device).fit(train)
    popularity_metrics = evaluate_model(
        popularity,
        targets,
        seen_items,
        n_users,
        EVAL_BATCH_SIZE,
        EVAL_KS,
        device,
        desc="Popularity validation",
    )
    display("Popularity", {k: round(v, 3) for k, v in popularity_metrics.items()})

    random_model = RandomRecommender(
        n_items=n_items,
        device=device,
        seed=RANDOM_SEED,
    ).fit(train)
    random_metrics = evaluate_model(
        random_model,
        targets,
        seen_items,
        n_users,
        EVAL_BATCH_SIZE,
        EVAL_KS,
        device,
        desc="Random validation",
    )
    display("Random", {k: round(v, 3) for k, v in random_metrics.items()})

    train_users = train["user_idx"].to_numpy().astype(np.int64, copy=False)
    train_items = train["item_idx"].to_numpy().astype(np.int64, copy=False)
    sampler = UniformNegativeSampler(
        train_users,
        train_items,
        n_items,
        seed=RANDOM_SEED,
    )
    print(f"Training interactions: {train_users.size:,}")
    print(f"Batches per epoch: {int(np.ceil(train_users.size / TRAIN_BATCH_SIZE)):,}")

    set_seed(RANDOM_SEED)
    pointwise = MatrixFactorization(n_users, n_items, MF_EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(
        pointwise.parameters(),
        lr=MF_LEARNING_RATE,
        weight_decay=MF_WEIGHT_DECAY,
    )
    pointwise_parameters = sum(p.numel() for p in pointwise.parameters())
    print(f"Pointwise MF: {pointwise_parameters:,} parameters")
    print(
        f"Pointwise epochs: {MF_EPOCHS} "
        f"| LR: {MF_LEARNING_RATE} | WD: {MF_WEIGHT_DECAY}"
    )
    pointwise_history = []
    for epoch in range(1, MF_EPOCHS + 1):
        loss = train_pointwise_epoch(
            model=pointwise,
            optimizer=optimizer,
            user_ids=train_users,
            item_ids=train_items,
            negative_sampler=sampler,
            batch_size=TRAIN_BATCH_SIZE,
            device=device,
            seed=RANDOM_SEED + epoch,
            desc=f"Pointwise MF epoch {epoch}",
        )
        pointwise_history.append({"epoch": epoch, "train_loss": loss})
        print(f"Epoch {epoch:02d} | train_loss={loss:.3f}")
    pointwise_metrics = evaluate_model(
        pointwise,
        targets,
        seen_items,
        n_users,
        EVAL_BATCH_SIZE,
        EVAL_KS,
        device,
        desc="Pointwise MF validation",
    )
    display("Pointwise MF", {k: round(v, 3) for k, v in pointwise_metrics.items()})

    set_seed(RANDOM_SEED)
    bpr = MatrixFactorization(n_users, n_items, BPR_EMBEDDING_DIM).to(device)
    bpr_optimizer = torch.optim.Adam(
        bpr.parameters(),
        lr=BPR_LEARNING_RATE,
        weight_decay=BPR_WEIGHT_DECAY,
    )
    bpr_parameters = sum(p.numel() for p in bpr.parameters())
    print(f"BPR MF: {bpr_parameters:,} parameters")
    print(
        f"BPR epochs: {BPR_EPOCHS} | LR: {BPR_LEARNING_RATE} | WD: {BPR_WEIGHT_DECAY}"
    )
    bpr_sampler = UniformNegativeSampler(
        train_users,
        train_items,
        n_items,
        seed=RANDOM_SEED,
    )
    bpr_history = []
    for epoch in range(1, BPR_EPOCHS + 1):
        loss = train_bpr_epoch(
            model=bpr,
            optimizer=bpr_optimizer,
            user_ids=train_users,
            item_ids=train_items,
            negative_sampler=bpr_sampler,
            batch_size=TRAIN_BATCH_SIZE,
            device=device,
            seed=RANDOM_SEED + epoch,
            desc=f"BPR MF epoch {epoch}",
        )
        bpr_history.append({"epoch": epoch, "train_loss": loss})
        print(f"Epoch {epoch:02d} | train_loss={loss:.3f}")
    bpr_metrics = evaluate_model(
        bpr,
        targets,
        seen_items,
        n_users,
        EVAL_BATCH_SIZE,
        EVAL_KS,
        device,
        desc="BPR MF validation",
    )
    display("BPR MF", {k: round(v, 3) for k, v in bpr_metrics.items()})

    results = pl.DataFrame(
        [
            {"model": "Random", **random_metrics},
            {"model": "Popularity", **popularity_metrics},
            {"model": "Pointwise MF", **pointwise_metrics},
            {"model": "BPR MF", **bpr_metrics},
        ]
    )
    display("All models", results.with_columns(pl.exclude("model").round(3)))

    relative = []
    for name, metrics in [
        ("Pointwise MF", pointwise_metrics),
        ("BPR MF", bpr_metrics),
    ]:
        row = {"model": name}
        for metric in ("recall@10", "mrr@10", "ndcg@10", "recall@50", "ndcg@50"):
            baseline_value = popularity_metrics[metric]
            row[f"{metric}_improvement_pct"] = round(
                (metrics[metric] - baseline_value) / baseline_value * 100,
                3,
            )
        relative.append(row)
    display("Relative improvement over popularity", pl.DataFrame(relative))

    mf_selection = (
        results.filter(pl.col("model").is_in(["Pointwise MF", "BPR MF"]))
        .select("model", "recall@10", "mrr@10", "ndcg@10", "recall@50", "ndcg@50")
        .sort("ndcg@10", descending=True)
    )
    selected = mf_selection.row(0, named=True)["model"]
    display(
        "MF model selection",
        mf_selection.with_columns(pl.exclude("model").round(3)),
    )
    print(f"Selected MF baseline: {selected}")

    manifest = load_json(ARTIFACT_MANIFEST_PATH)
    validation_data_identity, validation_experiment_identity = validation_identities(
        manifest,
        user_mapping,
        item_mapping,
    )
    experiment_results = {
        "artifact_schema_version": 2,
        "evaluation_split": "validation",
        "validation_data_identity": validation_data_identity,
        "validation_experiment_identity": validation_experiment_identity,
        "validation_population": {
            "users": n_users,
            "items": n_items,
            "warm_targets": warm_targets,
            "cold_targets": cold_targets,
            "cold_item_index": COLD_ITEM_INDEX,
        },
        "metrics": {
            "random": random_metrics,
            "popularity": popularity_metrics,
            "pointwise_mf": pointwise_metrics,
            "bpr_mf": bpr_metrics,
        },
        "selected_mf_model": selected,
    }
    for path in (results_path, history_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    save_json(experiment_results, results_path)
    save_json(
        {"pointwise_mf": pointwise_history, "bpr_mf": bpr_history},
        history_path,
    )
    print(f"Validation data ID: {validation_data_identity['validation_data_id']}")
    print(
        "Validation experiment ID: "
        f"{validation_experiment_identity['validation_experiment_id']}"
    )
    print(f"Saved validation report: {results_path}")
    print(f"Saved training histories: {history_path}")


if __name__ == "__main__":
    main()
