"""Train and validate Two-Tower retrieval and publish an isolated selection release.

Run from the repository root:
    uv run python scripts/04_two_tower_retrieval.py --run-id retrieval-001

Requires the validation-only neural CF report from script 03 for the same run ID.
Writes outputs under artifacts/reproductions/<run-id>/, without modifying
canonical model releases under artifacts/models/.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import torch

from recsys.config import (
    ARTIFACT_MANIFEST_PATH,
    EVAL_BATCH_SIZE,
    ITEM_MAPPING_PATH,
    MODEL_DIR,
    PREPROCESSING_PROTOCOL_VERSION,
    RANDOM_SEED,
    RANKING_CANDIDATE_SIZE,
    RETRIEVAL_EMBEDDING_DIM,
    RETRIEVAL_EPOCHS,
    RETRIEVAL_HIDDEN_DIM,
    RETRIEVAL_KS,
    RETRIEVAL_LEARNING_RATE,
    RETRIEVAL_WEIGHT_DECAY,
    TRAIN_BATCH_SIZE,
    TRAIN_PATH,
    USER_MAPPING_PATH,
    VALIDATION_PATH,
)
from recsys.data.loader import load_json, load_parquet, save_json
from recsys.data.sampling import UniformNegativeSampler
from recsys.evaluation.evaluator import (
    benchmark_embedding_retrieval,
    build_seen_items_csr,
    build_target_array,
    evaluate_embedding_retrieval,
)
from recsys.models.two_tower import TwoTowerRecommender
from recsys.training import train_bpr_epoch
from recsys.utils.artifacts import (
    canonical_id_mapping_sha256,
    canonical_json_sha256,
    sha256_file,
)
from recsys.utils.helpers import get_torch_device, set_seed
from recsys.utils.model_io import save_model_checkpoint, save_tensor


def show(title: str, value: object) -> None:
    print(f"\n{'=' * 14} {title} {'=' * 14}")
    print(value)


def validate_data_identity(
    manifest: dict,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
) -> dict:
    """Compute a validation-only identity using training and validation artifacts."""
    if manifest.get("preprocessing_protocol_version") != PREPROCESSING_PROTOCOL_VERSION:
        raise RuntimeError("Processed-data preprocessing protocol mismatch.")

    processed = manifest.get("processed_artifacts")
    if not isinstance(processed, dict):
        raise TypeError("artifact_manifest.processed_artifacts must be a dictionary.")

    paths = {
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "user_mapping": USER_MAPPING_PATH,
        "item_mapping": ITEM_MAPPING_PATH,
    }
    hashes = {}
    for name, path in paths.items():
        metadata = processed.get(name)
        if not isinstance(metadata, dict):
            raise TypeError(f"Missing processed artifact metadata: {name}.")
        expected = metadata.get("sha256")
        if not isinstance(expected, str):
            raise TypeError(f"Missing processed artifact SHA-256: {name}.")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Processed artifact hash mismatch: {name}.")
        hashes[name] = actual

    semantic = manifest.get("semantic_mapping_hashes")
    if not isinstance(semantic, dict):
        raise TypeError("Semantic mapping hashes must be a dictionary.")
    user_hash = canonical_id_mapping_sha256(user_mapping, "userId", "user_idx")
    item_hash = canonical_id_mapping_sha256(item_mapping, "movieId", "item_idx")
    if user_hash != semantic.get("user_mapping_sha256"):
        raise RuntimeError("User mapping semantic fingerprint mismatch.")
    if item_hash != semantic.get("item_mapping_sha256"):
        raise RuntimeError("Item mapping semantic fingerprint mismatch.")

    payload = {
        "identity_schema_version": 1,
        "preprocessing_protocol_version": PREPROCESSING_PROTOCOL_VERSION,
        "artifact_hashes": hashes,
        "semantic_mapping_hashes": {
            "user_mapping_sha256": user_hash,
            "item_mapping_sha256": item_hash,
        },
    }
    fingerprint = canonical_json_sha256(payload)
    assert "test" not in hashes
    return {
        "validation_data_id": f"validation-data-{fingerprint[:20]}",
        "validation_data_fingerprint": fingerprint,
        "identity": payload,
    }


def validate_ncf_parent(ncf_results: dict, data_identity: dict) -> dict:
    """Validate the upstream neural CF report without reading test data."""
    if ncf_results.get("artifact_schema_version") != 2:
        raise RuntimeError("NCF report schema must be version 2.")
    if ncf_results.get("evaluation_split") != "validation":
        raise RuntimeError("NCF report is not validation-only.")
    if ncf_results.get("selected_ranker") != "neumf":
        raise RuntimeError("NeuMF is not the selected neural ranker.")
    if ncf_results.get("validation_data_identity") != data_identity:
        raise RuntimeError("NCF validation data identity does not match.")

    parent = ncf_results.get("validation_experiment_identity")
    if not isinstance(parent, dict):
        raise TypeError("NCF validation experiment identity is missing.")
    payload = parent.get("identity")
    if not isinstance(payload, dict):
        raise TypeError("NCF validation experiment payload is missing.")
    fingerprint = canonical_json_sha256(payload)
    if parent.get("validation_experiment_fingerprint") != fingerprint:
        raise RuntimeError("NCF validation experiment fingerprint is invalid.")
    if (
        payload.get("validation_data_fingerprint")
        != data_identity["validation_data_fingerprint"]
    ):
        raise RuntimeError("NCF experiment is bound to different validation data.")
    return parent


def build_retrieval_identity(data_identity: dict, parent: dict) -> tuple[dict, dict]:
    """Build the deterministic retrieval selection configuration and fingerprint."""
    config = {
        "experiment_family": "two_tower_retrieval",
        "evaluation_split": "validation",
        "training_data": "train",
        "objective": "BPR",
        "optimizer": "AdamW",
        "negative_sampling": "uniform",
        "negatives_per_positive": 1,
        "embedding_dim": RETRIEVAL_EMBEDDING_DIM,
        "hidden_dim": RETRIEVAL_HIDDEN_DIM,
        "epochs": RETRIEVAL_EPOCHS,
        "learning_rate": RETRIEVAL_LEARNING_RATE,
        "weight_decay": RETRIEVAL_WEIGHT_DECAY,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "retrieval_depths": list(RETRIEVAL_KS),
        "random_seed": RANDOM_SEED,
        "candidate_policy": {
            "policy_type": "fixed_configuration",
            "candidate_size": RANKING_CANDIDATE_SIZE,
            "config_name": "RANKING_CANDIDATE_SIZE",
        },
    }
    payload = {
        "identity_schema_version": 1,
        "validation_data_fingerprint": data_identity["validation_data_fingerprint"],
        "parent_validation_experiment_fingerprint": (
            parent["validation_experiment_fingerprint"]
        ),
        "selection_config": config,
    }
    fingerprint = canonical_json_sha256(payload)
    identity = {
        "validation_experiment_id": f"validation-two-tower-{fingerprint[:20]}",
        "validation_experiment_fingerprint": fingerprint,
        "identity": payload,
    }
    return config, identity


def verify_selection_release(
    release_dir: Path,
    manifest: dict,
    data_identity: dict,
    parent: dict,
    user_shape: list[int],
    item_shape: list[int],
) -> None:
    """Recheck the published manifest, checkpoint and both embeddings."""
    published = load_json(release_dir / "selection_release_manifest.json")
    if published != manifest:
        raise RuntimeError("Published selection manifest differs from staged manifest.")
    identity = published.get("identity")
    if not isinstance(identity, dict):
        raise TypeError("Published selection release identity is missing.")
    if canonical_json_sha256(identity) != published["selection_release_fingerprint"]:
        raise RuntimeError("Published selection release fingerprint is invalid.")
    if published.get("validation_data_identity") != data_identity:
        raise RuntimeError("Selection release has wrong validation data identity.")
    if published.get("parent_validation_experiment_identity") != parent:
        raise RuntimeError("Selection release has wrong parent experiment.")

    checkpoint = identity["checkpoint"]
    checkpoint_path = release_dir / checkpoint["filename"]
    if sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise RuntimeError("Published Two-Tower checkpoint SHA-256 mismatch.")

    embeddings = identity["embeddings"]
    for name, shape in (("user", user_shape), ("item", item_shape)):
        metadata = embeddings[name]
        path = release_dir / metadata["filename"]
        if sha256_file(path) != metadata["sha256"]:
            raise RuntimeError(f"Published {name} embedding SHA-256 mismatch.")
        if metadata["parent_checkpoint_sha256"] != checkpoint["sha256"]:
            raise RuntimeError(f"Published {name} embedding parent mismatch.")
        if metadata["shape"] != shape:
            raise RuntimeError(f"Published {name} embedding shape mismatch.")
        if metadata["embedding_dim"] != RETRIEVAL_EMBEDDING_DIM:
            raise RuntimeError(f"Published {name} embedding dimension mismatch.")
    print(f"Validated selection release: {published['selection_release_id']}")


def require_matching_release_manifest(existing: dict, expected: dict) -> None:
    """Reject a release ID collision with different manifest contents."""
    if existing != expected:
        raise RuntimeError("Existing selection release manifest mismatch.")


def publish_selection_release(
    root: Path,
    model: TwoTowerRecommender,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    metrics: dict[str, float],
    data_identity: dict,
    parent: dict,
    selection_config: dict,
    experiment_identity: dict,
) -> tuple[Path, dict]:
    """Publish a checkpoint-bound immutable selection release within this run."""
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".two-tower-selection-", dir=root))
    try:
        checkpoint_path = staging / "two_tower.pt"
        user_path = staging / "two_tower_user_embeddings.pt"
        item_path = staging / "two_tower_item_embeddings.pt"
        fingerprint = experiment_identity["validation_experiment_fingerprint"]
        save_model_checkpoint(
            model=model,
            path=checkpoint_path,
            model_config={
                "model": "TwoTowerRecommender",
                "n_users": model.n_users,
                "n_items": model.n_items,
                "embedding_dim": RETRIEVAL_EMBEDDING_DIM,
                "hidden_dim": RETRIEVAL_HIDDEN_DIM,
            },
            metrics=metrics,
            training_config={
                **selection_config,
                "validation_data_id": data_identity["validation_data_id"],
                "validation_data_fingerprint": data_identity[
                    "validation_data_fingerprint"
                ],
                "validation_experiment_id": experiment_identity[
                    "validation_experiment_id"
                ],
                "validation_experiment_fingerprint": fingerprint,
                "parent_validation_experiment_id": parent["validation_experiment_id"],
            },
        )
        save_tensor(user_embeddings, user_path)
        save_tensor(item_embeddings, item_path)
        checkpoint_hash = sha256_file(checkpoint_path)
        user_hash = sha256_file(user_path)
        item_hash = sha256_file(item_path)

        payload = {
            "identity_schema_version": 1,
            "validation_data_fingerprint": data_identity["validation_data_fingerprint"],
            "validation_experiment_fingerprint": fingerprint,
            "parent_validation_experiment_fingerprint": parent[
                "validation_experiment_fingerprint"
            ],
            "n_users": model.n_users,
            "n_items": model.n_items,
            "candidate_policy": {
                "policy_type": "fixed_configuration",
                "candidate_size": RANKING_CANDIDATE_SIZE,
            },
            "checkpoint": {
                "filename": checkpoint_path.name,
                "sha256": checkpoint_hash,
                "model": "TwoTowerRecommender",
            },
            "embeddings": {
                "user": {
                    "filename": user_path.name,
                    "sha256": user_hash,
                    "parent_checkpoint_sha256": checkpoint_hash,
                    "shape": list(user_embeddings.shape),
                    "embedding_dim": RETRIEVAL_EMBEDDING_DIM,
                },
                "item": {
                    "filename": item_path.name,
                    "sha256": item_hash,
                    "parent_checkpoint_sha256": checkpoint_hash,
                    "shape": list(item_embeddings.shape),
                    "embedding_dim": RETRIEVAL_EMBEDDING_DIM,
                },
            },
        }
        release_fingerprint = canonical_json_sha256(payload)
        release_id = f"two-tower-selection-{release_fingerprint[:20]}"
        release_manifest = {
            "artifact_schema_version": 1,
            "selection_release_id": release_id,
            "selection_release_fingerprint": release_fingerprint,
            "validation_data_identity": data_identity,
            "parent_validation_experiment_identity": parent,
            "validation_experiment_identity": experiment_identity,
            "identity": payload,
        }
        save_json(release_manifest, staging / "selection_release_manifest.json")
        release_dir = root / release_id
        if release_dir.exists():
            existing = load_json(release_dir / "selection_release_manifest.json")
            require_matching_release_manifest(existing, release_manifest)
            shutil.rmtree(staging)
        else:
            os.replace(staging, release_dir)

        verify_selection_release(
            release_dir,
            release_manifest,
            data_identity,
            parent,
            list(user_embeddings.shape),
            list(item_embeddings.shape),
        )
        pointer = {
            "pointer_schema_version": 1,
            "selection_release_id": release_id,
            "selection_release_fingerprint": release_fingerprint,
        }
        temporary_pointer = root / ".current_two_tower_selection.json.tmp"
        save_json(pointer, temporary_pointer)
        os.replace(temporary_pointer, root / "current_two_tower_selection.json")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Selection release: {release_id}")
    print(f"Checkpoint SHA-256: {checkpoint_hash}")
    print(f"User embedding SHA-256: {user_hash}")
    print(f"Item embedding SHA-256: {item_hash}")
    print(f"Release directory: {release_dir}")
    return release_dir, release_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Same run ID as script 03")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id) is None:
        parser.error("--run-id must have 1-64 letters, digits, hyphens or underscores")

    run_dir = MODEL_DIR.parent / "reproductions" / args.run_id
    ncf_path = run_dir / "ncf_results.json"
    results_path = run_dir / "two_tower_results.json"
    history_path = run_dir / "two_tower_training_history.json"
    selection_root = run_dir / "selection_releases"
    pointer_path = selection_root / "current_two_tower_selection.json"
    for path in (results_path, history_path, pointer_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite reproduction: {path}")
    ncf_results = load_json(ncf_path)

    set_seed(RANDOM_SEED)
    device = get_torch_device()
    print(f"Device: {device} | Training batch: {TRAIN_BATCH_SIZE:,}")
    print(f"Eval batch: {EVAL_BATCH_SIZE:,}")
    print(f"Embedding: {RETRIEVAL_EMBEDDING_DIM} | Hidden: {RETRIEVAL_HIDDEN_DIM}")
    print(f"Epochs: {RETRIEVAL_EPOCHS} | Candidate K: {RETRIEVAL_KS}")
    print(f"Run directory: {run_dir}")
    print("Validation-only; held-out test data will not be loaded.")

    train = load_parquet(TRAIN_PATH)
    validation = load_parquet(VALIDATION_PATH)
    user_mapping = load_parquet(USER_MAPPING_PATH)
    item_mapping = load_parquet(ITEM_MAPPING_PATH)
    n_users, n_items = user_mapping.height, item_mapping.height
    manifest = load_json(ARTIFACT_MANIFEST_PATH)
    data_identity = validate_data_identity(manifest, user_mapping, item_mapping)
    parent = validate_ncf_parent(ncf_results, data_identity)
    print(f"Validation data ID: {data_identity['validation_data_id']}")
    print(f"Parent NCF experiment: {parent['validation_experiment_id']}")

    users = train["user_idx"].to_numpy().astype(np.int64, copy=False)
    items = train["item_idx"].to_numpy().astype(np.int64, copy=False)
    seen = build_seen_items_csr(train, n_users=n_users)
    targets = build_target_array(validation, n_users=n_users)
    cold = int(np.sum(targets < 0))
    warm = targets.size - cold
    print(f"Training interactions: {users.size:,} | Users: {n_users:,}")
    print(f"Items: {n_items:,} | Warm targets: {warm:,} | Cold: {cold:,}")

    set_seed(RANDOM_SEED)
    model = TwoTowerRecommender(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=RETRIEVAL_EMBEDDING_DIM,
        hidden_dim=RETRIEVAL_HIDDEN_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=RETRIEVAL_LEARNING_RATE,
        weight_decay=RETRIEVAL_WEIGHT_DECAY,
    )
    print(f"Two-Tower parameters: {sum(p.numel() for p in model.parameters()):,}")
    sampler = UniformNegativeSampler(users, items, n_items, seed=RANDOM_SEED)
    history = []
    for epoch in range(1, RETRIEVAL_EPOCHS + 1):
        loss = train_bpr_epoch(
            model=model,
            optimizer=optimizer,
            user_ids=users,
            item_ids=items,
            negative_sampler=sampler,
            batch_size=TRAIN_BATCH_SIZE,
            device=device,
            seed=RANDOM_SEED + epoch,
            desc=f"Two-Tower epoch {epoch}",
        )
        history.append({"epoch": epoch, "train_loss": loss})
        print(f"Epoch {epoch:02d} | train_loss={loss:.3f}")

    item_embeddings = model.get_item_embeddings(device=device, batch_size=4096)
    print(f"Item embedding shape: {tuple(item_embeddings.shape)}")
    retrieval_metrics = evaluate_embedding_retrieval(
        model=model,
        item_embeddings=item_embeddings,
        target_items=targets,
        seen_items=seen,
        n_users=n_users,
        batch_size=EVAL_BATCH_SIZE,
        ks=RETRIEVAL_KS,
        device=device,
        desc="Two-Tower validation",
    )
    summary = pl.DataFrame(
        [
            {
                "candidate_size": k,
                "candidate_recall": retrieval_metrics[f"recall@{k}"],
                "candidate_hitrate": retrieval_metrics[f"hitrate@{k}"],
                "ndcg": retrieval_metrics[f"ndcg@{k}"],
            }
            for k in RETRIEVAL_KS
        ]
    ).with_columns(pl.exclude("candidate_size").round(3))
    show("Retrieval quality", summary)

    benchmark = benchmark_embedding_retrieval(
        model=model,
        item_embeddings=item_embeddings,
        batch_size=EVAL_BATCH_SIZE,
        device=device,
    )
    show("Exact retrieval benchmark", benchmark)
    candidate_reduction = pl.DataFrame(
        [
            {
                "candidate_size": k,
                "catalog_size": n_items,
                "catalog_retained_pct": k / n_items * 100,
                "catalog_reduction_pct": (1 - k / n_items) * 100,
                "candidate_recall": retrieval_metrics[f"recall@{k}"],
            }
            for k in RETRIEVAL_KS
        ]
    ).with_columns(pl.exclude("candidate_size", "catalog_size").round(3))
    show("Candidate reduction", candidate_reduction)

    speed = ncf_results["inference"]["neumf"]["million_scores_per_second"] * 1e6
    print(f"NeuMF throughput: {speed / 1e6:.3f}M scores/s")
    if RANKING_CANDIDATE_SIZE not in RETRIEVAL_KS:
        raise RuntimeError("Ranking candidate size is not among evaluated depths.")
    tradeoff = pl.DataFrame(
        [
            {
                "candidate_size": k,
                "candidate_recall": retrieval_metrics[f"recall@{k}"],
                "catalog_retained_pct": k / n_items * 100,
                "ranking_scores_millions": n_users * k / 1e6,
                "estimated_neumf_seconds": n_users * k / speed,
            }
            for k in RETRIEVAL_KS
        ]
    ).with_columns(pl.exclude("candidate_size").round(3))
    show("Retrieval and ranking tradeoff", tradeoff)
    print(f"Configured candidate size: {RANKING_CANDIDATE_SIZE:,}")
    print(
        "Configured candidate recall: "
        f"{retrieval_metrics[f'recall@{RANKING_CANDIDATE_SIZE}']:.3f}"
    )
    print(f"Catalog reduction: {(1 - RANKING_CANDIDATE_SIZE / n_items) * 100:.3f}%")

    user_embeddings = model.get_user_embeddings(device=device, batch_size=4096)
    print(f"User embedding shape: {tuple(user_embeddings.shape)}")
    config, experiment_identity = build_retrieval_identity(data_identity, parent)
    release_dir, release_manifest = publish_selection_release(
        root=selection_root,
        model=model,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        metrics=retrieval_metrics,
        data_identity=data_identity,
        parent=parent,
        selection_config=config,
        experiment_identity=experiment_identity,
    )
    report = {
        "artifact_schema_version": 2,
        "evaluation_split": "validation",
        "validation_data_identity": data_identity,
        "parent_validation_experiment_identity": parent,
        "validation_experiment_identity": experiment_identity,
        "selection_release": {
            "selection_release_id": release_manifest["selection_release_id"],
            "selection_release_fingerprint": release_manifest[
                "selection_release_fingerprint"
            ],
            "manifest_path": str(
                (release_dir / "selection_release_manifest.json").relative_to(run_dir)
            ),
        },
        "validation_population": {
            "users": n_users,
            "items": n_items,
            "warm_targets": warm,
            "cold_targets": cold,
        },
        "validation": {
            "metrics": retrieval_metrics,
            "candidate_policy": {
                "policy_type": "fixed_configuration",
                "candidate_size": RANKING_CANDIDATE_SIZE,
            },
        },
        "inference": benchmark,
        "candidate_reduction": candidate_reduction.to_dicts(),
        "candidate_tradeoff": tradeoff.to_dicts(),
        "embedding_dim": RETRIEVAL_EMBEDDING_DIM,
        "user_embedding_shape": list(user_embeddings.shape),
        "item_embedding_shape": list(item_embeddings.shape),
    }
    save_json(report, results_path)
    save_json({"two_tower": history}, history_path)
    print(f"Validation experiment: {experiment_identity['validation_experiment_id']}")
    print(f"Saved retrieval results: {results_path}")
    print(f"Saved training history: {history_path}")


if __name__ == "__main__":
    main()
