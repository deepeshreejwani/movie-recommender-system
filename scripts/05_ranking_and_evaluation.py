"""Evaluate two-stage ranking on validation and inspect final-test evidence.

Run from the project root using the same run ID as scripts 02-04:
    uv run python scripts/05_ranking_and_evaluation.py --run-id experiment-001

Loads checkpoint-bound validation models and the Two-Tower selection release.
Selects candidate depth using validation metrics and saves a ranking report.
Verifies the canonical model release, processed test file, and final-test report
without retraining models or reevaluating held-out test targets.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl
import torch

from recsys.config import (
    ARTIFACT_MANIFEST_PATH,
    COLD_ITEM_INDEX,
    EVAL_BATCH_SIZE,
    EVAL_KS,
    FINAL_TRAINING_PROTOCOL,
    ITEM_MAPPING_PATH,
    MODEL_DIR,
    NCF_ITEM_SCORE_BATCH_SIZE,
    PREPROCESSING_PROTOCOL_VERSION,
    RANDOM_SEED,
    RANKING_CANDIDATE_SIZE,
    RANKING_RESULTS_PATH,
    RANKING_SCORE_BATCH_SIZE,
    TEST_PATH,
    TRAIN_PATH,
    USER_MAPPING_PATH,
    VALIDATION_PATH,
)
from recsys.data.loader import load_json, load_parquet, save_json
from recsys.evaluation.evaluator import (
    build_seen_items_csr,
    build_target_array,
    evaluate_embedding_retrieval,
    evaluate_reranking_candidate_sizes,
    evaluate_retrieval_reranking,
)
from recsys.evaluation.metrics import round_metric_values
from recsys.models.ncf import NeuMF
from recsys.models.two_tower import TwoTowerRecommender
from recsys.utils.artifacts import (
    canonical_id_mapping_sha256,
    canonical_json_sha256,
    sha256_file,
)
from recsys.utils.helpers import get_torch_device, set_seed
from recsys.utils.model_io import (
    load_model_checkpoint,
    load_tensor,
    restore_model_from_checkpoint,
)
from recsys.utils.releases import resolve_current_model_release

METRICS = ("recall@10", "mrr@10", "ndcg@10", "recall@50", "mrr@50", "ndcg@50")
QUALITY_METRICS = ("recall@10", "ndcg@10", "recall@50", "ndcg@50")
CANDIDATE_SIZES = (100, 250, 500, 1000)


def show(title: str, value: object) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")
    print(value)


def validation_identity(
    manifest: dict,
    mapping_users: pl.DataFrame,
    mapping_items: pl.DataFrame,
) -> dict:
    if manifest.get("preprocessing_protocol_version") != PREPROCESSING_PROTOCOL_VERSION:
        raise RuntimeError("Processed-data preprocessing protocol mismatch.")

    processed = manifest.get("processed_artifacts")

    if not isinstance(processed, dict):
        raise TypeError("Processed-artifact metadata must be a dictionary.")

    hashes = {}

    for name, path in {
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "user_mapping": USER_MAPPING_PATH,
        "item_mapping": ITEM_MAPPING_PATH,
    }.items():
        metadata = processed.get(name)

        if not isinstance(metadata, dict):
            raise TypeError(f"Missing processed metadata: {name}")

        expected = metadata.get("sha256")

        if not isinstance(expected, str):
            raise TypeError(f"Missing processed SHA-256: {name}")

        actual = sha256_file(path)

        if expected != actual:
            raise RuntimeError(f"Processed artifact fingerprint mismatch: {name}")

        hashes[name] = actual

    semantic = manifest.get("semantic_mapping_hashes")

    if not isinstance(semantic, dict):
        raise TypeError("Semantic mapping hashes must be a dictionary.")

    user_hash = canonical_id_mapping_sha256(mapping_users, "userId", "user_idx")
    item_hash = canonical_id_mapping_sha256(mapping_items, "movieId", "item_idx")

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

    assert "test" not in hashes

    fingerprint = canonical_json_sha256(payload)

    return {
        "validation_data_id": f"validation-data-{fingerprint[:20]}",
        "validation_data_fingerprint": fingerprint,
        "identity": payload,
    }


def validate_experiment_identity(
    value: object,
    label: str,
    data_fingerprint: str,
) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{label} validation identity is missing.")

    payload = value.get("identity")

    if not isinstance(payload, dict):
        raise TypeError(f"{label} validation identity payload is missing.")

    actual_fingerprint = canonical_json_sha256(payload)

    if value.get("validation_experiment_fingerprint") != actual_fingerprint:
        raise RuntimeError(f"{label} experiment fingerprint is invalid.")

    if payload.get("validation_data_fingerprint") != data_fingerprint:
        raise RuntimeError(f"{label} experiment belongs to different validation data.")

    return value


def load_selection_release(
    run_dir: Path,
    data_identity: dict,
    device: torch.device,
    n_users: int,
    n_items: int,
) -> tuple:
    """Restore checkpoint-bound Two-Tower validation artifacts."""
    retrieval_results = load_json(run_dir / "two_tower_results.json")

    if retrieval_results.get("artifact_schema_version") != 2:
        raise RuntimeError("Two-Tower retrieval report schema mismatch.")

    if retrieval_results.get("evaluation_split") != "validation":
        raise RuntimeError("Two-Tower retrieval report must be validation-only.")

    if retrieval_results.get("validation_data_identity") != data_identity:
        raise RuntimeError("Two-Tower retrieval validation data identity mismatch.")

    fingerprint = data_identity["validation_data_fingerprint"]

    experiment = validate_experiment_identity(
        retrieval_results.get("validation_experiment_identity"),
        "Two-Tower retrieval",
        fingerprint,
    )

    reference = retrieval_results.get("selection_release")

    if not isinstance(reference, dict):
        raise TypeError("Selection release reference is missing.")

    relative = reference.get("manifest_path")

    if not isinstance(relative, str):
        raise TypeError("Selection release manifest path is missing.")

    relative_path = Path(relative)

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Selection release manifest path is unsafe.")

    manifest_path = run_dir / relative_path
    selection = load_json(manifest_path)

    identity = selection.get("identity")

    if not isinstance(identity, dict):
        raise TypeError("Selection release identity is missing.")

    release_hash = canonical_json_sha256(identity)

    if selection.get("selection_release_fingerprint") != release_hash:
        raise RuntimeError("Selection release fingerprint is invalid.")

    if (
        reference.get("selection_release_id") != selection.get("selection_release_id")
        or reference.get("selection_release_fingerprint") != release_hash
    ):
        raise RuntimeError(
            "Two-Tower retrieval report refers to the wrong selection release."
        )

    if selection.get("validation_data_identity") != data_identity:
        raise RuntimeError("Selection release validation data identity mismatch.")

    manifest_experiment = validate_experiment_identity(
        selection.get("validation_experiment_identity"),
        "Two-Tower selection release",
        fingerprint,
    )

    if manifest_experiment != experiment:
        raise RuntimeError("Selection manifest and retrieval report disagree.")

    pointer = load_json(
        run_dir / "selection_releases" / "current_two_tower_selection.json"
    )

    if (
        pointer.get("selection_release_id") != selection["selection_release_id"]
        or pointer.get("selection_release_fingerprint") != release_hash
    ):
        raise RuntimeError("Reproduction Two-Tower selection pointer mismatch.")

    checkpoint_meta = identity["checkpoint"]
    checkpoint_path = manifest_path.parent / checkpoint_meta["filename"]
    checkpoint_hash = sha256_file(checkpoint_path)

    if checkpoint_hash != checkpoint_meta["sha256"]:
        raise RuntimeError("Two-Tower checkpoint SHA-256 mismatch.")

    checkpoint = load_model_checkpoint(checkpoint_path, device=device)
    config = checkpoint["model_config"]
    training = checkpoint["training_config"]

    if config["n_users"] != n_users or config["n_items"] != n_items:
        raise RuntimeError("Two-Tower checkpoint vocabulary mismatch.")

    if (
        training.get("validation_data_fingerprint") != fingerprint
        or training.get("validation_experiment_fingerprint")
        != experiment["validation_experiment_fingerprint"]
    ):
        raise RuntimeError("Two-Tower checkpoint validation lineage mismatch.")

    model = TwoTowerRecommender(
        n_users=config["n_users"],
        n_items=config["n_items"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
    ).to(device)

    model = restore_model_from_checkpoint(model, checkpoint, device)

    print(f"Two-Tower selection release: {selection['selection_release_id']}")
    print(f"Two-Tower checkpoint SHA-256: {checkpoint_hash}")

    item_meta = identity["embeddings"]["item"]
    item_path = manifest_path.parent / item_meta["filename"]
    item_hash = sha256_file(item_path)

    if item_hash != item_meta["sha256"]:
        raise RuntimeError("Two-Tower item embedding SHA-256 mismatch.")

    if item_meta["parent_checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("Item embeddings are bound to another checkpoint.")

    embeddings = load_tensor(item_path, device=device)

    shape = [n_items, config["embedding_dim"]]

    if list(embeddings.shape) != shape or item_meta["shape"] != shape:
        raise RuntimeError("Two-Tower item embedding shape mismatch.")

    if item_meta["embedding_dim"] != config["embedding_dim"]:
        raise RuntimeError("Two-Tower item embedding dimension mismatch.")

    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Two-Tower item embeddings contain non-finite values.")

    print(f"Item embeddings: {tuple(embeddings.shape)}")

    embedding_mb = embeddings.numel() * embeddings.element_size() / 1024**2
    print(f"Embedding memory: {embedding_mb:.3f} MB")
    print(f"Item embeddings SHA-256: {item_hash}")

    return model, embeddings, selection, checkpoint_hash, item_hash


def load_neumf(
    run_dir: Path,
    data_identity: dict,
    selection: dict,
    device: torch.device,
    n_users: int,
    n_items: int,
) -> tuple:
    ncf = load_json(run_dir / "ncf_results.json")

    if ncf.get("artifact_schema_version") != 2:
        raise RuntimeError("NCF report schema mismatch.")

    if ncf.get("evaluation_split") != "validation":
        raise RuntimeError("NCF report must be validation-only.")

    if ncf.get("selected_ranker") != "neumf":
        raise RuntimeError("NeuMF is not the selected neural ranker.")

    if ncf.get("validation_data_identity") != data_identity:
        raise RuntimeError("NCF validation data identity mismatch.")

    fingerprint = data_identity["validation_data_fingerprint"]

    experiment = validate_experiment_identity(
        ncf.get("validation_experiment_identity"),
        "Neural CF",
        fingerprint,
    )

    if selection.get("parent_validation_experiment_identity") != experiment:
        raise RuntimeError("Two-Tower release is bound to another NCF experiment.")

    path = run_dir / "neumf.pt"
    checkpoint_hash = sha256_file(path)
    checkpoint = load_model_checkpoint(path, device=device)

    config = checkpoint["model_config"]
    training = checkpoint["training_config"]

    if config["n_users"] != n_users or config["n_items"] != n_items:
        raise RuntimeError("NeuMF checkpoint vocabulary mismatch.")

    if (
        training.get("validation_data_fingerprint") != fingerprint
        or training.get("validation_experiment_fingerprint")
        != experiment["validation_experiment_fingerprint"]
    ):
        raise RuntimeError("NeuMF checkpoint validation lineage mismatch.")

    model = NeuMF(
        n_users=config["n_users"],
        n_items=config["n_items"],
        embedding_dim=config["embedding_dim"],
        hidden_layers=tuple(config["hidden_layers"]),
        item_score_batch_size=config.get(
            "item_score_batch_size",
            NCF_ITEM_SCORE_BATCH_SIZE,
        ),
    ).to(device)

    model = restore_model_from_checkpoint(model, checkpoint, device)

    print(f"NCF experiment: {experiment['validation_experiment_id']}")
    print(f"NeuMF checkpoint SHA-256: {checkpoint_hash}")

    return model, ncf, experiment, checkpoint_hash


def select_candidate_size(candidate_results: dict) -> int:
    """Select candidate depth using rounded validation metrics and size tie-breaks."""

    def key(size: int) -> tuple[float, float, float, float, int]:
        metrics = candidate_results[size]["metrics"]

        return (
            round(metrics["recall@50"], 3),
            round(metrics["ndcg@50"], 3),
            round(metrics["recall@10"], 3),
            round(metrics["ndcg@10"], 3),
            -size,
        )

    chosen = max(CANDIDATE_SIZES, key=key)

    if chosen != RANKING_CANDIDATE_SIZE:
        raise RuntimeError(
            "Configured RANKING_CANDIDATE_SIZE does not match the validation "
            f"quality/efficiency policy: configured={RANKING_CANDIDATE_SIZE}, "
            f"selected={chosen}."
        )

    return chosen


def report_final_test(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    selected_size: int,
    selected_metrics: dict,
    n_users: int,
    n_items: int,
) -> None:
    """Read canonical final test results without retraining or re-evaluation."""
    release = resolve_current_model_release()
    final = load_json(RANKING_RESULTS_PATH)
    manifest = load_json(ARTIFACT_MANIFEST_PATH)

    payload = manifest.get("identity")

    if not isinstance(payload, dict):
        raise TypeError("Current processed-data identity is missing.")

    fingerprint = canonical_json_sha256(payload)

    if fingerprint != manifest.get("dataset_fingerprint"):
        raise RuntimeError("Current processed dataset fingerprint is invalid.")

    if manifest.get("data_run_id") != f"processed-{fingerprint[:20]}":
        raise RuntimeError("Current processed data-run ID is invalid.")

    processed = manifest.get("processed_artifacts")

    if not isinstance(processed, dict):
        raise TypeError("Current processed artifact manifest is invalid.")

    test_meta = processed.get("test")

    if not isinstance(test_meta, dict):
        raise TypeError("Current test artifact metadata is missing.")

    test_hash = sha256_file(TEST_PATH)

    if test_meta.get("sha256") != test_hash:
        raise RuntimeError("Current test.parquet SHA-256 mismatch.")

    identity_hashes = payload.get("artifact_hashes")

    if (
        not isinstance(identity_hashes, dict)
        or identity_hashes.get("test") != test_hash
    ):
        raise RuntimeError("Current processed identity not bound to test artifact.")

    evaluation_data = final["lineage"]["evaluation_data"]

    if (
        evaluation_data.get("data_run_id") != manifest["data_run_id"]
        or evaluation_data.get("dataset_fingerprint") != manifest["dataset_fingerprint"]
        or evaluation_data.get("test_sha256") != test_hash
    ):
        raise RuntimeError("Final test evaluation data lineage mismatch.")

    if final["split"] != "test":
        raise RuntimeError("Final report must be on test split.")

    if final["evaluation_protocol"] != PREPROCESSING_PROTOCOL_VERSION:
        raise RuntimeError("Final evaluation protocol mismatch.")

    if final["training_data"] != FINAL_TRAINING_PROTOCOL:
        raise RuntimeError("Final training data protocol mismatch.")

    if final["candidate_size"] != selected_size:
        raise RuntimeError("Final candidate size differs from validation selection.")

    if final["n_users"] != n_users or final["n_items"] != n_items:
        raise RuntimeError("Final model vocabulary mismatch.")

    release_data = final["lineage"]["model_release"]

    if (
        release_data["model_release_id"] != release.model_release_id
        or release_data["model_release_fingerprint"]
        != release.model_release_fingerprint
        or release_data["manifest_sha256"] != sha256_file(release.manifest_path)
    ):
        raise RuntimeError("Final evaluation refers to wrong model release.")

    if final["lineage"]["model_lineage"] != release.lineage:
        raise RuntimeError("Final evaluation model lineage mismatch.")

    print(f"Model release ID: {release.model_release_id}")
    print(f"Model release lineage: {release.lineage['release_lineage_id']}")
    print(f"Processed data run: {manifest['data_run_id']}")
    print(f"Test SHA-256: {test_hash}")
    print("Final evaluation release, processed identity, and test SHA-256 validated.")

    history = final["final_training_history"]
    population = final["evaluation_population"]

    warm = validation.filter(pl.col("item_idx") >= 0).height
    cold = validation.filter(pl.col("item_idx") == COLD_ITEM_INDEX).height

    if history["final_training_interactions"] != train.height + warm:
        raise RuntimeError("Final training interaction count mismatch.")

    if history["warm_validation_interactions"] != warm:
        raise RuntimeError("Final training warm validation count mismatch.")

    if history["excluded_cold_validation_interactions"] != cold:
        raise RuntimeError("Final training cold validation count mismatch.")

    show(
        "Final training protocol",
        pl.DataFrame(
            [
                {
                    "metric": "Train interactions",
                    "value": train.height,
                },
                {
                    "metric": "Warm validation added",
                    "value": warm,
                },
                {
                    "metric": "Cold validation excluded",
                    "value": cold,
                },
                {
                    "metric": "Final training interactions",
                    "value": history["final_training_interactions"],
                },
            ]
        ),
    )

    show(
        "Final test population",
        pl.DataFrame(
            [
                {
                    "metric": "Total test targets",
                    "value": population["total_test_targets"],
                },
                {
                    "metric": "Warm test targets",
                    "value": population["warm_test_targets"],
                },
                {
                    "metric": "Cold test targets",
                    "value": population["cold_test_targets"],
                },
                {
                    "metric": "Cold test target pct",
                    "value": round(population["cold_test_target_pct"], 3),
                },
            ]
        ),
    )

    retrieval = final["retrieval"]
    ranked = final["retrieval_reranking"]["metrics"]

    show(
        "Final test ranking metrics",
        pl.DataFrame(
            [
                {
                    "pipeline": "Two-Tower",
                    **{key: retrieval[key] for key in METRICS},
                },
                {
                    "pipeline": "Two-Tower + NeuMF",
                    **{key: ranked[key] for key in METRICS},
                },
            ]
        ).with_columns(pl.exclude("pipeline").round(3)),
    )

    generalization = [
        {
            "metric": key,
            "validation": selected_metrics[key],
            "test": ranked[key],
            "absolute_gap": ranked[key] - selected_metrics[key],
            "relative_gap_pct": (
                (ranked[key] - selected_metrics[key]) / selected_metrics[key] * 100
            ),
        }
        for key in METRICS
    ]

    show(
        "Validation to canonical final-test generalization",
        pl.DataFrame(generalization).with_columns(pl.exclude("metric").round(3)),
    )

    improvement = [
        {
            "metric": key,
            "retrieval_only": retrieval[key],
            "reranked": ranked[key],
            "relative_improvement_pct": (
                (ranked[key] - retrieval[key]) / retrieval[key] * 100
            ),
        }
        for key in METRICS
    ]

    show(
        "Final test reranking improvement",
        pl.DataFrame(improvement).with_columns(pl.exclude("metric").round(3)),
    )

    show(
        "Canonical model release artifacts",
        pl.DataFrame(
            [
                {
                    "artifact": name,
                    "filename": meta["filename"],
                    "size_mb": meta["size_bytes"] / 1024**2,
                    "sha256": meta["sha256"],
                }
                for name, meta in release.manifest["artifacts"].items()
            ]
        ).with_columns(pl.col("size_mb").round(3)),
    )

    print("Final test results were READ only; no model release was altered.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Existing reproduction run ID",
    )
    args = parser.parse_args()

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id) is None:
        parser.error("--run-id must contain 1-64 letters, digits, - or _")

    run_dir = MODEL_DIR.parent / "reproductions" / args.run_id
    result_path = run_dir / "ranking_validation_results.json"

    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {result_path}")

    set_seed(RANDOM_SEED)
    device = get_torch_device()

    print(f"Device: {device} | Eval batch: {EVAL_BATCH_SIZE:,}")
    print(f"Candidate size: {RANKING_CANDIDATE_SIZE:,}")
    print(f"Ranking score batch: {RANKING_SCORE_BATCH_SIZE:,} | K: {EVAL_KS}")
    print(
        "Validation models: reproduction release; "
        "final test: canonical report (read-only)."
    )

    train = load_parquet(TRAIN_PATH)
    validation = load_parquet(VALIDATION_PATH)
    mapping_users = load_parquet(USER_MAPPING_PATH)
    mapping_items = load_parquet(ITEM_MAPPING_PATH)

    n_users = mapping_users.height
    n_items = mapping_items.height

    data_identity = validation_identity(
        load_json(ARTIFACT_MANIFEST_PATH),
        mapping_users,
        mapping_items,
    )

    seen = build_seen_items_csr(interactions=train, n_users=n_users)
    targets = build_target_array(holdout=validation, n_users=n_users)

    cold = validation.filter(pl.col("item_idx") == COLD_ITEM_INDEX).height
    warm = validation.height - cold

    print(f"Users: {n_users:,} | Items: {n_items:,} | Train: {train.height:,}")
    print(
        f"Validation targets: {validation.height:,} | Warm: {warm:,} | Cold: {cold:,}"
    )
    print(f"Validation data ID: {data_identity['validation_data_id']}")

    tower, embeddings, selection, tower_sha, item_sha = load_selection_release(
        run_dir, data_identity, device, n_users, n_items
    )

    neumf, ncf, ncf_experiment, neumf_sha = load_neumf(
        run_dir, data_identity, selection, device, n_users, n_items
    )

    retrieval_metrics = evaluate_embedding_retrieval(
        model=tower,
        item_embeddings=embeddings,
        target_items=targets,
        seen_items=seen,
        n_users=n_users,
        batch_size=EVAL_BATCH_SIZE,
        ks=EVAL_KS,
        device=device,
        desc="Retrieval-only validation",
    )

    show(
        "Retrieval-only validation",
        round_metric_values(retrieval_metrics, 3),
    )

    reranking = evaluate_retrieval_reranking(
        retrieval_model=tower,
        ranker=neumf,
        item_embeddings=embeddings,
        target_items=targets,
        seen_items=seen,
        n_users=n_users,
        user_batch_size=EVAL_BATCH_SIZE,
        candidate_size=RANKING_CANDIDATE_SIZE,
        ks=EVAL_KS,
        score_batch_size=RANKING_SCORE_BATCH_SIZE,
        device=device,
        desc="Two-Tower + NeuMF validation",
    )

    print(
        f"Candidate Recall@{RANKING_CANDIDATE_SIZE}: "
        f"{reranking['candidate_recall']:.3f}"
    )
    print(f"Evaluation time: {reranking['elapsed_seconds']:.3f} seconds")
    print(f"Users / second: {reranking['users_per_second']:.3f}")

    ranked_metrics = reranking["metrics"]

    show(
        "Two-Tower + NeuMF validation",
        round_metric_values(ranked_metrics, 3),
    )

    show(
        "Retrieval vs reranking",
        pl.DataFrame(
            [
                {
                    "pipeline": "Two-Tower",
                    **{key: retrieval_metrics[key] for key in METRICS},
                },
                {
                    "pipeline": "Two-Tower + NeuMF",
                    **{key: ranked_metrics[key] for key in METRICS},
                },
            ]
        ).with_columns(pl.exclude("pipeline").round(3)),
    )

    show(
        "Reranking improvement",
        pl.DataFrame(
            [
                {
                    "metric": key,
                    "retrieval_only": retrieval_metrics[key],
                    "reranked": ranked_metrics[key],
                    "relative_improvement_pct": (
                        (ranked_metrics[key] - retrieval_metrics[key])
                        / retrieval_metrics[key]
                        * 100
                    ),
                }
                for key in METRICS
            ]
        ).with_columns(pl.exclude("metric").round(3)),
    )

    full_neumf = ncf["validation"]["neumf"]

    print(
        f"Full catalog NeuMF provenance: {ncf_experiment['validation_experiment_id']}"
    )

    show(
        "Full catalog vs two stage",
        pl.DataFrame(
            [
                {
                    "pipeline": "Full-Catalog NeuMF",
                    "items_scored_per_user": n_items,
                    **{key: full_neumf[key] for key in QUALITY_METRICS},
                },
                {
                    "pipeline": "Two-Tower + NeuMF",
                    "items_scored_per_user": RANKING_CANDIDATE_SIZE,
                    **{key: ranked_metrics[key] for key in QUALITY_METRICS},
                },
            ]
        ).with_columns(pl.exclude("pipeline", "items_scored_per_user").round(3)),
    )

    reduction_pct = (1 - RANKING_CANDIDATE_SIZE / n_items) * 100

    print(f"Neural score reduction: {reduction_pct:.3f}%")

    candidate_results = evaluate_reranking_candidate_sizes(
        retrieval_model=tower,
        ranker=neumf,
        item_embeddings=embeddings,
        target_items=targets,
        seen_items=seen,
        n_users=n_users,
        user_batch_size=EVAL_BATCH_SIZE,
        candidate_sizes=CANDIDATE_SIZES,
        ks=EVAL_KS,
        score_batch_size=RANKING_SCORE_BATCH_SIZE,
        device=device,
        desc="Candidate-size trade-off",
    )

    show(
        "Candidate size comparison",
        pl.DataFrame(
            [
                {
                    "candidate_size": size,
                    "candidate_recall": candidate_results[size]["candidate_recall"],
                    **{
                        key: candidate_results[size]["metrics"][key]
                        for key in QUALITY_METRICS
                    },
                    "catalog_retained_pct": (size / n_items * 100),
                    "neural_score_reduction_pct": ((1 - size / n_items) * 100),
                }
                for size in CANDIDATE_SIZES
            ]
        ).with_columns(pl.exclude("candidate_size").round(3)),
    )

    benchmark = candidate_results["_benchmark"]

    print(f"Candidate-size evaluation time: {benchmark['elapsed_seconds']:.3f} seconds")
    print(f"Candidate-size users/second: {benchmark['users_per_second']:.3f}")

    selected_size = select_candidate_size(candidate_results)
    selected_result = candidate_results[selected_size]
    selected_metrics = selected_result["metrics"]

    rule = {
        "rule_version": 2,
        "selection_scope": "validation_only",
        "candidate_sizes": list(CANDIDATE_SIZES),
        "policy": "smallest_candidate_with_near_best_quality",
        "metric_decimals": 3,
        "primary_metrics": [
            "recall@50",
            "ndcg@50",
            "recall@10",
            "ndcg@10",
        ],
        "tie_breaker": "smaller_candidate_size",
    }

    selection_config = {
        "experiment_family": "two_stage_ranking",
        "evaluation_split": "validation",
        "retrieval_model": "TwoTowerRecommender",
        "ranking_model": "NeuMF",
        "candidate_sizes": list(CANDIDATE_SIZES),
        "ks": list(EVAL_KS),
        "eval_batch_size": EVAL_BATCH_SIZE,
        "ranking_score_batch_size": RANKING_SCORE_BATCH_SIZE,
        "candidate_selection_rule": rule,
    }

    selection_payload = {
        "identity_schema_version": 1,
        "validation_data_fingerprint": data_identity["validation_data_fingerprint"],
        "two_tower_selection_release_fingerprint": selection[
            "selection_release_fingerprint"
        ],
        "two_tower_validation_experiment_fingerprint": selection[
            "validation_experiment_identity"
        ]["validation_experiment_fingerprint"],
        "ncf_validation_experiment_fingerprint": ncf_experiment[
            "validation_experiment_fingerprint"
        ],
        "two_tower_checkpoint_sha256": tower_sha,
        "two_tower_item_embeddings_sha256": item_sha,
        "neumf_checkpoint_sha256": neumf_sha,
        "selection_config": selection_config,
    }

    fingerprint = canonical_json_sha256(selection_payload)

    experiment_identity = {
        "validation_experiment_id": (f"validation-ranking-{fingerprint[:20]}"),
        "validation_experiment_fingerprint": fingerprint,
        "identity": selection_payload,
    }

    report = {
        "artifact_schema_version": 1,
        "evaluation_split": "validation",
        "validation_data_identity": data_identity,
        "validation_experiment_identity": experiment_identity,
        "two_tower_selection_release": {
            "selection_release_id": selection["selection_release_id"],
            "selection_release_fingerprint": selection["selection_release_fingerprint"],
        },
        "ncf_validation_experiment_identity": ncf_experiment,
        "candidate_selection_rule": rule,
        "candidate_results": {
            str(size): {
                "candidate_recall": candidate_results[size]["candidate_recall"],
                "metrics": candidate_results[size]["metrics"],
            }
            for size in CANDIDATE_SIZES
        },
        "selected_candidate_size": selected_size,
    }

    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {result_path}")

    save_json(report, result_path)

    show(
        "Ranking funnel",
        pl.DataFrame(
            [
                {
                    "stage": f"Retrieved Top-{selected_size}",
                    "positive_recall": selected_result["candidate_recall"],
                },
                {
                    "stage": "Reranked Top-50",
                    "positive_recall": selected_metrics["recall@50"],
                },
                {
                    "stage": "Reranked Top-10",
                    "positive_recall": selected_metrics["recall@10"],
                },
            ]
        ).with_columns(pl.col("positive_recall").round(3)),
    )

    print(f"Validation-selected candidate size: {selected_size}")
    print(
        "Ranking validation experiment: "
        f"{experiment_identity['validation_experiment_id']}"
    )
    print(f"Saved selection artifact: {result_path}")

    candidate_recall = selected_result["candidate_recall"]

    print(
        "Retrieved positives promoted to Top-50: "
        f"{selected_metrics['recall@50'] / candidate_recall * 100:.3f}%"
    )
    print(
        "Retrieved positives promoted to Top-10: "
        f"{selected_metrics['recall@10'] / candidate_recall * 100:.3f}%"
    )

    report_final_test(
        train,
        validation,
        selected_size,
        selected_metrics,
        n_users,
        n_items,
    )


if __name__ == "__main__":
    main()
