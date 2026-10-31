"""Train and evaluate GMF, MLP, and pretrained NeuMF on validation.

Run from the project root:
    uv run python scripts/03_neural_cf.py --run-id neural-001

Requires the baseline report from script 02 with the same run ID.
Existing run checkpoints and reports are never overwritten.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import polars as pl
import torch

from recsys.config import (
    ARTIFACT_MANIFEST_PATH,
    COLD_ITEM_INDEX,
    EVAL_BATCH_SIZE,
    EVAL_KS,
    ITEM_MAPPING_PATH,
    MODEL_DIR,
    NCF_EMBEDDING_DIM,
    NCF_EPOCHS,
    NCF_ITEM_SCORE_BATCH_SIZE,
    NCF_LEARNING_RATE,
    NCF_MLP_LAYERS,
    NCF_WEIGHT_DECAY,
    NEUMF_EPOCHS,
    NEUMF_LEARNING_RATE,
    NEUMF_PRETRAIN_ALPHA,
    NEUMF_WEIGHT_DECAY,
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
    benchmark_full_catalog_scoring,
    build_seen_items_csr,
    build_target_array,
    evaluate_model,
)
from recsys.evaluation.metrics import round_metric_values
from recsys.models.ncf import (
    GMF,
    MLPRecommender,
    NeuMF,
    initialize_neumf_from_pretrained,
)
from recsys.training import train_pointwise_epoch
from recsys.utils.artifacts import (
    canonical_id_mapping_sha256,
    canonical_json_sha256,
    sha256_file,
)
from recsys.utils.helpers import get_torch_device, set_seed
from recsys.utils.model_io import save_model_checkpoint


def show(title: str, table: object) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")
    print(table)


def validate_parent(
    manifest: dict,
    previous_results: dict,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
) -> tuple[dict, dict, str]:
    """Validate the parent report and data lineage without using test data."""
    if previous_results.get("artifact_schema_version") != 2:
        raise RuntimeError("Baseline report schema must be version 2.")
    if previous_results.get("evaluation_split") != "validation":
        raise RuntimeError("Parent results must use the validation split.")
    if manifest.get("preprocessing_protocol_version") != PREPROCESSING_PROTOCOL_VERSION:
        raise RuntimeError("Preprocessing protocol differs from current configuration.")

    processed_artifacts = manifest.get("processed_artifacts")
    if not isinstance(processed_artifacts, dict):
        raise TypeError("Processed artifact metadata must be a dictionary.")

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
            raise TypeError(f"Missing artifact metadata for {name}.")
        expected = metadata.get("sha256")
        if not isinstance(expected, str):
            raise TypeError(f"Missing SHA-256 for {name}.")
        actual = sha256_file(path)
        if expected != actual:
            raise RuntimeError(f"Artifact hash mismatch: {name}.")
        artifact_hashes[name] = actual

    semantic_hashes = manifest.get("semantic_mapping_hashes")
    if not isinstance(semantic_hashes, dict):
        raise TypeError("Semantic mapping hashes must be a dictionary.")
    user_hash = canonical_id_mapping_sha256(user_mapping, "userId", "user_idx")
    item_hash = canonical_id_mapping_sha256(item_mapping, "movieId", "item_idx")
    if user_hash != semantic_hashes.get("user_mapping_sha256"):
        raise RuntimeError("User mapping fingerprint mismatch.")
    if item_hash != semantic_hashes.get("item_mapping_sha256"):
        raise RuntimeError("Item mapping fingerprint mismatch.")

    payload = {
        "identity_schema_version": 1,
        "preprocessing_protocol_version": PREPROCESSING_PROTOCOL_VERSION,
        "artifact_hashes": artifact_hashes,
        "semantic_mapping_hashes": {
            "user_mapping_sha256": user_hash,
            "item_mapping_sha256": item_hash,
        },
    }
    fingerprint = canonical_json_sha256(payload)
    data_identity = {
        "validation_data_id": f"validation-data-{fingerprint[:20]}",
        "validation_data_fingerprint": fingerprint,
        "identity": payload,
    }
    if previous_results.get("validation_data_identity") != data_identity:
        raise RuntimeError("Parent data identity does not match current artifacts.")

    parent_identity = previous_results.get("validation_experiment_identity")
    if not isinstance(parent_identity, dict):
        raise TypeError("Parent experiment identity is missing.")
    parent_payload = parent_identity.get("identity")
    if not isinstance(parent_payload, dict):
        raise TypeError("Parent experiment identity payload is missing.")
    parent_fingerprint = canonical_json_sha256(parent_payload)
    if parent_fingerprint != parent_identity.get("validation_experiment_fingerprint"):
        raise RuntimeError("Parent experiment fingerprint is invalid.")
    if parent_payload.get("validation_data_fingerprint") != fingerprint:
        raise RuntimeError("Parent experiment is bound to different validation data.")
    if "test" in artifact_hashes:
        raise RuntimeError("Test artifact must not enter validation identity.")
    return data_identity, parent_identity, fingerprint


def neural_identity(validation_fingerprint: str, parent: dict) -> dict:
    """Build the neural validation selection configuration and fingerprint."""
    selection_config = {
        "experiment_family": "neural_collaborative_filtering",
        "evaluation_split": "validation",
        "evaluation_ks": list(EVAL_KS),
        "evaluation_batch_size": EVAL_BATCH_SIZE,
        "training_batch_size": TRAIN_BATCH_SIZE,
        "item_score_batch_size": NCF_ITEM_SCORE_BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "cold_item_index": COLD_ITEM_INDEX,
        "gmf": {
            "embedding_dim": NCF_EMBEDDING_DIM,
            "epochs": NCF_EPOCHS,
            "learning_rate": NCF_LEARNING_RATE,
            "weight_decay": NCF_WEIGHT_DECAY,
            "negative_sampling": "uniform",
            "negatives_per_positive": 1,
        },
        "mlp": {
            "embedding_dim": NCF_EMBEDDING_DIM,
            "hidden_layers": list(NCF_MLP_LAYERS),
            "epochs": NCF_EPOCHS,
            "learning_rate": NCF_LEARNING_RATE,
            "weight_decay": NCF_WEIGHT_DECAY,
            "negative_sampling": "uniform",
            "negatives_per_positive": 1,
        },
        "neumf": {
            "embedding_dim": NCF_EMBEDDING_DIM,
            "hidden_layers": list(NCF_MLP_LAYERS),
            "epochs": NEUMF_EPOCHS,
            "learning_rate": NEUMF_LEARNING_RATE,
            "weight_decay": NEUMF_WEIGHT_DECAY,
            "pretrain_alpha": NEUMF_PRETRAIN_ALPHA,
            "initialization": "pretrained_gmf_plus_mlp",
            "negative_sampling": "uniform",
            "negatives_per_positive": 1,
        },
        "model_selection": {
            "candidate_models": ["GMF", "MLP", "NeuMF"],
            "primary_metric": "ndcg@10",
            "sort_order": "descending",
        },
    }
    payload = {
        "identity_schema_version": 1,
        "validation_data_fingerprint": validation_fingerprint,
        "parent_validation_experiment_fingerprint": (
            parent["validation_experiment_fingerprint"]
        ),
        "selection_config": selection_config,
    }
    fingerprint = canonical_json_sha256(payload)
    return {
        "validation_experiment_id": f"validation-ncf-{fingerprint[:20]}",
        "validation_experiment_fingerprint": fingerprint,
        "identity": payload,
    }


def train_epochs(
    label: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: UniformNegativeSampler,
    users: np.ndarray,
    items: np.ndarray,
    epochs: int,
    device: torch.device,
) -> list[dict]:
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_pointwise_epoch(
            model=model,
            optimizer=optimizer,
            user_ids=users,
            item_ids=items,
            negative_sampler=sampler,
            batch_size=TRAIN_BATCH_SIZE,
            device=device,
            seed=RANDOM_SEED + epoch,
            desc=f"{label} epoch {epoch}",
        )
        history.append({"epoch": epoch, "train_loss": loss})
        print(f"{label} epoch {epoch:02d} | train_loss={loss:.3f}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Same run ID as script 02")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id) is None:
        parser.error("--run-id must contain 1-64 letters, digits, - or _")

    output_dir = MODEL_DIR.parent / "reproductions" / args.run_id
    baseline_path = output_dir / "baseline_mf_results.json"
    checkpoint_path = output_dir / "neumf.pt"
    report_path = output_dir / "ncf_results.json"
    history_path = output_dir / "ncf_training_history.json"
    for output in (checkpoint_path, report_path, history_path):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite reproduction: {output}")
    parent_results = load_json(baseline_path)

    set_seed(RANDOM_SEED)
    device = get_torch_device()
    print(f"Device: {device} | Users batch: {EVAL_BATCH_SIZE}")
    print(
        f"Training batch: {TRAIN_BATCH_SIZE} | Item chunk: {NCF_ITEM_SCORE_BATCH_SIZE}"
    )
    print(f"NCF epochs: {NCF_EPOCHS} | NeuMF epochs: {NEUMF_EPOCHS}")
    print("Training on train; selecting on validation; test split is untouched.")

    train = load_parquet(TRAIN_PATH)
    validation = load_parquet(VALIDATION_PATH)
    user_mapping = load_parquet(USER_MAPPING_PATH)
    item_mapping = load_parquet(ITEM_MAPPING_PATH)
    n_users, n_items = user_mapping.height, item_mapping.height
    manifest = load_json(ARTIFACT_MANIFEST_PATH)
    data_id, parent_id, data_fingerprint = validate_parent(
        manifest,
        parent_results,
        user_mapping,
        item_mapping,
    )
    parent_validation = parent_results["metrics"]
    print(f"Validation data ID: {data_id['validation_data_id']}")
    print(f"Parent experiment ID: {parent_id['validation_experiment_id']}")

    users = train["user_idx"].to_numpy().astype(np.int64, copy=False)
    items = train["item_idx"].to_numpy().astype(np.int64, copy=False)
    seen = build_seen_items_csr(interactions=train, n_users=n_users)
    targets = build_target_array(holdout=validation, n_users=n_users)
    cold = int(np.sum(targets == COLD_ITEM_INDEX))
    warm = targets.size - cold
    print(f"Training rows: {train.height:,} | Users: {n_users:,} | Items: {n_items:,}")
    print(f"Validation targets: {targets.size:,} | Warm: {warm:,} | Cold: {cold:,}")

    gmf_sampler = UniformNegativeSampler(users, items, n_items, seed=RANDOM_SEED)
    gmf = GMF(n_users, n_items, embedding_dim=NCF_EMBEDDING_DIM).to(device)
    gmf_optimizer = torch.optim.AdamW(
        gmf.parameters(),
        lr=NCF_LEARNING_RATE,
        weight_decay=NCF_WEIGHT_DECAY,
    )
    print(f"GMF parameters: {sum(p.numel() for p in gmf.parameters()):,}")
    gmf_history = train_epochs(
        "GMF",
        gmf,
        gmf_optimizer,
        gmf_sampler,
        users,
        items,
        NCF_EPOCHS,
        device,
    )
    gmf_metrics = evaluate_model(
        model=gmf,
        target_items=targets,
        seen_items=seen,
        n_users=n_users,
        batch_size=EVAL_BATCH_SIZE,
        ks=EVAL_KS,
        device=device,
        desc="GMF validation",
    )
    show("GMF metrics", round_metric_values(gmf_metrics, decimals=3))

    set_seed(RANDOM_SEED)
    mlp_sampler = UniformNegativeSampler(users, items, n_items, seed=RANDOM_SEED)
    mlp = MLPRecommender(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=NCF_EMBEDDING_DIM,
        hidden_layers=NCF_MLP_LAYERS,
        item_score_batch_size=NCF_ITEM_SCORE_BATCH_SIZE,
    ).to(device)
    mlp_optimizer = torch.optim.AdamW(
        mlp.parameters(),
        lr=NCF_LEARNING_RATE,
        weight_decay=NCF_WEIGHT_DECAY,
    )
    print(f"MLP parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    mlp_history = train_epochs(
        "MLP",
        mlp,
        mlp_optimizer,
        mlp_sampler,
        users,
        items,
        NCF_EPOCHS,
        device,
    )

    # Benchmark MLP, then GMF, before evaluating MLP recommendations.
    mlp_benchmark = benchmark_full_catalog_scoring(
        model=mlp,
        batch_size=EVAL_BATCH_SIZE,
        n_items=n_items,
        device=device,
    )
    gmf_benchmark = benchmark_full_catalog_scoring(
        model=gmf,
        batch_size=EVAL_BATCH_SIZE,
        n_items=n_items,
        device=device,
    )
    show(
        "GMF vs MLP throughput",
        pl.DataFrame(
            [
                {"model": "GMF", **gmf_benchmark},
                {"model": "MLP", **mlp_benchmark},
            ]
        ).with_columns(pl.exclude("model", "batch_size", "items").round(3)),
    )
    print(
        "Estimated MLP full validation minutes: "
        f"{n_users / mlp_benchmark['users_per_second'] / 60:.3f}"
    )
    mlp_metrics = evaluate_model(
        model=mlp,
        target_items=targets,
        seen_items=seen,
        n_users=n_users,
        batch_size=EVAL_BATCH_SIZE,
        ks=EVAL_KS,
        device=device,
        desc="MLP validation",
    )
    show("MLP metrics", round_metric_values(mlp_metrics, decimals=3))

    set_seed(RANDOM_SEED)
    neumf = NeuMF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=NCF_EMBEDDING_DIM,
        hidden_layers=NCF_MLP_LAYERS,
        item_score_batch_size=NCF_ITEM_SCORE_BATCH_SIZE,
    ).to(device)
    initialize_neumf_from_pretrained(
        neumf=neumf,
        gmf=gmf,
        mlp=mlp,
        alpha=NEUMF_PRETRAIN_ALPHA,
    )
    neumf_optimizer = torch.optim.AdamW(
        neumf.parameters(),
        lr=NEUMF_LEARNING_RATE,
        weight_decay=NEUMF_WEIGHT_DECAY,
    )
    print(f"NeuMF parameters: {sum(p.numel() for p in neumf.parameters()):,}")
    neumf_sampler = UniformNegativeSampler(users, items, n_items, seed=RANDOM_SEED)
    neumf_history = train_epochs(
        "NeuMF",
        neumf,
        neumf_optimizer,
        neumf_sampler,
        users,
        items,
        NEUMF_EPOCHS,
        device,
    )
    neumf_benchmark = benchmark_full_catalog_scoring(
        model=neumf,
        batch_size=EVAL_BATCH_SIZE,
        n_items=n_items,
        device=device,
    )
    print(
        "Estimated NeuMF full validation minutes: "
        f"{n_users / neumf_benchmark['users_per_second'] / 60:.3f}"
    )
    neumf_metrics = evaluate_model(
        model=neumf,
        target_items=targets,
        seen_items=seen,
        n_users=n_users,
        batch_size=EVAL_BATCH_SIZE,
        ks=EVAL_KS,
        device=device,
        desc="NeuMF validation",
    )
    show("NeuMF metrics", round_metric_values(neumf_metrics, decimals=3))

    # Round comparison metrics to three decimals before model selection.
    # Select using rounded NDCG@10, not the raw evaluation precision.
    comparison = pl.DataFrame(
        [
            {"model": "Popularity", **parent_validation["popularity"]},
            {"model": "Pointwise MF", **parent_validation["pointwise_mf"]},
            {"model": "BPR MF", **parent_validation["bpr_mf"]},
            {"model": "GMF", **gmf_metrics},
            {"model": "MLP", **mlp_metrics},
            {"model": "NeuMF", **neumf_metrics},
        ]
    ).with_columns(pl.exclude("model").round(3))
    show("All validation models", comparison)
    selection = (
        comparison.filter(pl.col("model").is_in(["GMF", "MLP", "NeuMF"]))
        .select("model", "recall@10", "mrr@10", "ndcg@10", "recall@50", "ndcg@50")
        .sort("ndcg@10", descending=True)
    )
    selected = selection.row(0, named=True)["model"]
    show("Neural ranker selection", selection)
    if selected != "NeuMF":
        raise RuntimeError("NeuMF did not win validation; downstream assumes NeuMF.")

    benchmarks = {
        "gmf": gmf_benchmark,
        "mlp": mlp_benchmark,
        "neumf": neumf_benchmark,
    }
    show(
        "Full-catalog throughput",
        pl.DataFrame(
            [
                {"model": name.upper() if name != "neumf" else "NeuMF", **values}
                for name, values in benchmarks.items()
            ]
        ).with_columns(pl.exclude("model", "batch_size", "items").round(3)),
    )
    quality_metrics = ("recall@10", "ndcg@10", "recall@50", "ndcg@50")
    improvement = []
    for name, reference in (
        ("BPR MF", parent_validation["bpr_mf"]),
        ("GMF", gmf_metrics),
        ("MLP", mlp_metrics),
    ):
        improvement.append(
            {
                "comparison": f"NeuMF vs {name}",
                **{
                    f"{metric}_improvement_pct": (
                        (neumf_metrics[metric] - reference[metric])
                        / reference[metric]
                        * 100
                    )
                    for metric in quality_metrics
                },
            }
        )
    show(
        "NeuMF improvement",
        pl.DataFrame(improvement).with_columns(
            pl.exclude("comparison").round(3),
        ),
    )
    gmf_throughput = gmf_benchmark["million_scores_per_second"]
    show(
        "Inference cost tradeoff",
        pl.DataFrame(
            [
                {
                    "model": name,
                    "million_scores_per_second": values["million_scores_per_second"],
                    "relative_speed_vs_gmf": (
                        values["million_scores_per_second"] / gmf_throughput
                    ),
                    "slowdown_vs_gmf": (
                        gmf_throughput / values["million_scores_per_second"]
                    ),
                }
                for name, values in (
                    ("GMF", gmf_benchmark),
                    ("MLP", mlp_benchmark),
                    ("NeuMF", neumf_benchmark),
                )
            ]
        ).with_columns(pl.exclude("model").round(3)),
    )

    scores_per_second = neumf_benchmark["million_scores_per_second"] * 1_000_000
    projection = pl.DataFrame(
        [
            {
                "candidate_size": size,
                "total_scores_millions": n_users * size / 1_000_000,
                "estimated_seconds": n_users * size / scores_per_second,
                "estimated_minutes": n_users * size / scores_per_second / 60,
            }
            for size in (100, 250, 500, 1000, n_items)
        ]
    ).with_columns(pl.exclude("candidate_size").round(3))
    show("Candidate pool cost projection", projection)

    selection_identity = neural_identity(data_fingerprint, parent_id)
    save_model_checkpoint(
        model=neumf,
        path=checkpoint_path,
        model_config={
            "model": "NeuMF",
            "n_users": n_users,
            "n_items": n_items,
            "embedding_dim": NCF_EMBEDDING_DIM,
            "hidden_layers": NCF_MLP_LAYERS,
            "item_score_batch_size": NCF_ITEM_SCORE_BATCH_SIZE,
        },
        metrics=neumf_metrics,
        training_config={
            "training_data": "train",
            "evaluation_split": "validation",
            "objective": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "initialization": "pretrained_gmf_plus_mlp",
            "pretrain_alpha": NEUMF_PRETRAIN_ALPHA,
            "negative_sampling": "uniform",
            "negatives_per_positive": 1,
            "learning_rate": NEUMF_LEARNING_RATE,
            "weight_decay": NEUMF_WEIGHT_DECAY,
            "epochs": NEUMF_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "seed": RANDOM_SEED,
            "validation_data_id": data_id["validation_data_id"],
            "validation_data_fingerprint": data_fingerprint,
            "validation_experiment_id": selection_identity["validation_experiment_id"],
            "validation_experiment_fingerprint": selection_identity[
                "validation_experiment_fingerprint"
            ],
            "parent_validation_experiment_id": parent_id["validation_experiment_id"],
        },
    )
    report = {
        "artifact_schema_version": 2,
        "evaluation_split": "validation",
        "validation_data_identity": data_id,
        "parent_validation_experiment_identity": parent_id,
        "validation_experiment_identity": selection_identity,
        "validation_population": {
            "users": n_users,
            "items": n_items,
            "warm_targets": warm,
            "cold_targets": cold,
            "cold_item_index": COLD_ITEM_INDEX,
        },
        "validation": {
            "gmf": gmf_metrics,
            "mlp": mlp_metrics,
            "neumf": neumf_metrics,
        },
        "inference": benchmarks,
        "candidate_cost_projection": projection.to_dicts(),
        "selected_ranker": "neumf",
    }
    save_json(report, report_path)
    save_json(
        {
            "gmf": gmf_history,
            "mlp": mlp_history,
            "neumf": neumf_history,
        },
        history_path,
    )
    print(f"Selected neural ranker: {selected}")
    print(f"Validation experiment: {selection_identity['validation_experiment_id']}")
    print(f"Saved selected NeuMF: {checkpoint_path}")
    print(f"Saved neural CF results: {report_path}")
    print(f"Saved training histories: {history_path}")


if __name__ == "__main__":
    main()
