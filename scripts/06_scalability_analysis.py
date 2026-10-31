"""Analyze canonical recommendation-system scalability and serving tradeoffs.

Run from the repository root with the run ID used by scripts 02-05:
    uv run python scripts/06_scalability_analysis.py --run-id experiment-001

Inputs: processed data and lineage, the immutable model release, the canonical
final-test report, and the run's validation ranking report.
Writes only artifacts/reproductions/<run-id>/scalability_results.json.
Canonical data, checkpoints, releases, and final-test results remain read-only.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import polars as pl

from recsys.config import (
    ARTIFACT_MANIFEST_PATH,
    EVAL_BATCH_SIZE,
    FINAL_TRAINING_PROTOCOL,
    ITEM_MAPPING_PATH,
    MODEL_DIR,
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
from recsys.data.loader import (
    load_json,
    load_parquet,
    save_json,
)
from recsys.evaluation.evaluator import (
    benchmark_candidate_ranking,
    benchmark_embedding_retrieval,
    build_seen_items_csr,
    generate_recommendations,
)
from recsys.evaluation.system_metrics import (
    artifact_size_mb,
    count_model_parameters,
    embedding_scaling_table,
    popularity_concentration,
    recommendation_catalog_metrics,
    recommendation_popularity_metrics,
    tensor_memory_mb,
)
from recsys.models.ncf import NeuMF
from recsys.models.two_tower import TwoTowerRecommender
from recsys.utils.artifacts import (
    canonical_json_sha256,
    sha256_file,
)
from recsys.utils.helpers import get_torch_device
from recsys.utils.model_io import (
    load_model_checkpoint,
    load_tensor,
    restore_model_from_checkpoint,
)
from recsys.utils.releases import resolve_current_model_release


def show(title: str, table: object) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Existing reproduction run ID")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id) is None:
        parser.error("--run-id must contain 1-64 letters, digits, - or _")

    run_dir = MODEL_DIR.parent / "reproductions" / args.run_id
    scalability_output = run_dir / "scalability_results.json"
    if scalability_output.exists():
        raise FileExistsError(f"Refusing to overwrite: {scalability_output}")
    if not (run_dir / "ranking_validation_results.json").is_file():
        raise FileNotFoundError(
            "Run scripts/05_ranking_and_evaluation.py for this run ID first."
        )
    print("Canonical production model/test artifacts: READ ONLY")
    print(f"Reproduction output: {scalability_output}")

    # Resolve the compute device and load canonical inputs.
    device = get_torch_device()

    train = load_parquet(TRAIN_PATH)
    validation = load_parquet(VALIDATION_PATH)

    user_mapping = load_parquet(USER_MAPPING_PATH)
    item_mapping = load_parquet(ITEM_MAPPING_PATH)

    final_results = load_json(RANKING_RESULTS_PATH)
    processed_manifest = load_json(ARTIFACT_MANIFEST_PATH)
    model_release = resolve_current_model_release()

    N_USERS = user_mapping.height
    N_ITEMS = item_mapping.height

    # ---------------------------------------------------------
    # Validate the current processed-data identity
    # ---------------------------------------------------------

    processed_identity_payload = processed_manifest.get("identity")

    if not isinstance(processed_identity_payload, dict):
        raise TypeError("Current processed-data identity payload is missing.")

    current_processed_fingerprint = canonical_json_sha256(processed_identity_payload)

    if processed_manifest.get("dataset_fingerprint") != current_processed_fingerprint:
        raise RuntimeError(
            "Current processed dataset fingerprint does not match its identity payload."
        )

    expected_processed_run_id = f"processed-{current_processed_fingerprint[:20]}"

    if processed_manifest.get("data_run_id") != expected_processed_run_id:
        raise RuntimeError(
            "Current processed data_run_id does not match its dataset fingerprint."
        )

    if (
        processed_manifest.get("preprocessing_protocol_version")
        != PREPROCESSING_PROTOCOL_VERSION
    ):
        raise RuntimeError(
            "Processed-data preprocessing protocol does not match "
            "the current project configuration."
        )

    # ---------------------------------------------------------
    # Validate the exact current test artifact
    # ---------------------------------------------------------

    processed_artifacts = processed_manifest.get("processed_artifacts")

    if not isinstance(processed_artifacts, dict):
        raise TypeError("Current processed artifact manifest is invalid.")

    test_metadata = processed_artifacts.get("test")

    if not isinstance(test_metadata, dict):
        raise TypeError("Current test artifact metadata is missing.")

    current_test_sha256 = sha256_file(TEST_PATH)

    if test_metadata.get("sha256") != current_test_sha256:
        raise RuntimeError(
            "Current test.parquet SHA-256 does not match "
            "the processed artifact manifest."
        )

    identity_artifact_hashes = processed_identity_payload.get("artifact_hashes")

    if not isinstance(identity_artifact_hashes, dict):
        raise TypeError("Processed identity artifact hashes are missing.")

    if identity_artifact_hashes.get("test") != current_test_sha256:
        raise RuntimeError(
            "Current processed-data identity is not bound to the current test artifact."
        )

    # ---------------------------------------------------------
    # Validate final evaluation provenance
    # ---------------------------------------------------------

    assert final_results["split"] == "test"

    assert final_results["evaluation_protocol"] == PREPROCESSING_PROTOCOL_VERSION

    assert final_results["training_data"] == FINAL_TRAINING_PROTOCOL

    assert final_results["candidate_size"] == RANKING_CANDIDATE_SIZE

    assert final_results["n_users"] == N_USERS
    assert final_results["n_items"] == N_ITEMS

    evaluation_data = final_results["lineage"]["evaluation_data"]

    if evaluation_data.get("data_run_id") != processed_manifest["data_run_id"]:
        raise RuntimeError(
            "Final evaluation was produced from a different processed data run."
        )

    if (
        evaluation_data.get("dataset_fingerprint")
        != processed_manifest["dataset_fingerprint"]
    ):
        raise RuntimeError(
            "Final evaluation was produced from a different "
            "processed dataset fingerprint."
        )

    if evaluation_data.get("test_sha256") != current_test_sha256:
        raise RuntimeError(
            "Final evaluation metrics are stale for the current test artifact."
        )

    # ---------------------------------------------------------
    # Validate immutable model-release provenance
    # ---------------------------------------------------------

    result_model_release = final_results["lineage"]["model_release"]

    if result_model_release["model_release_id"] != model_release.model_release_id:
        raise RuntimeError("Final evaluation references a different model release.")

    if (
        result_model_release["model_release_fingerprint"]
        != model_release.model_release_fingerprint
    ):
        raise RuntimeError("Final evaluation model-release fingerprint mismatch.")

    current_model_manifest_sha256 = sha256_file(model_release.manifest_path)

    if result_model_release["manifest_sha256"] != current_model_manifest_sha256:
        raise RuntimeError("Final evaluation model-release manifest SHA-256 mismatch.")

    if final_results["lineage"]["model_lineage"] != model_release.lineage:
        raise RuntimeError(
            "Final evaluation model lineage does not match "
            "the current immutable model release."
        )

    # ---------------------------------------------------------
    # Validate final training-history accounting
    # ---------------------------------------------------------

    warm_validation = validation.filter(pl.col("item_idx") >= 0)

    validation_warm_targets = warm_validation.height
    validation_cold_targets = validation.height - validation_warm_targets

    assert (
        final_results["final_training_history"]["warm_validation_interactions"]
        == validation_warm_targets
    )

    assert (
        final_results["final_training_history"]["excluded_cold_validation_interactions"]
        == validation_cold_targets
    )

    final_retrieval_metrics = final_results["retrieval"]

    final_pipeline_results = final_results["retrieval_reranking"]

    final_pipeline_metrics = final_pipeline_results["metrics"]

    print(f"Device                  : {device}")
    print(f"Users                   : {N_USERS:,}")
    print(f"Movies                  : {N_ITEMS:,}")
    print(f"Candidate size          : {RANKING_CANDIDATE_SIZE:,}")
    print(f"Warm validation added   : {validation_warm_targets:,}")
    print(f"Cold validation excluded: {validation_cold_targets:,}")
    print(f"Model release           : {model_release.model_release_id}")
    print(f"Processed data run      : {processed_manifest['data_run_id']}")
    print(f"Current test SHA-256    : {current_test_sha256}")
    print("Final evaluation provenance validated.")

    # Load the immutable Two-Tower checkpoint.
    two_tower_checkpoint = load_model_checkpoint(
        model_release.two_tower_model_path,
        device=device,
    )

    two_tower_config = two_tower_checkpoint["model_config"]

    final_two_tower = TwoTowerRecommender(
        n_users=two_tower_config["n_users"],
        n_items=two_tower_config["n_items"],
        embedding_dim=two_tower_config["embedding_dim"],
        hidden_dim=two_tower_config["hidden_dim"],
    ).to(device)

    final_two_tower = restore_model_from_checkpoint(
        model=final_two_tower,
        checkpoint=two_tower_checkpoint,
        device=device,
    )

    # Load the immutable NeuMF checkpoint.
    neumf_checkpoint = load_model_checkpoint(
        model_release.neumf_model_path,
        device=device,
    )

    neumf_config = neumf_checkpoint["model_config"]

    final_neumf = NeuMF(
        n_users=neumf_config["n_users"],
        n_items=neumf_config["n_items"],
        embedding_dim=neumf_config["embedding_dim"],
        hidden_layers=tuple(neumf_config["hidden_layers"]),
        item_score_batch_size=(neumf_config["item_score_batch_size"]),
    ).to(device)

    final_neumf = restore_model_from_checkpoint(
        model=final_neumf,
        checkpoint=neumf_checkpoint,
        device=device,
    )

    # Load production user and item embeddings.
    user_embeddings = load_tensor(
        model_release.user_embeddings_path,
        device="cpu",
    )

    item_embeddings = load_tensor(
        model_release.item_embeddings_path,
        device=device,
    )

    assert user_embeddings.shape[0] == N_USERS
    assert item_embeddings.shape[0] == N_ITEMS

    print(f"User embeddings : {tuple(user_embeddings.shape)}")

    print(f"Item embeddings : {tuple(item_embeddings.shape)}")

    # Summarize model parameter counts and checkpoint sizes.
    two_tower_parameters = count_model_parameters(final_two_tower)

    neumf_parameters = count_model_parameters(final_neumf)

    model_size_summary = pl.DataFrame(
        [
            {
                "model": "Two-Tower",
                "parameters": (two_tower_parameters["total_parameters"]),
                "checkpoint_mb": artifact_size_mb(model_release.two_tower_model_path),
            },
            {
                "model": "NeuMF",
                "parameters": (neumf_parameters["total_parameters"]),
                "checkpoint_mb": artifact_size_mb(model_release.neumf_model_path),
            },
        ]
    ).with_columns(pl.col("checkpoint_mb").round(3))

    show("Model Size Summary", model_size_summary)

    # Measure embedding memory use and on-disk size.
    embedding_artifact_summary = pl.DataFrame(
        [
            {
                "artifact": "User embeddings",
                "rows": user_embeddings.shape[0],
                "embedding_dim": user_embeddings.shape[1],
                "memory_mb": tensor_memory_mb(user_embeddings),
                "disk_mb": artifact_size_mb(model_release.user_embeddings_path),
            },
            {
                "artifact": "Item embeddings",
                "rows": item_embeddings.shape[0],
                "embedding_dim": item_embeddings.shape[1],
                "memory_mb": tensor_memory_mb(item_embeddings),
                "disk_mb": artifact_size_mb(model_release.item_embeddings_path),
            },
        ]
    ).with_columns(
        pl.col(
            "memory_mb",
            "disk_mb",
        ).round(3)
    )

    show("Embedding Artifact Summary", embedding_artifact_summary)

    # Project item-embedding memory across catalog sizes.
    item_embedding_scaling = embedding_scaling_table(
        entity_counts=(
            N_ITEMS,
            100_000,
            1_000_000,
            10_000_000,
            100_000_000,
        ),
        embedding_dims=(
            32,
            64,
            128,
            256,
        ),
    )

    show("Item Embedding Scaling", item_embedding_scaling)

    # Select scaling projections for the deployed embedding dimension.
    selected_embedding_scaling = item_embedding_scaling.filter(
        pl.col("embedding_dim") == two_tower_config["embedding_dim"]
    ).with_columns((pl.col("memory_mb") / 1024).round(3).alias("memory_gb"))

    show("Selected Embedding Scaling", selected_embedding_scaling)

    # Benchmark retrieval at several online and batch sizes.
    retrieval_batch_sizes = (
        1,
        8,
        32,
        128,
        256,
        512,
        1024,
    )
    retrieval_batch_benchmark_rows = []

    for batch_size in retrieval_batch_sizes:
        benchmark = benchmark_embedding_retrieval(
            model=final_two_tower,
            item_embeddings=item_embeddings,
            batch_size=batch_size,
            device=device,
            warmup_runs=2,
            benchmark_runs=10,
        )

        retrieval_batch_benchmark_rows.append(benchmark)

    retrieval_batch_benchmarks_raw = pl.DataFrame(retrieval_batch_benchmark_rows)

    retrieval_batch_benchmarks = retrieval_batch_benchmarks_raw.with_columns(
        pl.exclude(
            "batch_size",
            "items",
        ).round(3)
    )

    show("Retrieval Batch Benchmarks", retrieval_batch_benchmarks)

    # Derive retrieval latency and throughput by batch size.
    retrieval_latency_summary = (
        retrieval_batch_benchmarks_raw.with_columns(
            (pl.col("avg_batch_seconds") * 1000).alias("batch_latency_ms"),
            (pl.col("avg_batch_seconds") / pl.col("batch_size") * 1000).alias(
                "effective_latency_per_user_ms"
            ),
        )
        .select(
            "batch_size",
            "batch_latency_ms",
            "effective_latency_per_user_ms",
            "users_per_second",
            "million_scores_per_second",
        )
        .with_columns(pl.exclude("batch_size").round(3))
    )

    show("Retrieval Latency Summary", retrieval_latency_summary)

    # Project ranking cost across candidate-pool sizes.
    ranking_candidate_sizes = (
        50,
        100,
        250,
        500,
        1000,
        2000,
    )

    ranking_cost_projection = pl.DataFrame(
        [
            {
                "candidate_size": candidate_size,
                "neural_scores_per_user": candidate_size,
                "catalog_retained_pct": (candidate_size / N_ITEMS * 100),
                "neural_score_reduction_pct": ((1 - candidate_size / N_ITEMS) * 100),
            }
            for candidate_size in ranking_candidate_sizes
        ]
    ).with_columns(
        pl.exclude(
            "candidate_size",
            "neural_scores_per_user",
        ).round(3)
    )

    show("Ranking Cost Projection", ranking_cost_projection)

    # Benchmark NeuMF scoring at several candidate depths.
    ranking_benchmark_rows = []

    for candidate_size in (
        100,
        250,
        500,
        1000,
    ):
        benchmark = benchmark_candidate_ranking(
            ranker=final_neumf,
            batch_size=EVAL_BATCH_SIZE,
            candidate_size=candidate_size,
            n_users=N_USERS,
            n_items=N_ITEMS,
            score_batch_size=RANKING_SCORE_BATCH_SIZE,
            device=device,
            seed=RANDOM_SEED,
            warmup_runs=2,
            benchmark_runs=5,
        )

        ranking_benchmark_rows.append(benchmark)

    ranking_benchmarks_raw = pl.DataFrame(ranking_benchmark_rows)

    ranking_benchmarks = ranking_benchmarks_raw.with_columns(
        pl.exclude(
            "batch_size",
            "candidate_size",
        ).round(3)
    )

    show("Ranking Benchmarks", ranking_benchmarks)

    # Summarize ranking latency and throughput.
    ranking_latency_summary = (
        ranking_benchmarks_raw.with_columns(
            (pl.col("avg_batch_seconds") / pl.col("batch_size") * 1000).alias(
                "effective_latency_per_user_ms"
            )
        )
        .select(
            "candidate_size",
            "batch_latency_ms",
            "effective_latency_per_user_ms",
            "users_per_second",
            "million_scores_per_second",
        )
        .with_columns(pl.exclude("candidate_size").round(3))
    )

    show("Ranking Latency Summary", ranking_latency_summary)

    # Compare throughput for the chosen retrieval and ranking settings.
    selected_retrieval_benchmark = retrieval_batch_benchmarks_raw.filter(
        pl.col("batch_size") == EVAL_BATCH_SIZE
    ).row(
        0,
        named=True,
    )

    selected_ranking_benchmark = ranking_benchmarks_raw.filter(
        pl.col("candidate_size") == RANKING_CANDIDATE_SIZE
    ).row(
        0,
        named=True,
    )

    stage_throughput_comparison = pl.DataFrame(
        [
            {
                "stage": "Two-Tower Retrieval",
                "items_scored_per_user": N_ITEMS,
                "users_per_second": (selected_retrieval_benchmark["users_per_second"]),
                "million_scores_per_second": (
                    selected_retrieval_benchmark["million_scores_per_second"]
                ),
            },
            {
                "stage": "NeuMF Ranking",
                "items_scored_per_user": (RANKING_CANDIDATE_SIZE),
                "users_per_second": (selected_ranking_benchmark["users_per_second"]),
                "million_scores_per_second": (
                    selected_ranking_benchmark["million_scores_per_second"]
                ),
            },
        ]
    ).with_columns(
        pl.exclude(
            "stage",
            "items_scored_per_user",
        ).round(3)
    )

    show("Stage Throughput Comparison", stage_throughput_comparison)

    # Project exact-retrieval latency for larger catalogs.
    online_retrieval_benchmark = retrieval_batch_benchmarks_raw.filter(
        pl.col("batch_size") == 1
    ).row(
        0,
        named=True,
    )

    batched_retrieval_benchmark = retrieval_batch_benchmarks_raw.filter(
        pl.col("batch_size") == EVAL_BATCH_SIZE
    ).row(
        0,
        named=True,
    )

    current_catalog_size = N_ITEMS

    catalog_sizes = (
        current_catalog_size,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
    )

    exact_retrieval_projection = []

    for catalog_size in catalog_sizes:
        scale_factor = catalog_size / current_catalog_size

        online_latency_ms = (
            online_retrieval_benchmark["avg_batch_seconds"] * 1000 * scale_factor
        )

        batched_latency_ms = (
            batched_retrieval_benchmark["avg_batch_seconds"] * 1000 * scale_factor
        )

        exact_retrieval_projection.append(
            {
                "catalog_size": catalog_size,
                "scale_vs_current": scale_factor,
                "estimated_online_latency_ms": online_latency_ms,
                "estimated_batch_256_latency_ms": batched_latency_ms,
            }
        )

    exact_retrieval_projection = pl.DataFrame(exact_retrieval_projection).with_columns(
        pl.exclude("catalog_size").round(3)
    )

    show("Exact Retrieval Projection", exact_retrieval_projection)

    # Compare exact retrieval with approximate index designs.
    retrieval_architecture_matrix = pl.DataFrame(
        [
            {
                "approach": "Exact Matrix Search",
                "retrieval_quality": "Exact",
                "latency_scaling": "Linear with catalog",
                "memory_overhead": "Low",
                "index_build_cost": "None",
                "update_complexity": "Low",
                "recommended_scale": "Small / medium catalog",
            },
            {
                "approach": "HNSW",
                "retrieval_quality": "Approximate / high recall",
                "latency_scaling": "Sub-linear search",
                "memory_overhead": "High",
                "index_build_cost": "High",
                "update_complexity": "Medium",
                "recommended_scale": "Millions+ items",
            },
            {
                "approach": "IVF",
                "retrieval_quality": "Approximate",
                "latency_scaling": "Partition-dependent",
                "memory_overhead": "Medium",
                "index_build_cost": "Medium",
                "update_complexity": "Medium",
                "recommended_scale": "Millions+ items",
            },
        ]
    )

    show("Retrieval Architecture Matrix", retrieval_architecture_matrix)

    # Combine training and warm-validation interactions for observed history.
    all_observed_interactions = pl.concat(
        [
            train,
            warm_validation,
        ],
        how="vertical",
    )

    assert all_observed_interactions.filter(pl.col("item_idx") < 0).height == 0

    popularity_concentration_summary = popularity_concentration(
        interactions=all_observed_interactions,
        item_col="item_idx",
    )

    show("Popularity Concentration Summary", popularity_concentration_summary)

    # Split items into popularity head and long tail.
    item_popularity = (
        all_observed_interactions.group_by("item_idx")
        .len()
        .rename(
            {
                "len": "interaction_count",
            }
        )
        .sort(
            "interaction_count",
            descending=True,
        )
        .with_row_index(
            "popularity_rank",
            offset=1,
        )
    )

    HEAD_CATALOG_FRACTION = 0.10

    head_item_count = int(np.ceil(N_ITEMS * HEAD_CATALOG_FRACTION))

    head_catalog = item_popularity.head(head_item_count)

    tail_catalog = item_popularity.slice(head_item_count)

    head_interactions = head_catalog["interaction_count"].sum()

    tail_interactions = tail_catalog["interaction_count"].sum()

    total_interactions = head_interactions + tail_interactions

    catalog_segment_summary = pl.DataFrame(
        [
            {
                "segment": "Head",
                "items": head_catalog.height,
                "catalog_share_pct": (head_catalog.height / N_ITEMS * 100),
                "interaction_share_pct": (head_interactions / total_interactions * 100),
            },
            {
                "segment": "Long Tail",
                "items": tail_catalog.height,
                "catalog_share_pct": (tail_catalog.height / N_ITEMS * 100),
                "interaction_share_pct": (tail_interactions / total_interactions * 100),
            },
        ]
    ).with_columns(
        pl.exclude(
            "segment",
            "items",
        ).round(3)
    )

    show("Catalog Segment Summary", catalog_segment_summary)

    # Build seen-item histories and identify high-popularity items.
    final_seen_items = build_seen_items_csr(
        interactions=all_observed_interactions,
        n_users=N_USERS,
    )

    head_items = (
        head_catalog["item_idx"]
        .to_numpy()
        .astype(
            np.int32,
            copy=False,
        )
    )

    print(f"Unique seen user-item pairs : {final_seen_items.indices.size:,}")

    print(f"Head catalog items          : {head_items.size:,}")

    # Generate retrieval-only Top-50 recommendations.
    retrieval_recommendations = generate_recommendations(
        retrieval_model=final_two_tower,
        item_embeddings=item_embeddings,
        seen_items=final_seen_items,
        n_users=N_USERS,
        user_batch_size=EVAL_BATCH_SIZE,
        top_k=50,
        device=device,
        desc="Two-Tower recommendations",
    )

    print(f"Retrieval recommendations shape: {retrieval_recommendations.shape}")

    # Generate reranked Top-50 recommendations.
    reranked_recommendations = generate_recommendations(
        retrieval_model=final_two_tower,
        item_embeddings=item_embeddings,
        seen_items=final_seen_items,
        n_users=N_USERS,
        user_batch_size=EVAL_BATCH_SIZE,
        top_k=50,
        device=device,
        ranker=final_neumf,
        candidate_size=RANKING_CANDIDATE_SIZE,
        score_batch_size=RANKING_SCORE_BATCH_SIZE,
        desc="Two-Tower + NeuMF recommendations",
    )

    print(f"Reranked recommendations shape: {reranked_recommendations.shape}")

    # Measure catalog exposure in Top-10 recommendations.
    top10_catalog_metrics = pl.DataFrame(
        [
            {
                "pipeline": "Two-Tower",
                **recommendation_catalog_metrics(
                    recommendations=(retrieval_recommendations[:, :10]),
                    n_items=N_ITEMS,
                    head_items=head_items,
                ),
            },
            {
                "pipeline": "Two-Tower + NeuMF",
                **recommendation_catalog_metrics(
                    recommendations=(reranked_recommendations[:, :10]),
                    n_items=N_ITEMS,
                    head_items=head_items,
                ),
            },
        ]
    ).with_columns(
        pl.exclude(
            "pipeline",
            "recommendations",
            "unique_items_recommended",
            "unique_head_items",
            "unique_tail_items",
        ).round(3)
    )

    show("Top10 Catalog Metrics", top10_catalog_metrics)

    # Measure catalog exposure in Top-50 recommendations.
    top50_catalog_metrics = pl.DataFrame(
        [
            {
                "pipeline": "Two-Tower",
                **recommendation_catalog_metrics(
                    recommendations=(retrieval_recommendations),
                    n_items=N_ITEMS,
                    head_items=head_items,
                ),
            },
            {
                "pipeline": "Two-Tower + NeuMF",
                **recommendation_catalog_metrics(
                    recommendations=(reranked_recommendations),
                    n_items=N_ITEMS,
                    head_items=head_items,
                ),
            },
        ]
    ).with_columns(
        pl.exclude(
            "pipeline",
            "recommendations",
            "unique_items_recommended",
            "unique_head_items",
            "unique_tail_items",
        ).round(3)
    )

    show("Top50 Catalog Metrics", top50_catalog_metrics)

    # Measure popularity and novelty for Top-10 recommendations.
    total_observed_interactions = all_observed_interactions.height

    retrieval_top10_popularity = recommendation_popularity_metrics(
        recommendations=(retrieval_recommendations[:, :10]),
        item_popularity=item_popularity,
        total_interactions=(total_observed_interactions),
    )

    reranked_top10_popularity = recommendation_popularity_metrics(
        recommendations=(reranked_recommendations[:, :10]),
        item_popularity=item_popularity,
        total_interactions=(total_observed_interactions),
    )

    top10_popularity_comparison = pl.DataFrame(
        [
            {
                "pipeline": "Two-Tower",
                **retrieval_top10_popularity,
            },
            {
                "pipeline": "Two-Tower + NeuMF",
                **reranked_top10_popularity,
            },
        ]
    ).with_columns(pl.exclude("pipeline").round(3))

    show("Top10 Popularity Comparison", top10_popularity_comparison)

    # Measure popularity and novelty for Top-50 recommendations.
    retrieval_top50_popularity = recommendation_popularity_metrics(
        recommendations=(retrieval_recommendations),
        item_popularity=item_popularity,
        total_interactions=(total_observed_interactions),
    )

    reranked_top50_popularity = recommendation_popularity_metrics(
        recommendations=(reranked_recommendations),
        item_popularity=item_popularity,
        total_interactions=(total_observed_interactions),
    )

    top50_popularity_comparison = pl.DataFrame(
        [
            {
                "pipeline": "Two-Tower",
                **retrieval_top50_popularity,
            },
            {
                "pipeline": "Two-Tower + NeuMF",
                **reranked_top50_popularity,
            },
        ]
    ).with_columns(pl.exclude("pipeline").round(3))

    show("Top50 Popularity Comparison", top50_popularity_comparison)

    # Summarize ID-based cold-start capabilities and fallbacks.
    cold_start_capability = pl.DataFrame(
        [
            {
                "scenario": "Known user / known movie",
                "supported": True,
                "fallback": "Two-Tower + NeuMF",
            },
            {
                "scenario": "New user / known movie",
                "supported": False,
                "fallback": "Popularity / contextual onboarding",
            },
            {
                "scenario": "Known user / new movie",
                "supported": False,
                "fallback": "Content-based item embedding",
            },
            {
                "scenario": "New user / new movie",
                "supported": False,
                "fallback": "Popularity + content metadata",
            },
        ]
    )

    show("Cold Start Capability", cold_start_capability)

    # Benchmark ranking performance across batch sizes.
    ranking_batch_sizes = (
        1,
        8,
        32,
        128,
        256,
    )

    ranking_batch_rows = []

    for batch_size in ranking_batch_sizes:
        benchmark = benchmark_candidate_ranking(
            ranker=final_neumf,
            batch_size=batch_size,
            candidate_size=RANKING_CANDIDATE_SIZE,
            n_users=N_USERS,
            n_items=N_ITEMS,
            score_batch_size=RANKING_SCORE_BATCH_SIZE,
            device=device,
            seed=RANDOM_SEED,
            warmup_runs=2,
            benchmark_runs=10,
        )

        ranking_batch_rows.append(benchmark)

    ranking_batch_benchmarks_raw = pl.DataFrame(ranking_batch_rows)

    ranking_batch_benchmarks = (
        ranking_batch_benchmarks_raw.with_columns(
            (pl.col("avg_batch_seconds") * 1000).alias("batch_latency_ms"),
            (pl.col("avg_batch_seconds") / pl.col("batch_size") * 1000).alias(
                "effective_latency_per_user_ms"
            ),
        )
        .select(
            "batch_size",
            "candidate_size",
            "batch_latency_ms",
            "effective_latency_per_user_ms",
            "users_per_second",
            "million_scores_per_second",
        )
        .with_columns(
            pl.exclude(
                "batch_size",
                "candidate_size",
            ).round(3)
        )
    )

    show("Ranking Batch Benchmarks", ranking_batch_benchmarks)

    # Estimate combined retrieval and ranking compute costs.
    serving_latency_rows = []

    for batch_size in ranking_batch_sizes:
        retrieval_row = retrieval_batch_benchmarks_raw.filter(
            pl.col("batch_size") == batch_size
        ).row(
            0,
            named=True,
        )

        ranking_row = ranking_batch_benchmarks_raw.filter(
            pl.col("batch_size") == batch_size
        ).row(
            0,
            named=True,
        )

        retrieval_latency_ms = retrieval_row["avg_batch_seconds"] * 1000

        ranking_latency_ms = ranking_row["avg_batch_seconds"] * 1000

        total_latency_ms = retrieval_latency_ms + ranking_latency_ms

        serving_latency_rows.append(
            {
                "batch_size": batch_size,
                "retrieval_latency_ms": (retrieval_latency_ms),
                "ranking_latency_ms": (ranking_latency_ms),
                "estimated_pipeline_latency_ms": (total_latency_ms),
                "estimated_latency_per_user_ms": (total_latency_ms / batch_size),
            }
        )

    serving_latency_summary = pl.DataFrame(serving_latency_rows).with_columns(
        pl.exclude("batch_size").round(3)
    )

    show("Serving Latency Summary", serving_latency_summary)

    # Compare interactive, micro-batched, and offline serving modes.
    serving_mode_matrix = pl.DataFrame(
        [
            {
                "mode": "Online request",
                "typical_batch": "1-8 users",
                "priority": "P95/P99 latency",
                "retrieval": "Exact / ANN",
                "ranking": "Real-time NeuMF",
                "freshness": "High",
                "use_case": "Interactive recommendation API",
            },
            {
                "mode": "Micro-batched online",
                "typical_batch": "8-128 users",
                "priority": "Latency + throughput",
                "retrieval": "Exact / ANN",
                "ranking": "Batched NeuMF",
                "freshness": "High",
                "use_case": "High-QPS serving",
            },
            {
                "mode": "Offline batch",
                "typical_batch": "128+ users",
                "priority": "Throughput",
                "retrieval": "Batched exact / ANN",
                "ranking": "Batched NeuMF",
                "freshness": "Scheduled",
                "use_case": "Precomputed recommendation feeds",
            },
        ]
    )

    show("Serving Mode Matrix", serving_mode_matrix)

    # Summarize sizes of immutable model-release artifacts.
    model_release_artifacts = pl.DataFrame(
        [
            {
                "artifact": artifact_name,
                "filename": metadata["filename"],
                "size_mb": (metadata["size_bytes"] / (1024**2)),
            }
            for artifact_name, metadata in model_release.manifest["artifacts"].items()
        ]
    ).with_columns(pl.col("size_mb").round(3))

    show("Model Release Artifacts", model_release_artifacts)

    # Compare full release size with online-only model assets.
    total_model_release_mb = model_release_artifacts["size_mb"].sum()

    online_model_artifact_names = {
        "two_tower_model",
        "neumf_model",
        "two_tower_item_embeddings",
        "two_tower_embedding_metadata",
    }

    online_model_artifact_mb = model_release_artifacts.filter(
        pl.col("artifact").is_in(online_model_artifact_names)
    )["size_mb"].sum()

    print(f"Complete model release : {total_model_release_mb:.3f} MB")

    print(f"Online model assets    : {online_model_artifact_mb:.3f} MB")

    # Project embedding memory at alternative numerical precisions.
    precision_projection_rows = []

    for catalog_size in (
        N_ITEMS,
        1_000_000,
        10_000_000,
        100_000_000,
    ):
        for precision, bytes_per_value in [
            ("FP32", 4),
            ("FP16", 2),
            ("INT8", 1),
        ]:
            memory_mb = (
                catalog_size
                * two_tower_config["embedding_dim"]
                * bytes_per_value
                / (1024**2)
            )

            precision_projection_rows.append(
                {
                    "catalog_size": catalog_size,
                    "precision": precision,
                    "memory_mb": memory_mb,
                    "memory_gb": memory_mb / 1024,
                }
            )

    precision_projection = pl.DataFrame(precision_projection_rows).with_columns(
        pl.col(
            "memory_mb",
            "memory_gb",
        ).round(3)
    )

    show("Precision Projection", precision_projection)

    # Outline feature and model freshness strategies.
    freshness_strategy = pl.DataFrame(
        [
            {
                "component": "Interaction features",
                "refresh_strategy": "Streaming / frequent batch",
                "suggested_cadence": "Minutes-hours",
                "reason": "Capture recent user behavior",
            },
            {
                "component": "Popularity fallback",
                "refresh_strategy": "Incremental aggregation",
                "suggested_cadence": "Hourly / daily",
                "reason": "Track trending content",
            },
            {
                "component": "User embeddings",
                "refresh_strategy": "Batch or online tower inference",
                "suggested_cadence": "Hours / daily",
                "reason": "Reflect changing user preferences",
            },
            {
                "component": "Item embeddings",
                "refresh_strategy": "Batch tower inference",
                "suggested_cadence": "On item change / daily",
                "reason": "Update retrieval index",
            },
            {
                "component": "Two-Tower model",
                "refresh_strategy": "Full retraining",
                "suggested_cadence": "Daily / weekly",
                "reason": "Refresh retrieval representation",
            },
            {
                "component": "NeuMF ranker",
                "refresh_strategy": "Full retraining",
                "suggested_cadence": "Daily / weekly",
                "reason": "Refresh ranking preferences",
            },
        ]
    )

    show("Freshness Strategy", freshness_strategy)

    # List system, recommendation, and business monitoring signals.
    production_monitoring = pl.DataFrame(
        [
            {
                "category": "System",
                "metric": "P50 / P95 / P99 latency",
                "purpose": "Serving performance",
            },
            {
                "category": "System",
                "metric": "Requests per second",
                "purpose": "Serving throughput",
            },
            {
                "category": "System",
                "metric": "Error rate",
                "purpose": "API reliability",
            },
            {
                "category": "System",
                "metric": "CPU / GPU / memory utilization",
                "purpose": "Resource saturation",
            },
            {
                "category": "Retrieval",
                "metric": "Candidate recall proxy",
                "purpose": "Retrieval degradation",
            },
            {
                "category": "Retrieval",
                "metric": "ANN index freshness",
                "purpose": "Embedding/index staleness",
            },
            {
                "category": "Ranking",
                "metric": "Score distribution",
                "purpose": "Model drift",
            },
            {
                "category": "Recommendation",
                "metric": "Catalog coverage",
                "purpose": "Content concentration",
            },
            {
                "category": "Recommendation",
                "metric": "Head / tail exposure",
                "purpose": "Popularity bias",
            },
            {
                "category": "Recommendation",
                "metric": "Novelty / diversity",
                "purpose": "Discovery quality",
            },
            {
                "category": "Business",
                "metric": "CTR",
                "purpose": "Recommendation engagement",
            },
            {
                "category": "Business",
                "metric": "Watch / completion rate",
                "purpose": "Content consumption quality",
            },
            {
                "category": "Business",
                "metric": "Session length",
                "purpose": "User engagement",
            },
            {
                "category": "Business",
                "metric": "Retention",
                "purpose": "Long-term product impact",
            },
        ]
    )

    show("Production Monitoring", production_monitoring)

    # Contrast offline evaluation and online product metrics.
    offline_online_metrics = pl.DataFrame(
        [
            {
                "evaluation": "Offline",
                "metric": "Recall@K",
                "measures": "Relevant-item retrieval",
            },
            {
                "evaluation": "Offline",
                "metric": "NDCG@K",
                "measures": "Ranking quality",
            },
            {
                "evaluation": "Offline",
                "metric": "MRR@K",
                "measures": "First relevant-item rank",
            },
            {
                "evaluation": "Offline",
                "metric": "Catalog coverage",
                "measures": "Catalog utilization",
            },
            {
                "evaluation": "Offline",
                "metric": "Novelty",
                "measures": "Popularity-adjusted discovery",
            },
            {
                "evaluation": "Online",
                "metric": "CTR",
                "measures": "Immediate engagement",
            },
            {
                "evaluation": "Online",
                "metric": "Watch / completion rate",
                "measures": "Consumption quality",
            },
            {
                "evaluation": "Online",
                "metric": "Session duration",
                "measures": "Engagement depth",
            },
            {
                "evaluation": "Online",
                "metric": "Retention",
                "measures": "Long-term user value",
            },
        ]
    )

    show("Offline Online Metrics", offline_online_metrics)

    # Summarize architecture decisions and their tradeoffs.
    neural_score_reduction_pct = (1 - RANKING_CANDIDATE_SIZE / N_ITEMS) * 100

    production_tradeoff_matrix = pl.DataFrame(
        [
            {
                "decision": "Retrieval model",
                "selected": "Two-Tower",
                "alternative": "Full-catalog neural scoring",
                "benefit": "High-throughput vector retrieval",
                "tradeoff": "Requires separate ranking stage",
            },
            {
                "decision": "Ranking model",
                "selected": "NeuMF",
                "alternative": "Retrieval score only",
                "benefit": "Higher Top-K ranking quality",
                "tradeoff": "Higher pairwise inference cost",
            },
            {
                "decision": "Candidate size",
                "selected": f"{RANKING_CANDIDATE_SIZE}",
                "alternative": "100 / 250 / 1000",
                "benefit": (
                    f"{neural_score_reduction_pct:.3f}% "
                    "fewer neural scores than full catalog"
                ),
                "tradeoff": ("Retrieval coverage limits downstream ranking"),
            },
            {
                "decision": "Current retrieval search",
                "selected": "Exact matrix search",
                "alternative": "ANN",
                "benefit": "Exact results and low complexity",
                "tradeoff": "Latency scales linearly with catalog",
            },
            {
                "decision": "Large-scale retrieval",
                "selected": "ANN-ready embeddings",
                "alternative": "Exact search",
                "benefit": "Sub-linear candidate retrieval",
                "tradeoff": "Approximation and index complexity",
            },
            {
                "decision": "Embedding dimension",
                "selected": "64",
                "alternative": "32 / 128 / 256",
                "benefit": "Compact representation",
                "tradeoff": "Capacity vs memory",
            },
            {
                "decision": "Feedback objective",
                "selected": "Implicit positive feedback",
                "alternative": "Explicit rating prediction",
                "benefit": "Aligned with Top-K ranking",
                "tradeoff": "Requires negative sampling",
            },
            {
                "decision": "Evaluation split",
                "selected": "Per-user temporal",
                "alternative": "Random split",
                "benefit": "Future-interaction simulation",
                "tradeoff": "More difficult evaluation",
            },
            {
                "decision": "Cold start",
                "selected": "Fallback + future metadata towers",
                "alternative": "ID-only embeddings",
                "benefit": "Supports unseen entities",
                "tradeoff": "Additional features and infrastructure",
            },
            {
                "decision": "Recommendation diversity",
                "selected": "Monitor + future reranking",
                "alternative": "Accuracy-only ranking",
                "benefit": "Improved discovery and coverage",
                "tradeoff": "Potential relevance trade-off",
            },
        ]
    )

    show("Production Tradeoff Matrix", production_tradeoff_matrix)

    # Summarize the selected system and frozen final-test metrics.
    single_user_latency = serving_latency_summary.filter(pl.col("batch_size") == 1).row(
        0,
        named=True,
    )

    selected_system_summary = pl.DataFrame(
        [
            {
                "metric": "Users",
                "value": f"{N_USERS:,}",
            },
            {
                "metric": "Movies",
                "value": f"{N_ITEMS:,}",
            },
            {
                "metric": "Retrieval embedding dimension",
                "value": str(two_tower_config["embedding_dim"]),
            },
            {
                "metric": "Selected candidate size",
                "value": f"{RANKING_CANDIDATE_SIZE:,}",
            },
            {
                "metric": "Complete model release",
                "value": (f"{total_model_release_mb:.3f} MB"),
            },
            {
                "metric": "Online model assets",
                "value": (f"{online_model_artifact_mb:.3f} MB"),
            },
            {
                "metric": "Item embedding footprint",
                "value": (f"{tensor_memory_mb(item_embeddings):.3f} MB"),
            },
            {
                "metric": "Single-user retrieval compute",
                "value": (f"{single_user_latency['retrieval_latency_ms']:.3f} ms"),
            },
            {
                "metric": "Single-user NeuMF ranking compute",
                "value": (f"{single_user_latency['ranking_latency_ms']:.3f} ms"),
            },
            {
                "metric": "Estimated single-user pipeline compute",
                "value": (
                    f"{single_user_latency['estimated_pipeline_latency_ms']:.3f} ms"
                ),
            },
            {
                "metric": "Test Candidate Recall@500",
                "value": (f"{final_pipeline_results['candidate_recall']:.3f}"),
            },
            {
                "metric": "Test Recall@10",
                "value": (f"{final_pipeline_metrics['recall@10']:.3f}"),
            },
            {
                "metric": "Test NDCG@10",
                "value": (f"{final_pipeline_metrics['ndcg@10']:.3f}"),
            },
            {
                "metric": "Test Recall@50",
                "value": (f"{final_pipeline_metrics['recall@50']:.3f}"),
            },
            {
                "metric": "Test NDCG@50",
                "value": (f"{final_pipeline_metrics['ndcg@50']:.3f}"),
            },
        ]
    )

    show("Selected System Summary", selected_system_summary)

    # Write this run's scalability report.
    scalability_results = {
        "artifact_schema_version": 2,
        "processed_data": {
            "data_run_id": (processed_manifest["data_run_id"]),
            "dataset_fingerprint": (processed_manifest["dataset_fingerprint"]),
            "test_sha256": current_test_sha256,
        },
        "evaluation_data": {
            "data_run_id": (evaluation_data["data_run_id"]),
            "dataset_fingerprint": (evaluation_data["dataset_fingerprint"]),
            "test_sha256": (evaluation_data["test_sha256"]),
        },
        "model_release": {
            "model_release_id": (model_release.model_release_id),
            "model_release_fingerprint": (model_release.model_release_fingerprint),
            "manifest_sha256": (current_model_manifest_sha256),
            "lineage": model_release.lineage,
        },
        "architecture": {
            "retrieval_model": "TwoTowerRecommender",
            "ranking_model": "NeuMF",
            "candidate_size": RANKING_CANDIDATE_SIZE,
            "embedding_dim": (two_tower_config["embedding_dim"]),
        },
        "final_test": {
            "candidate_recall": (final_pipeline_results["candidate_recall"]),
            "retrieval": final_retrieval_metrics,
            "retrieval_reranking": (final_pipeline_metrics),
        },
        "artifacts": {
            "complete_model_release_mb": (total_model_release_mb),
            "online_model_assets_mb": (online_model_artifact_mb),
            "release_artifacts": (model_release_artifacts.to_dicts()),
        },
        "retrieval_batch_benchmarks": (retrieval_latency_summary.to_dicts()),
        "ranking_batch_benchmarks": (ranking_batch_benchmarks.to_dicts()),
        "serving_latency": (serving_latency_summary.to_dicts()),
        "embedding_scaling": (selected_embedding_scaling.to_dicts()),
        "precision_scaling": (precision_projection.to_dicts()),
        "popularity_concentration": (popularity_concentration_summary.to_dicts()),
        "catalog_segments": (catalog_segment_summary.to_dicts()),
        "top10_catalog_metrics": (top10_catalog_metrics.to_dicts()),
        "top50_catalog_metrics": (top50_catalog_metrics.to_dicts()),
        "top10_popularity_metrics": (top10_popularity_comparison.to_dicts()),
        "top50_popularity_metrics": (top50_popularity_comparison.to_dicts()),
        "cold_start": (cold_start_capability.to_dicts()),
        "tradeoffs": (production_tradeoff_matrix.to_dicts()),
    }

    save_json(
        scalability_results,
        scalability_output,
    )

    print(f"Saved scalability results: {scalability_output}")


if __name__ == "__main__":
    main()
