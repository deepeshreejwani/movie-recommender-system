from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from recsys.config import (
    API_MAX_K,
    COLD_ITEM_INDEX,
    FINAL_TRAINING_PROTOCOL,
    RANKING_CANDIDATE_SIZE,
    RANKING_SCORE_BATCH_SIZE,
    SERVING_DIR,
    SERVING_MOVIE_METADATA_PATH,
    SERVING_SEEN_INDICES_PATH,
    SERVING_SEEN_INDPTR_PATH,
    SERVING_USER_MAPPING_PATH,
)
from recsys.data.loader import (
    load_json,
    load_parquet,
)
from recsys.evaluation.evaluator import (
    SeenItemsCSR,
    mask_seen_items,
    score_candidate_pairs,
    validate_finite_scores,
)
from recsys.models.ncf import NeuMF
from recsys.models.two_tower import (
    TwoTowerRecommender,
)
from recsys.utils.artifacts import (
    canonical_id_mapping_sha256,
    canonical_json_sha256,
    sha256_file,
    validate_sha256,
)
from recsys.utils.artifacts import (
    require_metadata_dict as _require_dict,
)
from recsys.utils.artifacts import (
    require_metadata_int as _require_int,
)
from recsys.utils.artifacts import (
    require_metadata_list as _require_list,
)
from recsys.utils.artifacts import (
    require_metadata_string as _require_string,
)
from recsys.utils.helpers import (
    get_torch_device,
)
from recsys.utils.model_io import (
    load_model_checkpoint,
    load_tensor,
    restore_model_from_checkpoint,
)
from recsys.utils.releases import (
    MODEL_RELEASE_MANIFEST_NAME,
    MODEL_RELEASE_MANIFEST_VERSION,
    MODEL_RELEASES_DIR,
    SERVING_RELEASE_MANIFEST_NAME,
    ModelRelease,
)

SUPPORTED_SERVING_BUNDLE_VERSION = 2
SUPPORTED_SERVING_RELEASE_POINTER_VERSION = 1

SERVING_RELEASES_DIR = SERVING_DIR / "releases"
SERVING_RELEASE_POINTER_PATH = SERVING_DIR / "current_release.json"


@dataclass(frozen=True)
class ServingRelease:
    """Validated immutable serving generation."""

    serving_release_id: str
    serving_release_fingerprint: str
    release_dir: Path
    manifest_path: Path
    manifest: dict[str, object]
    user_mapping_path: Path
    movie_metadata_path: Path
    seen_indptr_path: Path
    seen_indices_path: Path


def _resolve_relative_path(
    root: Path,
    relative_path: str,
    field_name: str,
) -> Path:
    """Resolve a relative path without allowing traversal outside root."""

    relative = Path(relative_path)

    if relative.is_absolute():
        raise ValueError(f"{field_name} must be relative.")

    root_resolved = root.resolve()

    resolved = (root / relative).resolve()

    try:
        resolved.relative_to(root_resolved)

    except ValueError as error:
        raise ValueError(
            f"{field_name} escapes its configured artifact root."
        ) from error

    return resolved


def _require_plain_filename(
    filename: str,
    field_name: str,
) -> str:
    """Require a filename with no parent directory components."""

    candidate = Path(filename)

    if candidate.is_absolute() or candidate.name != filename:
        raise ValueError(f"{field_name} must be a plain filename.")

    return filename


def _validate_file_metadata(
    *,
    path: Path,
    metadata: dict,
    metadata_name: str,
) -> str:
    """Validate one artifact against declared size and SHA-256."""

    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")

    expected_size = _require_int(
        metadata,
        "size_bytes",
        metadata_name,
    )

    actual_size = path.stat().st_size

    if actual_size != expected_size:
        raise RuntimeError(
            f"{metadata_name} size mismatch: "
            f"expected={expected_size}, actual={actual_size}."
        )

    expected_sha256 = validate_sha256(
        _require_string(
            metadata,
            "sha256",
            metadata_name,
        ),
        f"{metadata_name}.sha256",
    )

    actual_sha256 = sha256_file(path)

    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"{metadata_name} fingerprint mismatch.")

    return actual_sha256


def resolve_current_serving_release() -> ServingRelease:
    """
    Resolve the atomically published serving pointer and validate
    every serving-owned artifact inside the selected immutable release.
    """

    pointer = load_json(SERVING_RELEASE_POINTER_PATH)

    pointer_version = pointer.get("pointer_version")

    if pointer_version != SUPPORTED_SERVING_RELEASE_POINTER_VERSION:
        raise RuntimeError(
            "Unsupported serving release pointer version: "
            f"expected={SUPPORTED_SERVING_RELEASE_POINTER_VERSION}, "
            f"actual={pointer_version}."
        )

    serving_release_id = _require_string(
        pointer,
        "serving_release_id",
        "serving_release_pointer",
    )

    serving_release_fingerprint = validate_sha256(
        _require_string(
            pointer,
            "serving_release_fingerprint",
            "serving_release_pointer",
        ),
        "serving_release_pointer.serving_release_fingerprint",
    )

    release_dir = _resolve_relative_path(
        root=SERVING_DIR,
        relative_path=_require_string(
            pointer,
            "serving_release_dir",
            "serving_release_pointer",
        ),
        field_name="serving_release_pointer.serving_release_dir",
    )

    if not release_dir.is_dir():
        raise FileNotFoundError(
            f"Current serving release directory does not exist: {release_dir}"
        )

    if release_dir.parent != SERVING_RELEASES_DIR.resolve():
        raise RuntimeError(
            "Current serving release is outside the configured "
            "serving releases directory."
        )

    if release_dir.name != serving_release_id:
        raise RuntimeError(
            "Serving release pointer ID does not match its release directory."
        )

    manifest_filename = _require_plain_filename(
        _require_string(
            pointer,
            "manifest_filename",
            "serving_release_pointer",
        ),
        "serving_release_pointer.manifest_filename",
    )

    if manifest_filename != SERVING_RELEASE_MANIFEST_NAME:
        raise RuntimeError(
            "Serving release pointer references an unexpected manifest filename."
        )

    manifest_path = release_dir / manifest_filename

    expected_manifest_sha256 = validate_sha256(
        _require_string(
            pointer,
            "manifest_sha256",
            "serving_release_pointer",
        ),
        "serving_release_pointer.manifest_sha256",
    )

    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError(
            "Serving release manifest fingerprint does not match the current pointer."
        )

    manifest = load_json(manifest_path)

    if manifest.get("serving_bundle_version") != SUPPORTED_SERVING_BUNDLE_VERSION:
        raise RuntimeError(
            "Unsupported serving bundle version: "
            f"expected={SUPPORTED_SERVING_BUNDLE_VERSION}, "
            f"actual={manifest.get('serving_bundle_version')}."
        )

    serving_release_metadata = _require_dict(
        manifest.get("serving_release"),
        "serving_manifest.serving_release",
    )

    if (
        _require_string(
            serving_release_metadata,
            "serving_release_id",
            "serving_manifest.serving_release",
        )
        != serving_release_id
    ):
        raise RuntimeError("Serving release pointer and manifest IDs do not match.")

    if (
        validate_sha256(
            _require_string(
                serving_release_metadata,
                "serving_release_fingerprint",
                "serving_manifest.serving_release",
            ),
            ("serving_manifest.serving_release.serving_release_fingerprint"),
        )
        != serving_release_fingerprint
    ):
        raise RuntimeError(
            "Serving release pointer and manifest fingerprints do not match."
        )

    runtime_artifacts = _require_dict(
        manifest.get("runtime_artifacts"),
        "serving_manifest.runtime_artifacts",
    )

    expected_files = {
        "user_mapping": SERVING_USER_MAPPING_PATH.name,
        "movie_metadata": SERVING_MOVIE_METADATA_PATH.name,
        "seen_indptr": SERVING_SEEN_INDPTR_PATH.name,
        "seen_indices": SERVING_SEEN_INDICES_PATH.name,
    }

    resolved_paths: dict[str, Path] = {}
    runtime_hashes: dict[str, str] = {}

    for artifact_name, expected_filename in expected_files.items():
        metadata_name = f"serving_manifest.runtime_artifacts[{artifact_name!r}]"

        metadata = _require_dict(
            runtime_artifacts.get(artifact_name),
            metadata_name,
        )

        filename = _require_plain_filename(
            _require_string(
                metadata,
                "filename",
                metadata_name,
            ),
            f"{metadata_name}.filename",
        )

        if filename != expected_filename:
            raise RuntimeError(
                "Serving release contains an unexpected filename "
                f"for '{artifact_name}'."
            )

        artifact_path = release_dir / filename

        runtime_hashes[artifact_name] = _validate_file_metadata(
            path=artifact_path,
            metadata=metadata,
            metadata_name=metadata_name,
        )

        resolved_paths[artifact_name] = artifact_path

    lineage = _require_dict(
        manifest.get("lineage"),
        "serving_manifest.lineage",
    )

    release = _require_dict(
        manifest.get("release"),
        "serving_manifest.release",
    )

    model_artifacts = _require_dict(
        manifest.get("model_artifacts"),
        "serving_manifest.model_artifacts",
    )

    identity_payload = {
        "serving_bundle_version": (SUPPORTED_SERVING_BUNDLE_VERSION),
        "lineage": dict(lineage),
        "model_release_id": _require_string(
            release,
            "model_release_id",
            "serving_manifest.release",
        ),
        "model_release_fingerprint": (
            validate_sha256(
                _require_string(
                    release,
                    "model_release_fingerprint",
                    "serving_manifest.release",
                ),
                ("serving_manifest.release.model_release_fingerprint"),
            )
        ),
        "runtime_artifact_hashes": {
            key: runtime_hashes[key] for key in sorted(runtime_hashes)
        },
        "model_artifact_hashes": {
            key: validate_sha256(
                _require_string(
                    model_artifacts,
                    key,
                    "serving_manifest.model_artifacts",
                ),
                (f"serving_manifest.model_artifacts.{key}"),
            )
            for key in sorted(model_artifacts)
        },
    }

    expected_fingerprint = canonical_json_sha256(identity_payload)

    if expected_fingerprint != serving_release_fingerprint:
        raise RuntimeError(
            "Serving release fingerprint does not match its immutable bundle contents."
        )

    expected_release_id = f"serving-{expected_fingerprint[:20]}"

    if expected_release_id != serving_release_id:
        raise RuntimeError(
            "Serving release ID does not match its content-derived fingerprint."
        )

    pointer_fields = {
        "serving_release_fingerprint": (serving_release_fingerprint),
        "model_release_id": (identity_payload["model_release_id"]),
        "model_release_fingerprint": (identity_payload["model_release_fingerprint"]),
        "release_lineage_id": _require_string(
            lineage,
            "release_lineage_id",
            "serving_manifest.lineage",
        ),
        "data_run_id": _require_string(
            lineage,
            "data_run_id",
            "serving_manifest.lineage",
        ),
    }

    for key, expected_value in pointer_fields.items():
        if pointer.get(key) != expected_value:
            raise RuntimeError(
                f"Serving release pointer does not match manifest field '{key}'."
            )

    return ServingRelease(
        serving_release_id=serving_release_id,
        serving_release_fingerprint=(serving_release_fingerprint),
        release_dir=release_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        user_mapping_path=resolved_paths["user_mapping"],
        movie_metadata_path=resolved_paths["movie_metadata"],
        seen_indptr_path=resolved_paths["seen_indptr"],
        seen_indices_path=resolved_paths["seen_indices"],
    )


def resolve_pinned_model_release(
    *,
    model_release_id: str,
    model_release_fingerprint: str,
    expected_manifest_sha256: str,
) -> ModelRelease:
    """
    Resolve the exact immutable model release pinned by a serving
    generation without consulting models/current_release.json.
    """

    model_release_fingerprint = validate_sha256(
        model_release_fingerprint,
        "serving_manifest.release.model_release_fingerprint",
    )

    expected_manifest_sha256 = validate_sha256(
        expected_manifest_sha256,
        "serving_manifest.release.model_release_manifest_sha256",
    )

    release_dir = (MODEL_RELEASES_DIR / model_release_id).resolve()

    if not release_dir.is_dir():
        raise FileNotFoundError(
            f"Pinned model release directory does not exist: {release_dir}"
        )

    if release_dir.parent != MODEL_RELEASES_DIR.resolve():
        raise RuntimeError(
            "Pinned model release is outside the configured model releases directory."
        )

    if release_dir.name != model_release_id:
        raise RuntimeError(
            "Pinned model release ID does not match its release directory."
        )

    manifest_path = release_dir / MODEL_RELEASE_MANIFEST_NAME

    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("Pinned model release manifest fingerprint mismatch.")

    manifest = load_json(manifest_path)

    if manifest.get("model_release_manifest_version") != MODEL_RELEASE_MANIFEST_VERSION:
        raise RuntimeError("Unsupported model release manifest version.")

    if (
        _require_string(
            manifest,
            "model_release_id",
            "model_release_manifest",
        )
        != model_release_id
    ):
        raise RuntimeError(
            "Pinned model release manifest ID does not match the serving bundle."
        )

    manifest_fingerprint = validate_sha256(
        _require_string(
            manifest,
            "model_release_fingerprint",
            "model_release_manifest",
        ),
        "model_release_manifest.model_release_fingerprint",
    )

    if manifest_fingerprint != model_release_fingerprint:
        raise RuntimeError(
            "Pinned model release fingerprint does not match the serving bundle."
        )

    lineage = _require_dict(
        manifest.get("lineage"),
        "model_release_manifest.lineage",
    )

    artifacts = _require_dict(
        manifest.get("artifacts"),
        "model_release_manifest.artifacts",
    )

    artifact_names = {
        "neumf_model",
        "two_tower_model",
        "two_tower_user_embeddings",
        "two_tower_item_embeddings",
        "two_tower_embedding_metadata",
    }

    if set(artifacts) != artifact_names:
        raise RuntimeError(
            "Pinned model release manifest contains an unexpected artifact set."
        )

    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}

    for artifact_name in sorted(artifact_names):
        metadata_name = f"model_release_manifest.artifacts[{artifact_name!r}]"

        metadata = _require_dict(
            artifacts.get(artifact_name),
            metadata_name,
        )

        filename = _require_plain_filename(
            _require_string(
                metadata,
                "filename",
                metadata_name,
            ),
            f"{metadata_name}.filename",
        )

        artifact_path = release_dir / filename

        artifact_hashes[artifact_name] = _validate_file_metadata(
            path=artifact_path,
            metadata=metadata,
            metadata_name=metadata_name,
        )

        artifact_paths[artifact_name] = artifact_path

    release_identity = {
        "model_release_manifest_version": (MODEL_RELEASE_MANIFEST_VERSION),
        "lineage": dict(lineage),
        "artifact_hashes": {
            key: artifact_hashes[key] for key in sorted(artifact_hashes)
        },
    }

    expected_fingerprint = canonical_json_sha256(release_identity)

    if expected_fingerprint != model_release_fingerprint:
        raise RuntimeError(
            "Pinned model release fingerprint does not match its artifact contents."
        )

    expected_release_id = f"model-{expected_fingerprint[:20]}"

    if expected_release_id != model_release_id:
        raise RuntimeError(
            "Pinned model release ID does not match its content-derived fingerprint."
        )

    catalog = _require_dict(
        manifest.get("catalog"),
        "model_release_manifest.catalog",
    )

    if (
        _require_int(
            catalog,
            "n_users",
            "model_release_manifest.catalog",
        )
        <= 0
    ):
        raise RuntimeError("Pinned model release user vocabulary must be positive.")

    if (
        _require_int(
            catalog,
            "n_items",
            "model_release_manifest.catalog",
        )
        <= 0
    ):
        raise RuntimeError("Pinned model release item vocabulary must be positive.")

    return ModelRelease(
        model_release_id=model_release_id,
        model_release_fingerprint=(model_release_fingerprint),
        release_dir=release_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        lineage={str(key): str(value) for key, value in lineage.items()},
        neumf_model_path=artifact_paths["neumf_model"],
        two_tower_model_path=artifact_paths["two_tower_model"],
        user_embeddings_path=artifact_paths["two_tower_user_embeddings"],
        item_embeddings_path=artifact_paths["two_tower_item_embeddings"],
        embedding_metadata_path=artifact_paths["two_tower_embedding_metadata"],
    )


@dataclass
class Recommendation:
    movie_id: int
    title: str
    genres: str
    score: float


class RecommendationService:
    """Two-Tower retrieval + NeuMF ranking inference service."""

    def __init__(self) -> None:
        self.device = get_torch_device()

        self.retrieval_model: TwoTowerRecommender | None = None

        self.ranker: NeuMF | None = None

        self.item_embeddings: torch.Tensor | None = None

        self.seen_items: SeenItemsCSR | None = None

        self.user_id_to_idx: dict[
            int,
            int,
        ] = {}

        self.movie_ids: np.ndarray | None = None

        self.movie_titles: np.ndarray | None = None

        self.movie_genres: np.ndarray | None = None

        self.manifest: dict[str, object] | None = None

        self.lineage: dict[str, str] | None = None

        self.model_release: ModelRelease | None = None

        self.serving_release: ServingRelease | None = None

        self.n_users = 0
        self.n_items = 0

        self.loaded = False
        self.manifest_validated = False

    def load(self) -> None:
        """
        Load and validate the complete serving bundle.

        The service is marked loaded only after every artifact
        passes validation and inference warmup succeeds.
        """

        self.loaded = False
        self.manifest_validated = False
        self.model_release = None
        self.serving_release = None

        self._load_and_validate_manifest()

        self._load_retrieval_model()
        self._load_ranker()
        self._load_embeddings()

        self._load_mappings_and_metadata()
        self._load_seen_items()

        self._validate_serving_bundle()

        self.warmup()

        self.loaded = True

    def _load_and_validate_manifest(
        self,
    ) -> None:
        """
        Resolve one immutable serving generation and validate its
        lineage, pinned model release, and cross-artifact fingerprints.
        """

        serving_release = resolve_current_serving_release()

        manifest = serving_release.manifest

        lineage = _require_dict(
            manifest.get("lineage"),
            "serving_manifest.lineage",
        )

        lineage_string_keys = (
            "data_run_id",
            "dataset_fingerprint",
            "preprocessing_protocol_version",
            "final_training_protocol",
            "train_sha256",
            "warm_validation_sha256",
            "user_mapping_sha256",
            "item_mapping_sha256",
            "release_lineage_id",
            "release_fingerprint",
        )

        normalized_lineage = {
            key: _require_string(
                lineage,
                key,
                "serving_manifest.lineage",
            )
            for key in lineage_string_keys
        }

        for key in (
            "dataset_fingerprint",
            "train_sha256",
            "warm_validation_sha256",
            "user_mapping_sha256",
            "item_mapping_sha256",
            "release_fingerprint",
        ):
            validate_sha256(
                normalized_lineage[key],
                (f"serving_manifest.lineage.{key}"),
            )

        if normalized_lineage["final_training_protocol"] != FINAL_TRAINING_PROTOCOL:
            raise RuntimeError(
                "Serving lineage final-training protocol does not "
                "match project configuration."
            )

        release_payload = {
            "data_run_id": (normalized_lineage["data_run_id"]),
            "dataset_fingerprint": (normalized_lineage["dataset_fingerprint"]),
            "preprocessing_protocol_version": (
                normalized_lineage["preprocessing_protocol_version"]
            ),
            "final_training_protocol": (normalized_lineage["final_training_protocol"]),
            "train_sha256": (normalized_lineage["train_sha256"]),
            "warm_validation_sha256": (normalized_lineage["warm_validation_sha256"]),
            "user_mapping_sha256": (normalized_lineage["user_mapping_sha256"]),
            "item_mapping_sha256": (normalized_lineage["item_mapping_sha256"]),
        }

        expected_release_fingerprint = canonical_json_sha256(release_payload)

        if normalized_lineage["release_fingerprint"] != expected_release_fingerprint:
            raise RuntimeError(
                "Serving release fingerprint does not match "
                "its processed-data lineage payload."
            )

        expected_release_lineage_id = f"release-{expected_release_fingerprint[:20]}"

        if normalized_lineage["release_lineage_id"] != expected_release_lineage_id:
            raise RuntimeError(
                "Serving release_lineage_id does not match "
                "the processed-data release fingerprint."
            )

        release = _require_dict(
            manifest.get("release"),
            "serving_manifest.release",
        )

        expected_release_fields = {
            "release_lineage_id": (normalized_lineage["release_lineage_id"]),
            "release_fingerprint": (normalized_lineage["release_fingerprint"]),
            "data_run_id": (normalized_lineage["data_run_id"]),
            "dataset_fingerprint": (normalized_lineage["dataset_fingerprint"]),
        }

        for key, expected_value in expected_release_fields.items():
            if release.get(key) != expected_value:
                raise RuntimeError(
                    f"Serving release metadata does not match lineage field '{key}'."
                )

        model_release_id = _require_string(
            release,
            "model_release_id",
            "serving_manifest.release",
        )

        model_release_fingerprint = validate_sha256(
            _require_string(
                release,
                "model_release_fingerprint",
                "serving_manifest.release",
            ),
            ("serving_manifest.release.model_release_fingerprint"),
        )

        model_release_manifest_sha256 = validate_sha256(
            _require_string(
                release,
                "model_release_manifest_sha256",
                "serving_manifest.release",
            ),
            ("serving_manifest.release.model_release_manifest_sha256"),
        )

        model_release = resolve_pinned_model_release(
            model_release_id=model_release_id,
            model_release_fingerprint=(model_release_fingerprint),
            expected_manifest_sha256=(model_release_manifest_sha256),
        )

        if model_release.lineage != normalized_lineage:
            mismatched_keys = sorted(
                key
                for key in (set(model_release.lineage) | set(normalized_lineage))
                if (model_release.lineage.get(key) != normalized_lineage.get(key))
            )

            raise RuntimeError(
                "Pinned model release lineage does not match "
                "the serving bundle. Mismatched fields: "
                f"{mismatched_keys}"
            )

        catalog = _require_dict(
            manifest.get("catalog"),
            "serving_manifest.catalog",
        )

        if catalog.get("cold_item_index") != COLD_ITEM_INDEX:
            raise RuntimeError(
                "Serving manifest cold-item index does not match project configuration."
            )

        if (
            _require_string(
                catalog,
                "user_mapping_sha256",
                "serving_manifest.catalog",
            )
            != normalized_lineage["user_mapping_sha256"]
        ):
            raise RuntimeError(
                "Serving catalog user mapping fingerprint "
                "does not match release lineage."
            )

        if (
            _require_string(
                catalog,
                "item_mapping_sha256",
                "serving_manifest.catalog",
            )
            != normalized_lineage["item_mapping_sha256"]
        ):
            raise RuntimeError(
                "Serving catalog item mapping fingerprint "
                "does not match release lineage."
            )

        training_history = _require_dict(
            manifest.get("training_history"),
            "serving_manifest.training_history",
        )

        if training_history.get("source") != FINAL_TRAINING_PROTOCOL:
            raise RuntimeError(
                "Serving manifest training history does not "
                "use the expected final-training protocol."
            )

        model_artifacts = _require_dict(
            manifest.get("model_artifacts"),
            "serving_manifest.model_artifacts",
        )

        expected_model_artifacts = {
            "two_tower_checkpoint_sha256": (model_release.two_tower_model_path),
            "neumf_checkpoint_sha256": (model_release.neumf_model_path),
            "two_tower_item_embeddings_sha256": (model_release.item_embeddings_path),
            "two_tower_embedding_metadata_sha256": (
                model_release.embedding_metadata_path
            ),
            "model_release_manifest_sha256": (model_release.manifest_path),
        }

        validated_model_hashes: dict[str, str] = {}

        for key, artifact_path in expected_model_artifacts.items():
            expected_hash = validate_sha256(
                _require_string(
                    model_artifacts,
                    key,
                    "serving_manifest.model_artifacts",
                ),
                (f"serving_manifest.model_artifacts.{key}"),
            )

            actual_hash = sha256_file(artifact_path)

            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Serving model artifact fingerprint mismatch for '{key}'."
                )

            validated_model_hashes[key] = actual_hash

        if (
            validated_model_hashes["model_release_manifest_sha256"]
            != model_release_manifest_sha256
        ):
            raise RuntimeError(
                "Pinned model release manifest fingerprint differs "
                "between serving manifest sections."
            )

        embedding_binding = _require_dict(
            manifest.get("embedding_binding"),
            "serving_manifest.embedding_binding",
        )

        if (
            _require_string(
                embedding_binding,
                "parent_checkpoint_sha256",
                "serving_manifest.embedding_binding",
            )
            != validated_model_hashes["two_tower_checkpoint_sha256"]
        ):
            raise RuntimeError(
                "Serving embedding parent checkpoint binding "
                "does not match the Two-Tower checkpoint."
            )

        if (
            _require_string(
                embedding_binding,
                "item_embeddings_sha256",
                "serving_manifest.embedding_binding",
            )
            != validated_model_hashes["two_tower_item_embeddings_sha256"]
        ):
            raise RuntimeError(
                "Serving item-embedding binding does not match "
                "the declared embedding artifact."
            )

        self.serving_release = serving_release
        self.model_release = model_release
        self.manifest = manifest
        self.lineage = normalized_lineage
        self.manifest_validated = True

    def _load_retrieval_model(
        self,
    ) -> None:
        """Load the Two-Tower checkpoint once and validate lineage."""

        if self.lineage is None:
            raise RuntimeError("Serving lineage is unavailable.")

        if self.model_release is None:
            raise RuntimeError("Model release is unavailable.")

        checkpoint = load_model_checkpoint(
            path=self.model_release.two_tower_model_path,
            device="cpu",
        )

        config = _require_dict(
            checkpoint.get("model_config"),
            "Two-Tower.model_config",
        )

        required_config = {
            "n_users",
            "n_items",
            "embedding_dim",
            "hidden_dim",
        }

        missing_config = required_config - set(config)

        if missing_config:
            raise ValueError(
                "Two-Tower checkpoint is missing "
                "configuration fields: "
                f"{sorted(missing_config)}"
            )

        training_config = _require_dict(
            checkpoint.get("training_config"),
            "Two-Tower.training_config",
        )

        if training_config.get("training_data") != FINAL_TRAINING_PROTOCOL:
            raise RuntimeError(
                "Two-Tower checkpoint was not trained "
                "using the expected final-training protocol."
            )

        checkpoint_lineage = _require_dict(
            training_config.get("lineage"),
            "Two-Tower.training_config.lineage",
        )

        if checkpoint_lineage != self.lineage:
            mismatched_keys = sorted(
                key
                for key in (set(checkpoint_lineage) | set(self.lineage))
                if checkpoint_lineage.get(key) != self.lineage.get(key)
            )

            raise RuntimeError(
                "Two-Tower checkpoint lineage does not match "
                "the serving release. Mismatched fields: "
                f"{mismatched_keys}"
            )

        model = TwoTowerRecommender(
            n_users=int(config["n_users"]),
            n_items=int(config["n_items"]),
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
        )

        self.retrieval_model = restore_model_from_checkpoint(
            model=model,
            checkpoint=checkpoint,
            device=self.device,
        )

        self.n_users = int(config["n_users"])

        self.n_items = int(config["n_items"])

    def _load_ranker(
        self,
    ) -> None:
        """Load the NeuMF checkpoint once and validate lineage."""

        if self.lineage is None:
            raise RuntimeError("Serving lineage is unavailable.")

        if self.model_release is None:
            raise RuntimeError("Model release is unavailable.")

        checkpoint = load_model_checkpoint(
            path=self.model_release.neumf_model_path,
            device="cpu",
        )

        config = _require_dict(
            checkpoint.get("model_config"),
            "NeuMF.model_config",
        )

        required_config = {
            "n_users",
            "n_items",
            "embedding_dim",
            "hidden_layers",
            "item_score_batch_size",
        }

        missing_config = required_config - set(config)

        if missing_config:
            raise ValueError(
                "NeuMF checkpoint is missing "
                "configuration fields: "
                f"{sorted(missing_config)}"
            )

        ranker_n_users = int(config["n_users"])

        ranker_n_items = int(config["n_items"])

        if ranker_n_users != self.n_users:
            raise RuntimeError("NeuMF and Two-Tower user vocabularies do not match.")

        if ranker_n_items != self.n_items:
            raise RuntimeError("NeuMF and Two-Tower item vocabularies do not match.")

        training_config = _require_dict(
            checkpoint.get("training_config"),
            "NeuMF.training_config",
        )

        if training_config.get("training_data") != FINAL_TRAINING_PROTOCOL:
            raise RuntimeError(
                "NeuMF checkpoint was not trained using "
                "the expected final-training protocol."
            )

        checkpoint_lineage = _require_dict(
            training_config.get("lineage"),
            "NeuMF.training_config.lineage",
        )

        if checkpoint_lineage != self.lineage:
            mismatched_keys = sorted(
                key
                for key in (set(checkpoint_lineage) | set(self.lineage))
                if checkpoint_lineage.get(key) != self.lineage.get(key)
            )

            raise RuntimeError(
                "NeuMF checkpoint lineage does not match "
                "the serving release. Mismatched fields: "
                f"{mismatched_keys}"
            )

        model = NeuMF(
            n_users=ranker_n_users,
            n_items=ranker_n_items,
            embedding_dim=int(config["embedding_dim"]),
            hidden_layers=tuple(config["hidden_layers"]),
            item_score_batch_size=int(config["item_score_batch_size"]),
        )

        self.ranker = restore_model_from_checkpoint(
            model=model,
            checkpoint=checkpoint,
            device=self.device,
        )

    def _load_embeddings(
        self,
    ) -> None:
        """
        Load Two-Tower item embeddings and validate their exact
        parent-checkpoint and release-lineage binding.
        """

        if self.lineage is None:
            raise RuntimeError("Serving lineage is unavailable.")

        if self.model_release is None:
            raise RuntimeError("Model release is unavailable.")

        embedding_metadata = load_json(self.model_release.embedding_metadata_path)

        if embedding_metadata.get("metadata_version") != 1:
            raise RuntimeError("Unsupported Two-Tower embedding metadata version.")

        if embedding_metadata.get("model") != "TwoTowerRecommender":
            raise RuntimeError(
                "Two-Tower embedding metadata contains an unexpected model identifier."
            )

        metadata_lineage = _require_dict(
            embedding_metadata.get("lineage"),
            "embedding_metadata.lineage",
        )

        if metadata_lineage != self.lineage:
            mismatched_keys = sorted(
                key
                for key in (set(metadata_lineage) | set(self.lineage))
                if metadata_lineage.get(key) != self.lineage.get(key)
            )

            raise RuntimeError(
                "Two-Tower embedding lineage does not match "
                "the serving release. Mismatched fields: "
                f"{mismatched_keys}"
            )

        parent_checkpoint = _require_dict(
            embedding_metadata.get("parent_checkpoint"),
            "embedding_metadata.parent_checkpoint",
        )

        parent_checkpoint_sha256 = validate_sha256(
            _require_string(
                parent_checkpoint,
                "sha256",
                "embedding_metadata.parent_checkpoint",
            ),
            "embedding_metadata.parent_checkpoint.sha256",
        )

        actual_checkpoint_sha256 = sha256_file(self.model_release.two_tower_model_path)

        if parent_checkpoint_sha256 != actual_checkpoint_sha256:
            raise RuntimeError(
                "Two-Tower embeddings are not bound to the loaded Two-Tower checkpoint."
            )

        item_metadata = _require_dict(
            embedding_metadata.get("item_embeddings"),
            "embedding_metadata.item_embeddings",
        )

        item_embedding_sha256 = validate_sha256(
            _require_string(
                item_metadata,
                "sha256",
                "embedding_metadata.item_embeddings",
            ),
            "embedding_metadata.item_embeddings.sha256",
        )

        actual_item_embedding_sha256 = sha256_file(
            self.model_release.item_embeddings_path
        )

        if item_embedding_sha256 != actual_item_embedding_sha256:
            raise RuntimeError(
                "Two-Tower item embeddings do not match their metadata fingerprint."
            )

        item_embeddings = (
            load_tensor(
                self.model_release.item_embeddings_path,
                device=self.device,
            )
            .to(
                dtype=torch.float32,
            )
            .contiguous()
        )

        if item_embeddings.ndim != 2:
            raise RuntimeError(
                "Item embeddings must have shape [n_items, embedding_dim]."
            )

        if item_embeddings.shape[0] != self.n_items:
            raise RuntimeError(
                "Item embedding count does not match "
                "the model item vocabulary. "
                f"embeddings={item_embeddings.shape[0]}, "
                f"items={self.n_items}."
            )

        expected_shape = _require_list(
            item_metadata,
            "shape",
            "embedding_metadata.item_embeddings",
        )

        if expected_shape != list(item_embeddings.shape):
            raise RuntimeError(
                "Two-Tower item embedding shape does not match its metadata."
            )

        if not torch.isfinite(item_embeddings).all():
            raise RuntimeError("Two-Tower item embeddings contain non-finite values.")

        self.item_embeddings = item_embeddings

    def _load_mappings_and_metadata(
        self,
    ) -> None:
        """
        Load serving mappings and verify their semantic identity
        against the release lineage.
        """

        if self.manifest is None:
            raise RuntimeError("Serving manifest is unavailable.")

        if self.lineage is None:
            raise RuntimeError("Serving lineage is unavailable.")

        if self.serving_release is None:
            raise RuntimeError("Serving release is unavailable.")

        user_mapping = load_parquet(self.serving_release.user_mapping_path).sort(
            "user_idx"
        )

        movies = load_parquet(self.serving_release.movie_metadata_path).sort("item_idx")

        required_user_columns = {
            "userId",
            "user_idx",
        }

        missing_user_columns = required_user_columns - set(user_mapping.columns)

        if missing_user_columns:
            raise ValueError(
                "Serving user mapping is missing columns: "
                f"{sorted(missing_user_columns)}"
            )

        if user_mapping.height != self.n_users:
            raise RuntimeError(
                "Serving user mapping size does not match the model user vocabulary."
            )

        if user_mapping["userId"].null_count() > 0:
            raise ValueError("Serving user mapping contains null userId values.")

        if user_mapping["user_idx"].null_count() > 0:
            raise ValueError("Serving user mapping contains null user_idx values.")

        if user_mapping["userId"].n_unique() != self.n_users:
            raise ValueError("Serving user mapping contains duplicate userId values.")

        if user_mapping["user_idx"].n_unique() != self.n_users:
            raise ValueError("Serving user mapping contains duplicate user_idx values.")

        user_indices = (
            user_mapping["user_idx"]
            .to_numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )

        expected_user_indices = np.arange(
            self.n_users,
            dtype=np.int64,
        )

        if not np.array_equal(
            user_indices,
            expected_user_indices,
        ):
            raise RuntimeError("Serving user mapping is not contiguous and zero-based.")

        actual_user_mapping_sha256 = canonical_id_mapping_sha256(
            mapping=user_mapping,
            raw_id_col="userId",
            index_col="user_idx",
        )

        if actual_user_mapping_sha256 != self.lineage["user_mapping_sha256"]:
            raise RuntimeError(
                "Serving user mapping semantic fingerprint "
                "does not match release lineage."
            )

        self.user_id_to_idx = {
            int(user_id): int(user_idx)
            for (
                user_id,
                user_idx,
            ) in zip(
                user_mapping["userId"].to_list(),
                user_mapping["user_idx"].to_list(),
                strict=True,
            )
        }

        required_movie_columns = {
            "item_idx",
            "movieId",
            "title",
            "genres",
        }

        missing_movie_columns = required_movie_columns - set(movies.columns)

        if missing_movie_columns:
            raise ValueError(
                "Serving movie metadata is missing columns: "
                f"{sorted(missing_movie_columns)}"
            )

        if movies.height != self.n_items:
            raise RuntimeError(
                "Movie metadata size does not match the model item vocabulary."
            )

        for column in (
            "item_idx",
            "movieId",
            "title",
            "genres",
        ):
            if movies[column].null_count() > 0:
                raise ValueError(f"Movie metadata contains null values in '{column}'.")

        if movies["item_idx"].n_unique() != self.n_items:
            raise ValueError("Movie metadata contains duplicate item_idx values.")

        if movies["movieId"].n_unique() != self.n_items:
            raise ValueError("Movie metadata contains duplicate movieId values.")

        movie_indices = (
            movies["item_idx"]
            .to_numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )

        expected_movie_indices = np.arange(
            self.n_items,
            dtype=np.int64,
        )

        if not np.array_equal(
            movie_indices,
            expected_movie_indices,
        ):
            raise RuntimeError(
                "Movie metadata item indices are not contiguous and zero-based."
            )

        # Because the serving metadata contains both movieId and
        # item_idx, the exact training item mapping can be
        # reconstructed semantically at startup.
        actual_item_mapping_sha256 = canonical_id_mapping_sha256(
            mapping=movies,
            raw_id_col="movieId",
            index_col="item_idx",
        )

        if actual_item_mapping_sha256 != self.lineage["item_mapping_sha256"]:
            raise RuntimeError(
                "Serving movie metadata does not preserve "
                "the release item_idx -> movieId identity."
            )

        catalog = _require_dict(
            self.manifest.get("catalog"),
            "serving_manifest.catalog",
        )

        if catalog.get("user_mapping_sha256") != actual_user_mapping_sha256:
            raise RuntimeError(
                "Serving catalog user mapping fingerprint does "
                "not match the loaded user mapping."
            )

        if catalog.get("item_mapping_sha256") != actual_item_mapping_sha256:
            raise RuntimeError(
                "Serving catalog item mapping fingerprint does "
                "not match the loaded movie metadata."
            )

        self.movie_ids = movies["movieId"].to_numpy()

        self.movie_titles = movies["title"].to_numpy()

        self.movie_genres = movies["genres"].to_numpy()

    def _load_seen_items(
        self,
    ) -> None:
        """
        Load memory-mapped seen-item CSR arrays.

        Arrays remain memory mapped. They are copied only when
        individual slices need to be converted into PyTorch tensors.
        """

        if self.serving_release is None:
            raise RuntimeError("Serving release is unavailable.")

        indptr = np.load(
            self.serving_release.seen_indptr_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        indices = np.load(
            self.serving_release.seen_indices_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if indptr.ndim != 1:
            raise RuntimeError("Seen-item indptr must be a 1D array.")

        if indices.ndim != 1:
            raise RuntimeError("Seen-item indices must be a 1D array.")

        if not np.issubdtype(
            indptr.dtype,
            np.integer,
        ):
            raise TypeError("Seen-item indptr must contain integers.")

        if not np.issubdtype(
            indices.dtype,
            np.integer,
        ):
            raise TypeError("Seen-item indices must contain integers.")

        if indptr.size != self.n_users + 1:
            raise RuntimeError(
                "Seen-item CSR indptr size does not match the user vocabulary."
            )

        if int(indptr[0]) != 0:
            raise RuntimeError("Seen-item CSR indptr must start at zero.")

        if np.any(indptr[1:] < indptr[:-1]):
            raise RuntimeError(
                "Seen-item CSR indptr must be monotonically non-decreasing."
            )

        if int(indptr[-1]) != indices.size:
            raise RuntimeError("Seen-item CSR indptr and indices are inconsistent.")

        if indices.size > 0:
            minimum_item = int(indices.min())

            maximum_item = int(indices.max())

            if minimum_item < 0:
                raise RuntimeError("Seen-item CSR contains negative item indices.")

            if maximum_item >= self.n_items:
                raise RuntimeError(
                    "Seen-item CSR contains item indices outside the model catalog."
                )

        self.seen_items = SeenItemsCSR(
            indptr=indptr,
            indices=indices,
        )

    def _validate_serving_bundle(
        self,
    ) -> None:
        """Validate cross-artifact dimensions and runtime invariants."""

        if not self.manifest_validated:
            raise RuntimeError("Serving manifest was not validated.")

        if self.manifest is None:
            raise RuntimeError("Serving manifest is unavailable.")

        if self.lineage is None:
            raise RuntimeError("Serving lineage is unavailable.")

        if self.model_release is None:
            raise RuntimeError("Model release is unavailable.")

        if self.serving_release is None:
            raise RuntimeError("Serving release is unavailable.")

        catalog = _require_dict(
            self.manifest.get("catalog"),
            "serving_manifest.catalog",
        )

        manifest_n_users = catalog.get("n_users")

        manifest_n_items = catalog.get("n_items")

        if manifest_n_users != self.n_users:
            raise RuntimeError("Manifest user count does not match the loaded model.")

        if manifest_n_items != self.n_items:
            raise RuntimeError("Manifest item count does not match the loaded model.")

        model_release_catalog = _require_dict(
            self.model_release.manifest.get("catalog"),
            "model_release_manifest.catalog",
        )

        if (
            _require_int(
                model_release_catalog,
                "n_users",
                "model_release_manifest.catalog",
            )
            != self.n_users
        ):
            raise RuntimeError(
                "Model release user count does not match the loaded model."
            )

        if (
            _require_int(
                model_release_catalog,
                "n_items",
                "model_release_manifest.catalog",
            )
            != self.n_items
        ):
            raise RuntimeError(
                "Model release item count does not match the loaded model."
            )

        if self.n_users <= 0:
            raise RuntimeError("Model user vocabulary is empty.")

        if self.n_items <= 0:
            raise RuntimeError("Model item vocabulary is empty.")

        if RANKING_CANDIDATE_SIZE <= 0:
            raise RuntimeError("RANKING_CANDIDATE_SIZE must be > 0.")

        if self.n_items < RANKING_CANDIDATE_SIZE:
            raise RuntimeError(
                "RANKING_CANDIDATE_SIZE cannot exceed the item catalog size."
            )

        if API_MAX_K <= 0:
            raise RuntimeError("API_MAX_K must be > 0.")

        if API_MAX_K > RANKING_CANDIDATE_SIZE:
            raise RuntimeError("API_MAX_K cannot exceed RANKING_CANDIDATE_SIZE.")

        if RANKING_SCORE_BATCH_SIZE <= 0:
            raise RuntimeError("RANKING_SCORE_BATCH_SIZE must be > 0.")

        if self.retrieval_model is None:
            raise RuntimeError("Retrieval model is not loaded.")

        if self.ranker is None:
            raise RuntimeError("Ranker is not loaded.")

        if self.item_embeddings is None:
            raise RuntimeError("Item embeddings are not loaded.")

        if self.seen_items is None:
            raise RuntimeError("Seen-item history is not loaded.")

        if self.movie_ids is None:
            raise RuntimeError("Movie IDs are not loaded.")

        if self.movie_titles is None:
            raise RuntimeError("Movie titles are not loaded.")

        if self.movie_genres is None:
            raise RuntimeError("Movie genres are not loaded.")

        training_history = _require_dict(
            self.manifest.get("training_history"),
            "serving_manifest.training_history",
        )

        if training_history.get("source") != FINAL_TRAINING_PROTOCOL:
            raise RuntimeError(
                "Serving training history uses an unexpected training protocol."
            )

        expected_seen_pairs = _require_int(
            training_history,
            "unique_seen_user_item_pairs",
            "serving_manifest.training_history",
        )

        if expected_seen_pairs != self.seen_items.indices.size:
            raise RuntimeError(
                "Serving seen-history pair count does not match the manifest."
            )

        probe_user = torch.tensor(
            [0],
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            probe_embedding = self.retrieval_model.encode_users(probe_user)

        if probe_embedding.ndim != 2 or probe_embedding.shape[0] != 1:
            raise RuntimeError(
                "Two-Tower user encoder returned an invalid embedding shape."
            )

        if probe_embedding.shape[1] != self.item_embeddings.shape[1]:
            raise RuntimeError(
                "Two-Tower user embedding dimension "
                "does not match persisted item embeddings."
            )

    def _require_serving_artifacts(
        self,
    ) -> None:
        """Ensure runtime artifacts required for recommendation exist."""

        if (
            self.retrieval_model is None
            or self.ranker is None
            or self.item_embeddings is None
            or self.movie_ids is None
            or self.movie_titles is None
            or self.movie_genres is None
            or self.seen_items is None
        ):
            raise RuntimeError("Serving artifacts are incomplete.")

    def _validate_k(
        self,
        k: int,
    ) -> None:
        """Validate requested recommendation count."""

        if not isinstance(
            k,
            int,
        ):
            raise TypeError("k must be an integer.")

        if k < 1 or k > API_MAX_K:
            raise ValueError(f"k must be between 1 and {API_MAX_K}.")

        if k > RANKING_CANDIDATE_SIZE:
            raise ValueError("k cannot exceed the ranking candidate size.")

    def _synchronize_device(
        self,
    ) -> None:
        """Synchronize asynchronous accelerator work."""

        if self.device.type == "mps":
            torch.mps.synchronize()

        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    def resolve_user_idx(
        self,
        user_id: int,
    ) -> int:
        """Map MovieLens userId to model user_idx."""

        user_idx = self.user_id_to_idx.get(user_id)

        if user_idx is None:
            raise KeyError(f"Unknown user_id: {user_id}")

        return user_idx

    @torch.inference_mode()
    def _score_user_indices(
        self,
        user_indices: np.ndarray,
        k: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Run batched retrieval and reranking for encoded users.

        Seen items remain invalid throughout both retrieval and
        reranking. If a user has fewer than ``k`` unseen catalog
        items, recommendation generation fails explicitly rather
        than returning a previously seen item.
        """

        self._require_serving_artifacts()
        self._validate_k(k)

        user_indices = np.asarray(
            user_indices,
            dtype=np.int64,
        )

        if user_indices.ndim != 1:
            raise ValueError("user_indices must be a 1D array.")

        if user_indices.size == 0:
            raise ValueError("user_indices cannot be empty.")

        if np.any(user_indices < 0):
            raise ValueError("user_indices cannot be negative.")

        if np.any(user_indices >= self.n_users):
            raise ValueError(
                "user_indices contain values outside the model user vocabulary."
            )

        assert self.retrieval_model is not None
        assert self.ranker is not None
        assert self.item_embeddings is not None
        assert self.seen_items is not None

        user_tensor = torch.as_tensor(
            user_indices,
            dtype=torch.long,
            device=self.device,
        )

        user_embeddings = self.retrieval_model.encode_users(user_tensor)

        retrieval_scores = user_embeddings @ self.item_embeddings.T

        validate_finite_scores(
            scores=retrieval_scores,
            context="Raw retrieval scores",
        )

        # Reuse the evaluation masking implementation.
        #
        # mask_seen_items() copies mmap-backed slices before
        # converting them into torch tensors, so read-only mmap
        # arrays remain safe.
        mask_seen_items(
            scores=retrieval_scores,
            user_ids=user_indices,
            seen_items=self.seen_items,
        )

        if (
            torch.isnan(retrieval_scores).any()
            or torch.isposinf(retrieval_scores).any()
        ):
            raise RuntimeError("Retrieval scores contain NaN or positive infinity.")

        (
            candidate_retrieval_scores,
            candidate_items,
        ) = torch.topk(
            retrieval_scores,
            k=RANKING_CANDIDATE_SIZE,
            dim=1,
        )

        # A fixed-width topk can contain indices whose score is
        # -inf when a dense-history user has fewer unseen items
        # than RANKING_CANDIDATE_SIZE. Preserve that validity
        # information through reranking so seen items can never
        # be reintroduced.
        candidate_valid_mask = ~torch.isneginf(candidate_retrieval_scores)

        valid_candidate_counts = candidate_valid_mask.sum(dim=1)

        insufficient_rows = torch.nonzero(
            valid_candidate_counts < k,
            as_tuple=False,
        ).flatten()

        if insufficient_rows.numel() > 0:
            row = int(insufficient_rows[0].item())

            user_idx = int(user_indices[row])

            available = int(valid_candidate_counts[row].item())

            raise ValueError(
                f"Cannot return {k} unseen recommendations "
                f"for user_idx={user_idx}; only "
                f"{available} unseen candidates are available."
            )

        candidate_scores = score_candidate_pairs(
            model=self.ranker,
            user_ids=user_tensor,
            candidate_items=candidate_items,
            score_batch_size=RANKING_SCORE_BATCH_SIZE,
        )

        valid_ranker_scores = candidate_scores[candidate_valid_mask]

        if not torch.isfinite(valid_ranker_scores).all():
            raise RuntimeError(
                "Ranker produced non-finite scores for valid recommendation candidates."
            )

        # Keep retrieval-invalid candidates invalid after NeuMF
        # scoring. Without this step a seen item selected as a
        # -inf retrieval placeholder could receive a finite ranker
        # score and become a final recommendation.
        candidate_scores = candidate_scores.masked_fill(
            ~candidate_valid_mask,
            -torch.inf,
        )

        (
            top_score_values,
            top_positions,
        ) = torch.topk(
            candidate_scores,
            k=k,
            dim=1,
        )

        if torch.isneginf(top_score_values).any():
            raise RuntimeError(
                "Final recommendation selection contains an invalid masked candidate."
            )

        top_items = (
            candidate_items.gather(
                dim=1,
                index=top_positions,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )

        top_scores = top_score_values.detach().cpu().numpy()

        return (
            top_items,
            top_scores,
        )

    def _build_recommendations(
        self,
        item_indices: np.ndarray,
        scores: np.ndarray,
    ) -> list[Recommendation]:
        """Convert model item indices into API recommendation objects."""

        if (
            self.movie_ids is None
            or self.movie_titles is None
            or self.movie_genres is None
        ):
            raise RuntimeError("Movie metadata is not loaded.")

        if item_indices.shape != scores.shape:
            raise RuntimeError("Recommendation item and score shapes do not match.")

        recommendations: list[Recommendation] = []

        for (
            item_idx,
            score,
        ) in zip(
            item_indices,
            scores,
            strict=True,
        ):
            item_idx = int(item_idx)

            if item_idx < 0 or item_idx >= self.n_items:
                raise RuntimeError(
                    "Model returned an item index outside the serving catalog."
                )

            recommendations.append(
                Recommendation(
                    movie_id=int(self.movie_ids[item_idx]),
                    title=str(self.movie_titles[item_idx]),
                    genres=str(self.movie_genres[item_idx]),
                    score=round(
                        float(score),
                        3,
                    ),
                )
            )

        return recommendations

    @torch.inference_mode()
    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> tuple[
        list[Recommendation],
        float,
    ]:
        """Generate Top-K recommendations for one known user."""

        if not self.loaded:
            raise RuntimeError("Recommendation service is not loaded.")

        self._validate_k(k)

        started = time.perf_counter()

        user_idx = self.resolve_user_idx(user_id)

        (
            top_items,
            top_scores,
        ) = self._score_user_indices(
            user_indices=np.asarray(
                [
                    user_idx,
                ],
                dtype=np.int64,
            ),
            k=k,
        )

        recommendations = self._build_recommendations(
            item_indices=(top_items[0]),
            scores=(top_scores[0]),
        )

        self._synchronize_device()

        latency_ms = (time.perf_counter() - started) * 1000

        return (
            recommendations,
            round(
                latency_ms,
                3,
            ),
        )

    @torch.inference_mode()
    def recommend_batch(
        self,
        user_ids: list[int],
        k: int = 10,
    ) -> tuple[
        list[list[Recommendation]],
        float,
    ]:
        """
        Generate recommendations for multiple users
        using one batched model pass.

        Returned latency is total batch latency.
        """

        if not self.loaded:
            raise RuntimeError("Recommendation service is not loaded.")

        self._validate_k(k)

        if not user_ids:
            raise ValueError("user_ids cannot be empty.")

        started = time.perf_counter()

        user_indices = np.asarray(
            [self.resolve_user_idx(user_id) for user_id in user_ids],
            dtype=np.int64,
        )

        (
            top_items,
            top_scores,
        ) = self._score_user_indices(
            user_indices=(user_indices),
            k=k,
        )

        results = [
            self._build_recommendations(
                item_indices=(row_items),
                scores=(row_scores),
            )
            for (
                row_items,
                row_scores,
            ) in zip(
                top_items,
                top_scores,
                strict=True,
            )
        ]

        self._synchronize_device()

        latency_ms = (time.perf_counter() - started) * 1000

        return (
            results,
            round(
                latency_ms,
                3,
            ),
        )

    def model_info(
        self,
    ) -> dict[str, object]:
        """Return serving-model metadata."""

        bundle_version: object | None = None

        if self.manifest is not None:
            bundle_version = self.manifest.get("serving_bundle_version")

        return {
            "loaded": (self.loaded),
            "manifest_validated": (self.manifest_validated),
            "serving_bundle_version": (bundle_version),
            "release_lineage_id": (
                self.lineage.get("release_lineage_id")
                if self.lineage is not None
                else None
            ),
            "data_run_id": (
                self.lineage.get("data_run_id") if self.lineage is not None else None
            ),
            "model_release_id": (
                self.model_release.model_release_id
                if self.model_release is not None
                else None
            ),
            "model_release_fingerprint": (
                self.model_release.model_release_fingerprint
                if self.model_release is not None
                else None
            ),
            "serving_release_id": (
                self.serving_release.serving_release_id
                if self.serving_release is not None
                else None
            ),
            "serving_release_fingerprint": (
                self.serving_release.serving_release_fingerprint
                if self.serving_release is not None
                else None
            ),
            "device": str(self.device),
            "users": (self.n_users),
            "items": (self.n_items),
            "candidate_size": (RANKING_CANDIDATE_SIZE),
            "max_k": (API_MAX_K),
            "retrieval_model": ("TwoTowerRecommender"),
            "ranking_model": ("NeuMF"),
        }

    @torch.inference_mode()
    def warmup(
        self,
    ) -> None:
        """Warm inference kernels before accepting requests."""

        self._require_serving_artifacts()

        if self.n_users <= 0:
            raise RuntimeError("Cannot warm an empty user vocabulary.")

        _ = self._score_user_indices(
            user_indices=np.asarray(
                [
                    0,
                ],
                dtype=np.int64,
            ),
            k=1,
        )

        self._synchronize_device()
