import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(
    os.environ.get(
        "RECSYS_PROJECT_ROOT",
        DEFAULT_PROJECT_ROOT,
    )
).resolve()


# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-32m"

PROCESSED_DATA_DIR = PROJECT_ROOT / "artifacts" / "processed"

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

EMBEDDING_DIR = PROJECT_ROOT / "artifacts" / "embeddings"

SERVING_DIR = PROJECT_ROOT / "artifacts" / "serving"


# ---------------------------------------------------------------------------
# Raw MovieLens files
# ---------------------------------------------------------------------------

RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"

MOVIES_PATH = RAW_DATA_DIR / "movies.csv"


# ---------------------------------------------------------------------------
# Processed dataset artifacts
# ---------------------------------------------------------------------------

TRAIN_PATH = PROCESSED_DATA_DIR / "train.parquet"

VALIDATION_PATH = PROCESSED_DATA_DIR / "validation.parquet"

TEST_PATH = PROCESSED_DATA_DIR / "test.parquet"

USER_MAPPING_PATH = PROCESSED_DATA_DIR / "user_mapping.parquet"

ITEM_MAPPING_PATH = PROCESSED_DATA_DIR / "item_mapping.parquet"

MOVIE_METADATA_PATH = PROCESSED_DATA_DIR / "movies.parquet"

PREPROCESSING_CONFIG_PATH = PROCESSED_DATA_DIR / "preprocessing_config.json"

ARTIFACT_MANIFEST_PATH = PROCESSED_DATA_DIR / "artifact_manifest.json"


# ---------------------------------------------------------------------------
# Artifact / protocol versions
# ---------------------------------------------------------------------------

PROCESSED_ARTIFACT_MANIFEST_VERSION = 1

PREPROCESSING_PROTOCOL_VERSION = "strict_temporal_leave_two_out_train_only_kcore_v1"

FINAL_TRAINING_PROTOCOL = "train+warm_validation"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

POSITIVE_RATING_THRESHOLD = 4.0

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 10

COLD_ITEM_INDEX = -1


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

TRAIN_BATCH_SIZE = 32_768
EVAL_BATCH_SIZE = 256

EVAL_KS = (
    10,
    50,
)


# ---------------------------------------------------------------------------
# Matrix Factorization
# ---------------------------------------------------------------------------

MF_EMBEDDING_DIM = 64

MF_LEARNING_RATE = 1e-3
MF_WEIGHT_DECAY = 1e-6

MF_EPOCHS = 20


# ---------------------------------------------------------------------------
# Bayesian Personalized Ranking
# ---------------------------------------------------------------------------

BPR_EMBEDDING_DIM = 64

BPR_LEARNING_RATE = 1e-3
BPR_WEIGHT_DECAY = 1e-6

BPR_EPOCHS = 20


# ---------------------------------------------------------------------------
# Neural Collaborative Filtering
# ---------------------------------------------------------------------------

NCF_EMBEDDING_DIM = 64

NCF_MLP_LAYERS = (
    128,
    64,
    32,
)

NCF_LEARNING_RATE = 1e-3
NCF_WEIGHT_DECAY = 1e-6

NCF_EPOCHS = 10

NCF_ITEM_SCORE_BATCH_SIZE = 2_048


# ---------------------------------------------------------------------------
# NeuMF
# ---------------------------------------------------------------------------

NEUMF_LEARNING_RATE = 5e-4
NEUMF_WEIGHT_DECAY = 1e-6

NEUMF_EPOCHS = 5

NEUMF_PRETRAIN_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Two-Tower retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_EMBEDDING_DIM = 64
RETRIEVAL_HIDDEN_DIM = 128

RETRIEVAL_LEARNING_RATE = 1e-3
RETRIEVAL_WEIGHT_DECAY = 1e-6

RETRIEVAL_EPOCHS = 10

RETRIEVAL_KS = (
    100,
    250,
    500,
    1_000,
)


# ---------------------------------------------------------------------------
# Retrieval + ranking
# ---------------------------------------------------------------------------

RANKING_CANDIDATE_SIZE = 500

RANKING_SCORE_BATCH_SIZE = 131_072


# ---------------------------------------------------------------------------
# Model-selection artifacts
# ---------------------------------------------------------------------------

NEUMF_MODEL_PATH = MODEL_DIR / "neumf.pt"

TWO_TOWER_RESULTS_PATH = MODEL_DIR / "two_tower_results.json"

RANKING_RESULTS_PATH = MODEL_DIR / "ranking_results.json"


# ---------------------------------------------------------------------------
# Final-model artifacts
# ---------------------------------------------------------------------------

FINAL_NEUMF_MODEL_PATH = MODEL_DIR / "final_neumf.pt"

FINAL_TWO_TOWER_MODEL_PATH = MODEL_DIR / "final_two_tower.pt"

FINAL_TWO_TOWER_USER_EMBEDDINGS_PATH = (
    EMBEDDING_DIR / "final_two_tower_user_embeddings.pt"
)

FINAL_TWO_TOWER_ITEM_EMBEDDINGS_PATH = (
    EMBEDDING_DIR / "final_two_tower_item_embeddings.pt"
)


# ---------------------------------------------------------------------------
# Scalability analysis
# ---------------------------------------------------------------------------

SCALABILITY_RESULTS_PATH = MODEL_DIR / "scalability_results.json"


# ---------------------------------------------------------------------------
# Serving artifacts
# ---------------------------------------------------------------------------

SERVING_USER_MAPPING_PATH = SERVING_DIR / "user_mapping.parquet"

SERVING_MOVIE_METADATA_PATH = SERVING_DIR / "movies.parquet"

SERVING_SEEN_INDPTR_PATH = SERVING_DIR / "seen_indptr.npy"

SERVING_SEEN_INDICES_PATH = SERVING_DIR / "seen_indices.npy"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

API_DEFAULT_K = 10
API_MAX_K = 50
