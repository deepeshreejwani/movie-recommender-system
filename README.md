# Movie Recommender System

A production-style **implicit-feedback recommendation system** built on the **MovieLens 32M** dataset.

The final system uses a two-stage retrieval-and-ranking architecture:

```text
User
  │
  ▼
Two-Tower Retrieval
  │
  ▼
Top-500 Candidates
  │
  ▼
NeuMF Ranking
  │
  ▼
Top-K Recommendations
```

The repository covers the full recommendation-system lifecycle: strict temporal preprocessing, baselines, matrix factorization, neural collaborative filtering, retrieval, reranking, offline evaluation, scalability analysis, immutable model releases, FastAPI serving, batched inference, and Docker deployment.

---

## Highlights

- **32,000,204** raw MovieLens ratings
- **15,938,231** positive interactions (`rating >= 4.0`)
- Leakage-resistant **strict temporal leave-two-out** evaluation
- **Train-only iterative k-core** filtering
- Frozen catalog mappings built from **training data only**
- Explicit handling of **cold validation/test targets**
- Baselines → MF/BPR → GMF/MLP/NeuMF → Two-Tower → Two-stage ranking
- Final architecture: **Two-Tower → Top-500 → NeuMF**
- Production models trained on **train + warm validation**
- Final held-out test evaluated **once**
- Content-addressed immutable **model releases**
- Immutable **serving releases** pinned to an exact model release
- FastAPI single-user and batch recommendation endpoints
- Dockerized CPU serving with read-only mounted artifacts

---

## Dataset

The project uses **MovieLens 32M**.

### Raw data

| Statistic | Value |
|---|---:|
| Ratings | 32,000,204 |
| Users | 200,948 |
| Rated movies | 84,432 |
| Positive threshold | `rating >= 4.0` |
| Positive interactions | 15,938,231 |

The task is formulated as **implicit-feedback Top-K recommendation**: a rating of 4.0 or higher is treated as a positive user-item interaction.

---

## Preprocessing and Evaluation Protocol

The preprocessing protocol is deliberately strict to avoid leakage from future interactions.

### 1. Temporal holdout first

For each user:

- latest timestamp group → **test**
- second-latest timestamp group → **validation**
- all earlier interactions → candidate **training history**

Users are retained only when:

- they have at least three distinct timestamps,
- the latest timestamp contains exactly one interaction,
- the second-latest timestamp contains exactly one interaction,
- temporal ordering satisfies `train < validation < test`.

Users with tied interactions at either holdout timestamp are excluded rather than resolved arbitrarily.

### 2. Train-only iterative k-core

After the temporal split, k-core filtering is applied **only to training interactions**:

- minimum user interactions: **5**
- minimum item interactions: **10**

Validation and test interactions never contribute to training eligibility.

### 3. Frozen vocabularies

User and item mappings are created from the filtered training set only.

Therefore:

- the model catalog contains only training-known items,
- a validation/test item absent from the training catalog is retained with `item_idx = -1`,
- cold holdout targets remain in evaluation and count as misses,
- `-1` is never used for model training or seen-item masking.

### Canonical processed dataset

| Split / Population | Interactions |
|---|---:|
| Training | 13,621,894 |
| Validation | 163,760 |
| Validation — warm | 162,577 |
| Validation — cold | 1,183 |
| Test | 163,760 |
| Test — warm | 162,300 |
| Test — cold | 1,460 |

| Vocabulary | Count |
|---|---:|
| Users | 163,760 |
| Movies | 17,818 |

Cold-target rates:

- validation: **0.722%**
- test: **0.892%**

---

## Model Development

The project progresses from simple baselines to a production-style retrieval-and-ranking system.

### Baselines

- Random recommender
- Popularity recommender

### Collaborative Filtering

- Pointwise Matrix Factorization
- Bayesian Personalized Ranking (BPR)

### Neural Collaborative Filtering

- GMF
- MLP
- NeuMF

### Retrieval

- Two-Tower neural retrieval model

### Ranking

- NeuMF used as the downstream candidate ranker

The test set is not used for model selection. Architecture and candidate-depth decisions are made on validation only.

Candidate-depth selection is persisted with a deterministic validation-only rule. Ranking metrics are compared at 3-decimal precision, with the smaller candidate pool preferred when downstream quality is tied.

---

## Validation Results

Results from completed validation-stage experiments (rounded to three decimal
places). The underlying reports are generated under local model and run-artifact
directories, which are excluded from version control. Retraining neural models
may yield different weights and metrics.

| Model / Pipeline | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| Popularity | 0.049 | 0.024 | 0.155 | 0.047 |
| BPR MF | 0.061 | 0.030 | 0.190 | 0.058 |
| NeuMF — full catalog | 0.075 | 0.037 | 0.243 | 0.073 |
| Two-Tower → Top-500 → NeuMF | 0.075 | 0.037 | 0.243 | 0.073 |

NeuMF improves ranking quality over classical MF, but full-catalog neural scoring is expensive. The Two-Tower stage reduces the number of items sent to NeuMF while preserving most useful candidates.

### Candidate-depth trade-off

| Candidates | Candidate Recall | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Neural Score Reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.317 | 0.074 | 0.037 | 0.232 | 0.071 | 99.439% |
| 250 | 0.508 | 0.075 | 0.037 | 0.242 | 0.073 | 98.597% |
| **500** | **0.671** | **0.075** | **0.037** | **0.243** | **0.073** | **97.194%** |
| 1,000 | 0.816 | 0.075 | 0.037 | 0.243 | 0.073 | 94.388% |

A candidate depth of **500** is selected by the validation-only candidate policy: downstream ranking metrics are compared at the project's 3-decimal reporting precision, and the smaller candidate set wins when quality is tied. At that precision, 500 and 1,000 candidates have the same downstream Top-K quality, while 500 requires half as many NeuMF candidate scores.

---

## Final Architecture

```text
                         ┌──────────────────────────────┐
                         │          Known User          │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │   Two-Tower User Encoder     │
                         └──────────────┬───────────────┘
                                        │
                              64-d user embedding
                                        │
                                        ▼
                ┌────────────────────────────────────────────┐
                │ Dot product vs precomputed item embeddings │
                └───────────────────────┬────────────────────┘
                                        │
                              Top-500 candidates
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │        NeuMF Ranker          │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                              Top-K recommendations
```

The Two-Tower model handles efficient candidate retrieval, while NeuMF performs higher-capacity nonlinear ranking over the much smaller candidate set.

---

## Final Training Protocol

Once model selection is complete, production models are trained exactly once using:

```text
training + warm validation
```

Cold validation targets are excluded because they are outside the frozen training catalog.

| Final-training component | Interactions |
|---|---:|
| Training | 13,621,894 |
| Warm validation | 162,577 |
| Cold validation excluded | 1,183 |
| **Final production training** | **13,784,471** |

The held-out test set is then evaluated once using:

- final production models,
- seen history = `train + warm validation`,
- cold test items retained as misses.

---

## Final Test Results

These results come from the completed final held-out evaluation. The canonical
local report is `artifacts/models/ranking_results.json` (not shipped with the
source repository). New model training is not guaranteed to reproduce the
exact reported values.

| Pipeline | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| Two-Tower retrieval | 0.061 | 0.030 | 0.197 | 0.059 |
| **Two-Tower + NeuMF** | **0.072** | **0.036** | **0.229** | **0.069** |

Additional final result:

- **Candidate Recall@500: 0.655**

Reranking improves the final held-out system over retrieval-only ordering while retaining the computational advantage of a small candidate pool.

---

## Scalability and Recommendation Quality

The scalability analysis measures model compute, memory, popularity concentration, catalog coverage, and novelty using the frozen canonical production release.

### Model and embedding footprint

| Artifact / Measurement | Value |
|---|---:|
| Item embedding dimension | 64 |
| Item embedding footprint | 4.350 MB |
| Complete immutable model release | 177.567 MB |
| Online model assets | 137.585 MB |

### Local model-compute benchmark

Measured on the development machine during the final scalability run:

| Stage | Single-user compute |
|---|---:|
| Two-Tower retrieval | 0.222 ms |
| NeuMF ranking | 0.863 ms |
| Estimated pipeline compute | 1.085 ms |

These are **model-compute measurements**, not end-to-end API latency. They exclude networking, serialization, container scheduling, and other serving overhead.

### Catalog coverage

| Pipeline | Top-10 Coverage | Top-50 Coverage |
|---|---:|---:|
| Two-Tower | 12.184% | 26.827% |
| Two-Tower + NeuMF | **26.316%** | **45.426%** |

### Novelty

| Pipeline | Top-10 Novelty | Top-50 Novelty |
|---|---:|---:|
| Two-Tower | 8.825 bits | 9.477 bits |
| Two-Tower + NeuMF | **9.053 bits** | **9.765 bits** |

The system remains popularity-biased, so diversity, catalog exposure, and long-tail behavior should be monitored separately from relevance metrics.

---

## Reproducibility and Artifact Lineage

The production path uses immutable, content-addressed releases.

### Repository assets and reproducibility

A clean clone contains the maintained Python source code, tests, and documented
workflows, but **not** the MovieLens dataset, processed Parquet files, trained
checkpoints, embeddings, release payloads, or generated evaluation reports.
These must be generated through the documented pipeline or provided separately.
Source code alone does not make the API ready to serve recommendations.

Model and serving releases are immutable and content-addressed. Artifact
loading validates processing lineage, checksums, frozen vocabulary identities,
and release metadata. Locally generated results belong to the artifacts and
configurations that produced them; they are not bundled in this source repo.

The published metrics in this README describe completed validation and final-test
runs. They do **not** guarantee bitwise-identical neural weights or hardware
benchmark times in a new environment. Held-out test targets are never used
for model selection.

### Model release

```text
artifacts/models/
├── current_release.json
└── releases/
    └── model-<content-fingerprint>/
        ├── final_neumf.pt
        ├── final_two_tower.pt
        ├── final_two_tower_user_embeddings.pt
        ├── final_two_tower_item_embeddings.pt
        ├── final_two_tower_embeddings_metadata.json
        └── model_release_manifest.json
```

A model release records and validates preprocessing/model-data lineage, vocabulary identity, training protocol, catalog size, and artifact fingerprints.

### Serving release

```text
artifacts/serving/
├── current_release.json
└── releases/
    └── serving-<content-fingerprint>/
        ├── user_mapping.parquet
        ├── movies.parquet
        ├── seen_indptr.npy
        ├── seen_indices.npy
        └── serving_manifest.json
```

A serving release pins the **exact immutable model release** it was built against. Runtime loading does not silently switch to a newer model release.

The API is marked ready only after the serving manifest, pinned model release, mappings, metadata, seen-history arrays, model checkpoints, embeddings, and cross-artifact fingerprints have been validated.

---

## Repository Structure

```text
movie-recommender-system/
├── app/
│   └── main.py
├── artifacts/
│   ├── embeddings/                 # Canonical payloads, gitignored
│   ├── models/                     # Canonical releases, gitignored
│   ├── processed/                  # Processed datasets, gitignored
│   ├── reproductions/              # Local experiment outputs, gitignored
│   └── serving/                    # Canonical serving releases, gitignored
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── 01_data_preparation.py
│   ├── 02_baselines_and_mf.py
│   ├── 03_neural_cf.py
│   ├── 04_two_tower_retrieval.py
│   ├── 05_ranking_and_evaluation.py
│   ├── 06_scalability_analysis.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── build_serving_artifacts.py
├── src/
│   └── recsys/
│       ├── data/
│       ├── evaluation/
│       ├── models/
│       ├── serving/
│       ├── utils/
│       └── config.py
├── tests/
├── Dockerfile
├── pyproject.toml
├── requirements-serving.txt
└── README.md
```

The six numbered Python scripts implement data inspection, baseline and
neural model development, validation-stage ranking, and scalability analysis.

---

## Running the Project

The commands below cover fresh development and use of an existing model
release. **A clean clone does not contain the raw dataset, processed datasets,
model checkpoints, embeddings, serving arrays, or generated evaluation reports.**
These must be generated or provided separately. Run all commands from the
repository root.

### 1. Install the environment

The project uses Python **3.14** and `uv`:

```bash
uv sync
```

### 2. Prepare MovieLens 32M data (fresh-build workflow only)

Download MovieLens 32M and place its raw files under:

```text
data/raw/ml-32m/
```

Run preprocessing once when producing a new canonical dataset:

```bash
uv run python scripts/preprocess.py
```

This creates the processed datasets, frozen mappings, and lineage manifests.
**Skip preprocessing when validating the already-prepared canonical artifacts.**

### 3. Inspect an existing canonical dataset

The first numbered script is read-only. It can explore raw ratings and validate
canonical processed data, or validate processed artifacts without loading all
32 million raw ratings:

```bash
uv run python scripts/01_data_preparation.py --processed-only
```

Omit `--processed-only` to include raw-data exploration.

### 4. Reproduce validation-stage development (optional training)

The numbered development scripts reuse the shared `src/recsys/`
implementations. **Scripts 02–04 train validation-stage models and write
run-specific artifacts.** They are not required to serve an existing
production release.

Each experiment requires a fresh, unused run ID. From one terminal session,
set an ID and run the validation stages in order:

```bash
RUN_ID="repro-$(date +%Y%m%d-%H%M%S)-$$"
uv run python scripts/02_baselines_and_mf.py --run-id "$RUN_ID"
uv run python scripts/03_neural_cf.py --run-id "$RUN_ID"
uv run python scripts/04_two_tower_retrieval.py --run-id "$RUN_ID"
```

Script 04 publishes an isolated, checkpoint-bound Two-Tower selection release
for this run. The stages use **validation**, not held-out test results, to
select models. Neural retraining can produce different weights and metrics;
identical configurations do not guarantee bitwise-identical results.

### 5. Prepare canonical production releases (fresh-build workflow only)

The canonical production path is separate from validation-only reproduction.
On a genuinely new build, after preprocessing and model selection, explicitly
perform these operations in order:

```bash
uv run python scripts/train.py --models all
uv run python scripts/evaluate.py
uv run python scripts/build_serving_artifacts.py
```

- `train.py` trains the final NeuMF and Two-Tower models using training plus
  **warm validation** interactions, then publishes an immutable model release.
- `evaluate.py` produces the frozen held-out test report. The test set is
  intended for **one final evaluation**, not repeated model selection.
- `build_serving_artifacts.py` creates a serving release pinned to the exact
  immutable model release.

**If you already have the validated canonical model release, serving release,
and final-test report, skip all three commands.**

### 6. Complete reproduction analysis (requires canonical assets)

The last two numbered scripts are **not** stand-alone steps after preprocessing:

- **Script 05** requires the run's validation checkpoints and selection
  release from scripts 02–04, the canonical model release, the processed
  `test.parquet` file, and the existing canonical final-test report. It
  recomputes validation-stage ranking comparisons and checks the existing
  final-test evidence; it does **not** reevaluate held-out test targets.
- **Script 06** requires the canonical production release and its supporting
  data/metadata, as well as the reproduction results. It analyzes the existing
  production models without retraining them.

Only after these prerequisites exist, finish the same fresh run:

```bash
uv run python scripts/05_ranking_and_evaluation.py --run-id "$RUN_ID"
uv run python scripts/06_scalability_analysis.py --run-id "$RUN_ID"
```

Keep `RUN_ID` from the same terminal session, or set it explicitly to the exact
ID used in scripts 02–04. The standard workflow refuses to overwrite existing
run artifacts. Do not run two processes using the same run ID concurrently;
interrupted publications may require manual recovery rather than a rerun using
the same ID.

Reproduction artifacts live under:

```text
artifacts/reproductions/<run-id>/
├── baseline_mf_results.json
├── baseline_mf_training_history.json
├── ncf_results.json
├── ncf_training_history.json
├── neumf.pt
├── two_tower_results.json
├── two_tower_training_history.json
├── ranking_validation_results.json
├── scalability_results.json
└── selection_releases/
    ├── current_two_tower_selection.json
    └── two-tower-selection-<fingerprint>/
        ├── selection_release_manifest.json
        ├── two_tower.pt
        ├── two_tower_user_embeddings.pt
        └── two_tower_item_embeddings.pt
```

Generated checkpoints, reports, and selection metadata remain local; they
are not included in the source repository. Publication is not yet safe under
concurrent same-ID execution, and interruption can leave partial outputs.

### 7. Run an already-prepared API locally

When the canonical model and serving releases (including all binary payloads)
are available under `artifacts/`, start FastAPI:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Use `/health` to confirm that the model and manifests actually loaded before
making a recommendation request. A clean clone will not be ready until the
required model and serving artifacts are created or provided separately.

---

## API

### Health

```bash
curl http://localhost:8000/health
```

Healthy response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "manifest_validated": true
}
```

### Model information

```bash
curl http://localhost:8000/model/info
```

Returns the loaded serving/model release IDs, device, vocabulary sizes, candidate size, and model names.

### Single-user recommendation

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 2, "k": 10}'
```

### Batch recommendation

```bash
curl -X POST http://localhost:8000/recommend/batch \
  -H "Content-Type: application/json" \
  -d '{"user_ids": [2, 3, 6], "k": 10}'
```

API behavior:

- `k` must be between **1 and 50**
- unknown users return **404**
- invalid request values are rejected by FastAPI/Pydantic
- the service returns **503** if the serving bundle is unavailable or fails validation
- batch requests support up to **100 users**

---

## Docker

Build the CPU-only serving image:

```bash
docker build -t movie-recommender .
```

The model and serving artifacts are **not copied into the image**. For local
Docker Desktop testing, bind-mount the existing canonical `artifacts/` directory
read-only and expose a loopback-only host port:

```bash
docker run --rm \
  --name movie-recommender-api \
  -p 127.0.0.1:18080:8000 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  movie-recommender
```

In another terminal, check readiness and a known-user recommendation:

```bash
curl -i http://localhost:18080/health
curl -i http://localhost:18080/model/info
curl -i -X POST http://localhost:18080/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 2, "k": 10}'
```

The repository's canonical release was smoke-tested on **macOS Docker Desktop**:
the non-root container reached `healthy`, both metadata endpoints returned
HTTP 200, and a real recommendation returned ten movies for known user ID 2.
These are smoke-test results, not throughput or latency benchmarks.

**Linux/other-host permissions:** The container runs as an unprivileged user.
Read-only bind mounts do not bypass POSIX read/traverse permissions; verify
that the container runtime UID can read model/serving pointers and traverse
release directories. A different host may need adjusted artifact permissions
or an appropriate non-root runtime UID/GID. Docker Desktop success is not a
guarantee that arbitrary Linux host mounts have compatible permissions.

---

## Tests and Quality Gates

Run the complete test suite and static checks:

```bash
uv run pytest -q
uv run ruff check .
uv lock --check
```

At the documented pre-release check, **94 tests passed**, Ruff passed, and the
157-package lockfile check succeeded. One nonblocking Starlette test-client
deprecation warning was reported.

The project also validates artifact fingerprints, vocabulary identity, release lineage, model-training protocol, serving-bundle consistency, and seen-history bounds at runtime.

---

## Limitations

### Warm-start collaborative system

The current Two-Tower and NeuMF models are ID-based. They require users and movies to exist in the frozen training vocabularies.

Unseen users and unseen items are therefore not natively supported.

A natural extension is to incorporate side information such as movie genres, title/text embeddings, release year, and user/context features into feature-aware retrieval/ranking towers.

### Exact retrieval

The current catalog contains 17,818 movies, so exact dot-product retrieval remains practical.

At much larger catalog sizes, the learned item embeddings can be indexed using ANN systems such as FAISS, HNSW, ScaNN, or a managed vector-search service.

### Popularity bias

Recommendation exposure remains concentrated in popular items. A production system should monitor and potentially optimize long-tail exposure, catalog coverage, novelty, diversity, fairness, and business constraints.

### Offline evaluation

Offline ranking metrics are useful for model selection but do not prove user impact. A deployed system should ultimately be validated with online experimentation such as A/B testing.

---

## Future Improvements

- ANN-based retrieval for larger catalogs
- metadata-aware cold-start towers
- sequence-aware user encoders
- diversity-aware reranking
- calibrated recommendation objectives
- online feature retrieval
- model monitoring and drift detection
- automated retraining orchestration
- A/B testing and experimentation infrastructure
- distributed retrieval/ranking for larger traffic volumes

---

## Summary

This project demonstrates how a recommendation system can evolve from classical baselines into a production-style two-stage architecture while preserving strict evaluation discipline and reproducible artifact lineage.

The final system combines:

- **Two-Tower retrieval**
- **Top-500 candidate generation**
- **NeuMF ranking**
- **strict temporal validation/test separation**
- **train + warm-validation final training**
- **immutable model and serving releases**
- **FastAPI inference**
- **batched recommendation**
- **Dockerized CPU deployment**

Final held-out performance:

| Metric | Value |
|---|---:|
| Candidate Recall@500 | **0.655** |
| Recall@10 | **0.072** |
| NDCG@10 | **0.036** |
| Recall@50 | **0.229** |
| NDCG@50 | **0.069** |
