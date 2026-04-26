# RAG Evaluation Lab - Architecture

> Experimentation platform for RAG pipelines with automated LLM-as-a-Judge evaluation, MLflow tracking, and declarative YAML configuration.
>
> The project is designed to answer one engineering question: which combination of chunking, embedding, retrieval, and generation produces the best answers for this corpus?

---

## 1. System Overview

The platform is split into sequential subsystems with explicit contracts. Each subsystem can evolve independently without rewriting the rest of the pipeline.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      RAG EVALUATION LAB - Pipeline                      │
└──────────────────────────────────────────────────────────────────────────┘

   PDFs                            ┌────────────────────────┐
   ──────►   ① INGESTION   ──────► │ Pinecone (namespaces) │
   corpus/   chunker+embedder      │  • exp_001 -> vectors │
                                   │  • exp_002 -> vectors │
                                   └────────────┬───────────┘
                                                │
   PDFs                                         │
   ──────►   ② QA GENERATION ────► benchmark_dataset.json
   corpus/   LLM generates Q&A    (versioned ground truth)
                                                │
                                                │
   YAML ─►   ③ RAG RUNNER    ─────► run_results.json
   config    retriever+generator   {Q, expected_A, ctx, predicted_A}
                                                │
                                                │
             ④ EVALUATION ──────► metrics.json
             LLM-as-a-Judge       (faithfulness, relevancy, recall)
                                                │
                                                ▼
             ⑤ MLflow Tracking ──► UI + reports/comparison.html
             params + metrics + artifacts
```

Architectural principles:

1. Declarative, not imperative configuration. Experiment changes live in YAML, not in code edits.
2. Pinecone namespaces are the experiment boundary. Different chunking strategies can coexist in one index.
3. Evaluation is a first-class output, not an afterthought.
4. Determinism where possible: fixed config, stable corpus hashing, zero-temperature judge.
5. Cost is a product concern: token estimates, embedding cache, and usage tracking are part of the design.

---

## 2. Technology Choices

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.11+ | Mature type system and LLM ecosystem |
| PDF parsing | `pypdf` | Reliable baseline for heterogeneous PDFs |
| Chunking | `langchain-text-splitters` | Strong default recursive splitter |
| Embeddings | OpenAI `text-embedding-3-small` | Good cost / quality ratio |
| Vector store | Pinecone | Native namespaces and simple hosted infra |
| Generation / Judge | OpenAI `gpt-5.4-mini` | Better answer quality with still-manageable cost |
| Config | `pydantic` + `pyyaml` | Fail fast before paid API calls |
| CLI | `typer` | Typed command-line interfaces |
| Tracking | `mlflow` | Built-in comparison UI |
| Reporting | `jinja2` + `pandas` | Static, portable artifacts |
| Logging | `rich` | Readable logs and progress bars |
| Retries | `tenacity` | Backoff for external calls |
| Token estimation | `tiktoken` | Cost estimation before execution |
| Similarity | `numpy` + `scikit-learn` | Deduplication for generated QA pairs |
| Tests | `pytest` + mocks | Fast offline validation |

---

## 3. Repository Layout

```
rag-eval-lab/
├── ARCHITECTURE.md
├── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
│
├── configs/
│   ├── exp_001_chunk256_ada.yaml
│   ├── exp_002_chunk512_ada.yaml
│   ├── exp_003_chunk128_topk10.yaml
│   └── exp_004_chunk256_topk3.yaml
│
├── data/
│   ├── corpus/
│   ├── benchmark/
│   ├── runs/
│   └── cache/
│
├── scripts/
│   ├── ingest_corpus.py
│   ├── generate_benchmark.py
│   ├── run_experiment.py
│   ├── evaluate_run.py
│   └── compare_experiments.py
│
├── src/rag_eval_lab/
│   ├── config/
│   ├── ingestion/
│   ├── qa_generation/
│   ├── rag/
│   ├── evaluation/
│   ├── tracking/
│   └── utils/
│
└── tests/
```

Generated artifacts such as `data/benchmark/`, `data/runs/`, `data/cache/`, `mlruns/`, and reports should not be committed.

---

## 4. Checkpoints

### CP0 - Bootstrap

Goal:
- Create a clean project baseline with packaging, logging, retries, config loading, and test scaffolding.

Expected deliverables:
- `pyproject.toml`
- `.env.example`
- `.gitignore`
- shared OpenAI wrapper
- I/O helpers
- logging helpers
- basic test fixtures

Current status:
- Implemented

### CP1 - Corpus Ingestion and Pinecone Indexing

Goal:
- Read the PDF corpus, split it into configurable chunks, embed those chunks, and index them into a Pinecone namespace keyed by `experiment_id`.

Expected deliverables:
- `scripts/ingest_corpus.py`
- configurable chunking
- embedding cache
- Pinecone upsert/query/delete namespace support
- ingestion report with chunk counts and estimated cost

Key design decisions:
- Fail if the namespace already exists unless `--rebuild` is explicitly used.
- Cache embeddings by corpus hash + model + chunk parameters.
- Store chunk text in Pinecone metadata for simpler retrieval.

Current status:
- Implemented

### CP2 - Benchmark Dataset Generation

Goal:
- Build a synthetic QA benchmark from the corpus itself.

Expected deliverables:
- `scripts/generate_benchmark.py`
- benchmark JSON versioned by corpus hash
- question type distribution logging
- deduplication and trivial-question filtering

Key design decisions:
- The LLM generates both the benchmark question and expected answer from each chunk.
- Output must be English-only.
- Deduplication is semantic, not string-based.
- The dataset is versioned by corpus hash to prevent mixing incompatible benchmarks.

Current status:
- Implemented

### CP3 - YAML Config + RAG Runner

Goal:
- Run one declarative experiment end to end using an existing benchmark and indexed corpus.

Expected deliverables:
- `scripts/run_experiment.py`
- retrieval module
- answer generation module
- `run_results.json`
- force-protected reruns

Key design decisions:
- The benchmark stays in JSON; only corpus chunks are indexed in Pinecone.
- If the namespace is empty, the runner triggers ingestion first.
- The output includes a config snapshot, token usage, and latency per question.

Current status:
- Implemented and validated with 4 real experiments
- 1054 questions answered per experiment across all 4 configs
- run_results.json saved per experiment with Q, expected_A, context, predicted_A, tokens, latency

### CP4 - LLM-as-a-Judge

Goal:
- Score each RAG answer on faithfulness, answer relevancy, and context recall.

Expected deliverables:
- evaluation prompts
- structured judge parsing
- `metrics.json`
- aggregation logic

Current status:
- Implemented: judge.py (3 prompts, n_reps median), metrics.py (aggregation + MetricStats), scripts/evaluate_run.py (CLI)
- 15 unit tests passing offline
- Real acceptance requires running scripts/evaluate_run.py against live run_results.json

### CP5 - MLflow Tracking and Comparative Report

Goal:
- Track experiments consistently and render an offline comparison artifact.

Expected deliverables:
- MLflow logger
- comparison report
- artifact logging

Current status:
- Not implemented yet

### CP6 - End-to-End Productization

Goal:
- Produce a polished repository with real experiment results, final README, and end-to-end smoke coverage.

Expected deliverables:
- multiple real runs
- comparison-ready outputs
- final README with real findings
- smoke E2E test

Current status:
- Not implemented yet

---

## 5. Runtime Behavior

### Ingestion

1. Load experiment config.
2. Read corpus pages from PDF.
3. Split pages into chunks.
4. Estimate token cost.
5. Reuse cached embeddings if available.
6. Otherwise call the embedding model.
7. Upsert vectors into Pinecone under the experiment namespace.

### Benchmark generation

1. Read the corpus.
2. Split it into chunks.
3. Ask the LLM to generate QA pairs from each chunk.
4. Filter malformed or trivial items.
5. Deduplicate semantically similar questions.
6. Save a versioned benchmark JSON.

### Experiment run

1. Load config.
2. Load benchmark JSON.
3. Verify the benchmark corpus hash matches the configured corpus.
4. Ensure the Pinecone namespace exists, ingesting if needed.
5. For each benchmark question:
   - embed the question
   - retrieve top-k chunks from Pinecone
   - answer using only the retrieved context
   - save retrieved context, predicted answer, tokens, and latency
6. Save `run_results.json`.

---

## 6. Acceptance Guidance

The project should be considered complete when:

1. A fresh clone can run the documented setup successfully.
2. Multiple experiment YAMLs produce distinct run outputs.
3. Benchmark generation, RAG runs, and evaluation artifacts are reproducible.
4. Tests pass offline with mocks.
5. Real runs produce a final comparison with grounded conclusions.

---

## 7. Out of Scope for the MVP

- Re-ranking with a cross-encoder
- Multi-corpus normalization
- Fully asynchronous execution
- Custom web UI beyond MLflow and static HTML
- Deployment-oriented CI/CD

These are roadmap items, not MVP blockers.
