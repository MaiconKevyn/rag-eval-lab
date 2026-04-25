# RAG Evaluation Lab

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991.svg)](https://openai.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-v8-1E88E5.svg)](https://www.pinecone.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-0194E2.svg)](https://mlflow.org/)

> Experimentation platform for RAG pipelines with automatic **LLM-as-a-Judge** evaluation, MLflow tracking, and 100% declarative YAML configuration.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Full Pipeline](#full-pipeline)
- [Experiments](#experiments)
- [Evaluation - LLM-as-a-Judge](#evaluation---llm-as-a-judge)
- [Results](#results)
- [Tests](#tests)
- [Roadmap](#roadmap)

---

## Overview

**RAG Evaluation Lab** was built to answer one precise question:

> *Which combination of chunking + embedding + retrieval + generation produces the best answers for this corpus?*

This project is not just "yet another RAG." It is an **experimentation platform** where each variable is isolated, every run is tracked, and quality is measured by an LLM judge across three orthogonal metrics.

### Core capabilities

- **Declarative configuration** - changing `chunk_size` from 256 to 512 requires editing YAML, not code
- **LLM-generated benchmark** - QA pairs created from the corpus with semantic-similarity deduplication
- **LLM-as-a-Judge** - 3-dimensional evaluation (faithfulness, answer relevancy, context recall) with 3 repetitions to reduce variance
- **Embedding cache** - running two experiments with the same chunking does not bill OpenAI twice
- **MLflow tracking** - all parameters, metrics, and artifacts versioned with a native comparison UI
- **Cost as a feature** - `tiktoken` estimates tokens before each call; cumulative cost is logged per experiment

---

## Architecture

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

**Architectural principles:**
1. **Declarative, not imperative configuration** - YAML changes across experiments; the code remains stable
2. **Namespaces as the unit of experimentation** - each `experiment_id` is an isolated namespace in Pinecone
3. **Evaluation is a first-class concern** - the benchmark is generated, validated, and versioned by corpus hash
4. **Determinism** - `temperature=0.0` in the judge, fixed seeds, cache invalidated by file hash
5. **Cost is a feature** - parquet embedding cache, Pydantic validation before any paid call

---

## Technology Stack

| Layer | Technology | Why |
|---|---|---|
| **Runtime** | Python 3.11+ | Mature type hints, strong LLM ecosystem |
| **PDF Parsing** | `pypdf` | Robust for heterogeneous corpora |
| **Chunking** | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` is state of the art |
| **Embeddings** | `openai` SDK (`text-embedding-3-small`) | Abstraction allows swapping to `voyage`, `cohere` |
| **Vector Store** | `pinecone` v8 | Native namespaces, free serverless tier |
| **LLM** | `openai` SDK (`gpt-4o-mini`) | Cost/quality balance, JSON mode |
| **Config** | `pydantic` v2 + `pyyaml` | Validates on load, fails before the API call |
| **CLI** | `typer` | Declarative, type-safe |
| **Tracking** | `mlflow` >= 2.10 | Built-in UI, side-by-side comparison |
| **Reporting** | `jinja2` + `pandas` | Static, versionable HTML |
| **Logging** | `rich` | Readable terminal output, progress bars |
| **Retries** | `tenacity` | Exponential backoff for every external call |
| **Tokenization** | `tiktoken` | Cost estimation before each call |
| **Similarity** | `numpy` + `scikit-learn` | Semantic deduplication of QA pairs |
| **Cache** | `pandas` + parquet | Embeddings reused across experiments |
| **Tests** | `pytest` + mocks | 47 tests, no network, <2s |

---

## Project Structure

```
rag-eval-lab/
├── configs/                          ← declarative experiments
│   ├── exp_001_chunk256_ada.yaml     # baseline: chunk=256, top_k=5
│   ├── exp_002_chunk512_ada.yaml     # larger chunk: 512, top_k=5
│   ├── exp_003_chunk128_topk10.yaml  # granular: chunk=128, top_k=10
│   └── exp_004_chunk256_topk3.yaml   # low recall: chunk=256, top_k=3
│
├── data/
│   └── corpus/
│       └── agentic_ai_landscape.pdf  # main corpus (34 pages)
│
├── src/rag_eval_lab/
│   ├── config/
│   │   ├── schema.py                 # Pydantic: ExperimentConfig + sub-configs
│   │   └── loader.py                 # load_config(path) -> ExperimentConfig
│   ├── ingestion/
│   │   ├── chunker.py                # Chunk dataclass + configurable Chunker
│   │   ├── embedder.py               # Embedder Protocol + OpenAIEmbedder
│   │   ├── pinecone_store.py         # upsert / query / delete_namespace
│   │   └── ingest.py                 # orchestrator with parquet cache
│   ├── qa_generation/
│   │   ├── generator.py              # QAGenerator: LLM -> QA pairs (JSON mode)
│   │   ├── validator.py              # cosine-similarity dedup + trivial filter
│   │   └── dataset.py                # BenchmarkDataset: versioned save / load
│   ├── rag/                          # [CP3] retriever + answerer + runner
│   ├── evaluation/                   # [CP4] judge + metrics + HTML reporter
│   ├── tracking/                     # [CP5] mlflow_logger
│   └── utils/
│       ├── io.py                     # sha256_of_file, read_json, write_json
│       ├── llm_client.py             # OpenAI wrapper: retry + cost + json_mode
│       └── logging.py                # rich handler configurable via LOG_LEVEL
│
├── scripts/
│   ├── ingest_corpus.py              # CLI: PDF -> Pinecone
│   ├── generate_benchmark.py         # CLI: corpus -> benchmark QA pairs
│   ├── run_experiment.py             # [CP3] CLI: benchmark + retrieval -> results
│   ├── evaluate_run.py               # [CP4] CLI: LLM-as-a-Judge
│   └── compare_experiments.py        # [CP5] CLI: generates comparison report
│
├── tests/
│   ├── conftest.py                   # FakeOpenAIClient fixture
│   ├── test_bootstrap.py             # utils: io, llm_client, logging
│   ├── test_chunker.py               # chunking: counts, overlap, unique IDs
│   ├── test_config_loader.py         # Pydantic schema + YAML loader
│   └── test_qa_validator.py          # JSON parsing, dedup, trivial filter
│
├── ARCHITECTURE.md                   # design decisions + detailed checkpoints
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### Prerequisites

- **Python** 3.11 or higher
- **OpenAI account** with API key - [platform.openai.com](https://platform.openai.com)
- **Pinecone account** (free tier is enough) - [pinecone.io](https://www.pinecone.io)
  - Create an index with `dimension=1536`, `metric=cosine`, serverless

### Installation

```bash
# Clone the repository
git clone https://github.com/MaiconKevyn/rag-eval-lab.git
cd rag-eval-lab

# Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=rag-eval-lab
PINECONE_ENVIRONMENT=us-east-1
LOG_LEVEL=INFO
```

### Full Pipeline

```bash
# 1. Index the corpus in Pinecone
python scripts/ingest_corpus.py --config configs/exp_001_chunk256_ada.yaml

# 2. Generate the QA-pair benchmark
python scripts/generate_benchmark.py \
    --corpus data/corpus/agentic_ai_landscape.pdf \
    --n-per-chunk 3 \
    --out data/benchmark/

# 3. Run the RAG experiment  [available in CP3]
python scripts/run_experiment.py --config configs/exp_001_chunk256_ada.yaml

# 4. Evaluate with LLM-as-a-Judge  [available in CP4]
python scripts/evaluate_run.py --run data/runs/exp_001_chunk256_ada/run_results.json

# 5. View in MLflow  [available in CP5]
mlflow ui
```

---

## Experiments

Four declarative configurations are ready to run:

| Config | `chunk_size` | `chunk_overlap` | `top_k` | `score_threshold` | Hypothesis |
|---|---|---|---|---|---|
| `exp_001_chunk256_ada` | 256 | 32 | 5 | 0.70 | Baseline |
| `exp_002_chunk512_ada` | 512 | 64 | 5 | 0.70 | Do larger chunks lose semantics? |
| `exp_003_chunk128_topk10` | 128 | 16 | 10 | 0.65 | More granularity + more context |
| `exp_004_chunk256_topk3` | 256 | 32 | 3 | 0.75 | Precision vs recall |

Each YAML file controls the entire pipeline - chunking, embedding, retrieval, and generation - without changing a single line of code.

---

## Evaluation - LLM-as-a-Judge

Each RAG answer is evaluated across **3 metrics** by a judge LLM (`gpt-4o-mini`, `temperature=0.0`):

| Metric | What it measures | Inputs |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | `retrieved_context` + `predicted_answer` |
| **Answer Relevancy** | Does the answer address the question that was asked? | `question` + `predicted_answer` |
| **Context Recall** | Did the context contain the necessary information? | `expected_answer` + `retrieved_context` |

**Design decisions:**
- Score from 1-5 (ordinal, easier for the LLM to calibrate than 0-1)
- 3 repetitions per evaluation -> median for robustness against outliers
- JSON mode (`response_format=json_object`) reduces parsing failures from ~5% to <1%
- `reasoning` is preserved for diagnosing where the RAG failed

---

## Results

> Results will be filled in after running the 4 experiments (CP6).

| exp_id | chunk | top_k | Faith | Relev | Recall | Overall | Cost |
|---|---|---|---|---|---|---|---|
| exp_001 | 256 | 5 | - | - | - | - | - |
| exp_002 | 512 | 5 | - | - | - | - | - |
| exp_003 | 128 | 10 | - | - | - | - | - |
| exp_004 | 256 | 3 | - | - | - | - | - |

---

## Tests

```bash
# Runs offline (LLM and Pinecone mocks), no cost
pytest

# With per-module verbosity
pytest -v
```

**47 tests** covering:
- `test_bootstrap.py` - utils: sha256, JSON roundtrip, mocked LLMClient
- `test_chunker.py` - chunk count, overlap, unique IDs, empty pages
- `test_config_loader.py` - Pydantic validation (overlap >= size, invalid paths, malformed YAML)
- `test_qa_validator.py` - valid/malformed JSON parsing, cosine-similarity dedup, trivial filter

---

## Roadmap

- [ ] **CP3** - RAG Runner + YAML configs (sequential runner, `run_results.json`)
- [ ] **CP4** - LLM-as-a-Judge (faithfulness, relevancy, recall with 3 repetitions)
- [ ] **CP5** - MLflow tracking + comparative HTML report
- [ ] **CP6** - E2E pipeline + final README with real results and insights
- [ ] Re-ranking with a cross-encoder
- [ ] Hybrid search (BM25 + vector)
- [ ] Async runner to parallelize LLM calls

---

*Detailed architecture and design decisions are available in [`ARCHITECTURE.md`](./ARCHITECTURE.md).*
