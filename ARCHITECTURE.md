# Cyber Assessment — RAG Evaluation Lab

> Sistema de **experimentação de pipelines RAG** com avaliação automática (LLM-as-a-Judge), tracking de métricas (MLflow) e configuração declarativa via YAML.
>
> O projeto não é "mais um RAG". É uma **plataforma para responder à pergunta**: *qual configuração de chunking + embedding + retrieval + generation produz as melhores respostas neste corpus?*

---

## 1. Visão Geral da Arquitetura

O sistema é dividido em **4 subsistemas independentes** que se encaixam em sequência. Cada subsistema tem entrada, saída e contrato bem definidos. Essa separação permite trocar implementações (modelo de embedding, vector store, LLM judge) sem reescrever o resto.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      RAG EVALUATION LAB — Pipeline                        │
└──────────────────────────────────────────────────────────────────────────┘

   PDFs                            ┌────────────────────────┐
   ──────►   ① INGESTION   ──────► │ Pinecone (namespaces)  │
   corpus/   chunker+embedder      │  • exp_001 → vetores   │
                                   │  • exp_002 → vetores   │
                                   └────────────┬───────────┘
                                                │
   PDFs                                         │
   ──────►   ② QA GENERATION ────► benchmark_dataset.json
   corpus/   LLM gera Q&A pairs    (ground truth versionado)
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
             params + métricas + artefatos
```

**Princípios arquiteturais:**

1. **Configuração declarativa, não imperativa.** Tudo que muda entre experimentos vive em YAML. O código é estável; o YAML varia.
2. **Namespaces como unidade de experimento.** Cada `experiment_id` vira um namespace no Pinecone — você roda 10 chunkings diferentes no mesmo índice sem conflito.
3. **Avaliação é cidadã de primeira classe.** O benchmark é gerado, validado e versionado. Não é um afterthought.
4. **Determinismo onde for possível.** `temperature=0.0` no judge, seeds fixas, hash de corpus para invalidar cache.
5. **Custo é uma feature.** Cache de embeddings, validação Pydantic *antes* de chamar API, retries com backoff exponencial.

---

## 2. Stack Tecnológica

| Camada | Tecnologia | Por quê |
|---|---|---|
| **Runtime** | Python 3.11+ | Type hints maduros, performance, ecossistema LLM |
| **Gerenciamento de deps** | `uv` ou `pip` + `requirements.txt` | Reprodutibilidade |
| **Parsing PDF** | `pypdf` (fallback `pymupdf`) | Robusto para corpus heterogêneo |
| **Chunking** | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` é estado-da-arte |
| **Embeddings** | `openai` SDK (default `text-embedding-3-small`) | Abstração permite trocar para `voyage`, `cohere` |
| **Vector store** | `pinecone-client` v3+ | Namespaces nativos, serverless |
| **LLM (gen + judge)** | `openai` SDK (`gpt-4o-mini` default) | Custo/qualidade, JSON mode |
| **Config** | `PyYAML` + `pydantic` v2 | Validação no load, não em runtime |
| **CLI** | `typer` | Declarativo, type-safe |
| **Tracking** | `mlflow` ≥ 2.10 | UI built-in, comparação lado-a-lado |
| **Reporting** | `jinja2` + `pandas` | HTML estático, fácil de versionar |
| **Logging** | `structlog` ou `rich.logging` | Logs estruturados, terminal bonito |
| **Retries** | `tenacity` | Backoff exponencial em chamadas LLM |
| **Tokenização** | `tiktoken` | Estimativa de custo antes da call |
| **Similaridade** | `numpy` + `scikit-learn` | Cosine similarity para deduplicação QA |
| **Testes** | `pytest` + `pytest-mock` | Mock de LLM/Pinecone |
| **Lint/Format** | `ruff` + `mypy` | Padrão atual |
| **Secrets** | `python-dotenv` | `.env` local, var de ambiente em CI |
| **Notebooks** | `jupyter` (opcional) | Exploração ad-hoc |

**APIs externas:**
- OpenAI (embeddings + chat completion)
- Pinecone (serverless tier funciona para o projeto)

**Estimativa de custo (corpus de ~50 páginas, 4 experimentos, 200 QA pairs):**
- Embeddings: ~$0.10
- QA generation: ~$1.50 (gpt-4o-mini)
- RAG runs (4 × 200): ~$2.00
- Judge (3 métricas × 3 repetições × 800): ~$5.00
- **Total estimado: ~$10**

---

## 3. Estrutura do Repositório

```
cyber_assessment/
├── ARCHITECTURE.md              ← este arquivo
├── README.md                    ← entregável do Checkpoint 6
├── .env.example                 ← OPENAI_API_KEY, PINECONE_API_KEY
├── .gitignore                   ← .env, .venv, data/, mlruns/
├── pyproject.toml               ← deps + tooling
├── requirements.txt             ← deps pin-locked
│
├── configs/                     ← experimentos declarativos
│   ├── exp_001_chunk256_ada.yaml
│   ├── exp_002_chunk512_ada.yaml
│   ├── exp_003_chunk128_topk10.yaml
│   └── exp_004_chunk256_topk3.yaml
│
├── data/                        ← gitignored (exceto corpus/)
│   ├── corpus/
│   │   └── relatorio.pdf
│   ├── benchmark/
│   │   └── benchmark_v1_<corpus_hash>.json
│   └── runs/
│       └── exp_001/
│           ├── run_results.json
│           └── metrics.json
│
├── src/cyber_assessment/
│   ├── __init__.py
│   ├── config/
│   │   ├── schema.py            ← Pydantic models
│   │   └── loader.py            ← load_config(path) → ExperimentConfig
│   ├── ingestion/
│   │   ├── chunker.py           ← parametrizável
│   │   ├── embedder.py          ← interface + implementações
│   │   ├── pinecone_store.py    ← upsert/query por namespace
│   │   └── ingest.py            ← orquestrador
│   ├── qa_generation/
│   │   ├── generator.py         ← LLM → QA pairs
│   │   ├── validator.py         ← dedup por similaridade
│   │   └── dataset.py           ← persistência versionada
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── runner.py            ← orquestrador
│   ├── evaluation/
│   │   ├── judge.py             ← prompts das 3 métricas
│   │   ├── metrics.py           ← agregação
│   │   └── reporter.py          ← HTML
│   ├── tracking/
│   │   └── mlflow_logger.py
│   └── utils/
│       ├── io.py                ← read/write JSON, hash de arquivo
│       ├── llm_client.py        ← wrapper com retry + tiktoken
│       └── logging.py
│
├── scripts/                     ← entry points (CLI via typer)
│   ├── ingest_corpus.py
│   ├── generate_benchmark.py
│   ├── run_experiment.py
│   ├── evaluate_run.py
│   └── compare_experiments.py
│
├── tests/
│   ├── conftest.py              ← fixtures + mocks LLM/Pinecone
│   ├── test_chunker.py
│   ├── test_config_loader.py
│   ├── test_qa_validator.py
│   ├── test_judge_parsing.py
│   └── fixtures/
│       ├── mini_corpus.pdf
│       └── mock_llm_responses.json
│
├── reports/
│   └── comparison.html          ← gerado pelo reporter
│
└── mlruns/                      ← gitignored, gerado pelo MLflow
```

---

## 4. Checkpoints Detalhados

### Convenção de status

Cada checkpoint tem:
- **Objetivo** — o problema que resolve
- **Entrega** — artefatos concretos
- **Componentes** — arquivos com interfaces propostas
- **Decisões de design** — o "porquê" que guia edge cases
- **Critérios de aceite** — testes que devem passar
- **Riscos & armadilhas** — o que costuma quebrar
- **Tarefas** — checklist executável

---

### 🔵 Checkpoint 0 — Bootstrap do projeto (0.5 dia)

> **Não estava no brief, mas é pré-requisito.** Sem isso, o Checkpoint 1 vai ter setup misturado com lógica de negócio.

**Entrega:** repositório navegável, `pip install -e .` funciona, `.env.example` documentado, CI mínimo opcional.

**Tarefas:**
- [x] `pyproject.toml` com deps mínimas (ver §2)
- [x] `.env.example` com `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_ENVIRONMENT`
- [x] `.gitignore` cobrindo `.env`, `.venv/`, `data/runs/`, `data/benchmark/`, `mlruns/`, `__pycache__/`
- [x] `src/rag_eval_lab/utils/llm_client.py` com wrapper único OpenAI (retry + log de tokens)
- [x] `src/rag_eval_lab/utils/io.py` com `sha256_of_file(path)`, `read_json`, `write_json`
- [x] `src/rag_eval_lab/utils/logging.py` configurando `rich` ou `structlog`
- [x] `tests/conftest.py` com fixture `mock_openai_client`
- [x] `README.md` (versão stub) com Quick Start de 3 comandos

**Decisão de design — wrapper único de LLM:** Toda chamada à OpenAI passa por `llm_client.complete(messages, model, json_mode=True)`. Isso permite:
- Logar tokens (custo)
- Retry centralizado
- Trocar provedor sem reescrever 5 módulos
- Mockar em testes com 1 fixture

---

### 🔵 Checkpoint 1 — Corpus Ingestion & Pinecone Indexing (3–4 dias)

**Objetivo:** transformar PDFs em vetores indexados, **com tudo parametrizável**, porque a hipótese do projeto é que `chunk_size` afeta a qualidade — então o módulo não pode hardcodar nada.

**Entrega:**
- `python scripts/ingest_corpus.py --config configs/exp_001.yaml` faz upsert no Pinecone
- Namespace = `experiment_id` do YAML
- Cada vetor tem metadados: `chunk_id`, `source_file`, `page`, `chunk_index`, `text`

#### Componentes

**`src/cyber_assessment/ingestion/chunker.py`**
```python
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class Chunk:
    chunk_id: str          # f"{source}_{page}_{idx}"
    text: str
    source: str            # nome do arquivo
    page: int
    chunk_index: int

class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int,
                 separators: list[str] | None = None):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, pages: list[tuple[int, str]], source: str) -> list[Chunk]:
        ...
```

**`src/cyber_assessment/ingestion/embedder.py`**
```python
from typing import Protocol

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dim(self) -> int: ...
    @property
    def model_name(self) -> str: ...

class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small",
                 batch_size: int = 100): ...
```

**`src/cyber_assessment/ingestion/pinecone_store.py`**
```python
class PineconeStore:
    def __init__(self, index_name: str, dim: int): ...
    def upsert(self, namespace: str, chunks: list[Chunk],
               embeddings: list[list[float]]) -> None: ...
    def query(self, namespace: str, vector: list[float],
              top_k: int, score_threshold: float | None = None) -> list[dict]: ...
    def delete_namespace(self, namespace: str) -> None: ...
    def list_namespaces(self) -> list[str]: ...
```

**`src/cyber_assessment/ingestion/ingest.py`**
```python
def ingest(config: ExperimentConfig) -> IngestionReport:
    """
    1. Lê PDF(s) do corpus
    2. Chunk parametrizado pelo YAML
    3. Embedda em batches
    4. Upsert no namespace = config.experiment_id
    5. Retorna report: {n_chunks, n_pages, total_tokens, namespace}
    """
```

#### Decisões de design

- **Idempotência:** antes do upsert, `delete_namespace(experiment_id)` se a flag `--rebuild` estiver setada. Sem flag, falha se namespace já existe e tem vetores. Evita upsert duplicado silencioso.
- **Cache de embeddings:** `data/cache/embeddings/<corpus_hash>_<model>_<chunk_size>_<overlap>.parquet`. Rodar dois experimentos com mesmo chunking + embedding não chama OpenAI duas vezes.
- **Hash do corpus:** `sha256` de cada arquivo, armazenado no metadata. Se o PDF muda, o cache invalida.
- **`metadata.text` no Pinecone:** o chunk de texto VAI no metadata. Sim, infla o índice — mas evita um segundo armazenamento. Para corpus grande (>1M chunks), refatorar para S3 + ID.

#### Critérios de aceite

- [x] PDF de 20 páginas → ≥30 chunks com `chunk_size=256` _(34 págs → 442 chunks)_
- [x] `pinecone.list_namespaces()` mostra o namespace criado _(`exp_001_chunk256_ada` confirmado)_
- [x] Query random no namespace retorna top_k vetores com `chunk_id`, `source`, `page` no metadata
- [x] `--rebuild` apaga e recria; sem `--rebuild`, falha em namespace existente
- [x] Teste unitário: `Chunker.split()` com texto de 1000 chars + `chunk_size=200, overlap=20` produz 6 chunks com overlap correto
- [x] Custo logado: "Embedded 442 chunks (~20238 tokens, ~$0.0004)"

#### Riscos & armadilhas

- **Pinecone serverless tem dimensão fixa.** Se você criou o índice com `dim=1536` e troca para `text-embedding-3-large` (3072), quebra silenciosamente. → validar `embedder.dim == index.describe().dimension` no boot.
- **Páginas vazias em PDFs.** `pypdf` retorna `""` em páginas escaneadas. Pular antes de chunkar, não no chunker.
- **`chunk_overlap >= chunk_size`** trava o splitter em loop. Validar no Pydantic.

---

### 🔵 Checkpoint 2 — Geração de Benchmark Dataset (4–5 dias)

**Objetivo:** gerar QA pairs **fiéis ao corpus** usando LLM, deduplicados e versionados. Esse é o coração intelectual do projeto — você está usando LLM para construir o ground truth, não só para responder.

**Entrega:**
- `python scripts/generate_benchmark.py --corpus data/corpus/X.pdf --n-per-chunk 3 --out data/benchmark/`
- `benchmark_v1_<corpus_hash>_<date>.json` com schema validado
- Distribuição de tipos de pergunta logada (factual / comparativo / por-quê / como)

#### Componentes

**`src/cyber_assessment/qa_generation/generator.py`**
```python
QA_PROMPT = """Dado o trecho abaixo, gere {n} pares (pergunta, resposta esperada).

Regras estritas:
- A pergunta deve ser respondível APENAS com o trecho fornecido
- A resposta deve ser fiel ao trecho, sem inferências externas
- Varie o tipo: factual, comparativo, "por que", "como"
- Se o trecho for muito curto/genérico para gerar {n} perguntas boas, gere menos

Retorne APENAS JSON válido neste schema:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "expected_answer": "...",
      "question_type": "factual|comparative|why|how",
      "source_chunk_id": "{chunk_id}"
    }}
  ]
}}

Trecho ({chunk_id}):
{chunk_text}
"""

class QAGenerator:
    def __init__(self, llm_client, model: str, n_per_chunk: int = 3): ...
    def generate_for_chunk(self, chunk: Chunk) -> list[QAPair]: ...
    def generate_for_corpus(self, chunks: list[Chunk]) -> list[QAPair]: ...
```

**`src/cyber_assessment/qa_generation/validator.py`**
```python
class QAValidator:
    """
    Filtra:
    1. JSON malformado / campos faltando
    2. Perguntas com cosine similarity > threshold (default 0.92)
    3. Perguntas triviais ("o que diz o documento?")
    """
    def __init__(self, embedder: Embedder, similarity_threshold: float = 0.92): ...
    def deduplicate(self, pairs: list[QAPair]) -> tuple[list[QAPair], DedupReport]: ...
    def filter_trivial(self, pairs: list[QAPair]) -> list[QAPair]: ...
```

**`src/cyber_assessment/qa_generation/dataset.py`**
```python
@dataclass
class BenchmarkDataset:
    version: str           # "v1"
    corpus_hash: str       # sha256 dos PDFs concatenados
    created_at: str        # ISO 8601
    generator_model: str
    qa_pairs: list[QAPair]

    def save(self, dir: Path) -> Path: ...   # benchmark_v1_<hash>_<date>.json
    @classmethod
    def load(cls, path: Path) -> "BenchmarkDataset": ...
```

#### Decisões de design

- **Dedup por embedding, não por string match.** "Qual é o orçamento de 2024?" e "Quanto custou em 2024?" são string-distantes e semanticamente idênticas.
- **`temperature=0.7` na geração.** Aqui você QUER variabilidade — perguntas iguais a cada run são inúteis. Mas seed fixa (`seed=42`) para reprodutibilidade quando OpenAI suportar.
- **Versionamento por hash do corpus.** O dataset *é* uma função do corpus. Se o PDF muda, o benchmark é outro. Não é possível misturar.
- **Distribuição de tipos:** rejeitar dataset onde >70% é "factual". Diversidade é critério de qualidade.

#### Critérios de aceite

- [ ] Corpus de 20 páginas → ≥50 QA pairs após dedup _(requer run real)_
- [ ] `question_type` distribuído (nenhum tipo > 60%) _(requer run real)_
- [ ] Schema do JSON validado por Pydantic ao carregar _(requer run real)_
- [ ] Re-rodar com mesmo corpus + mesmo modelo gera arquivo diferente (timestamp), mas com `corpus_hash` igual _(requer run real)_
- [x] Teste unitário: `validator.deduplicate([qa_a, qa_a_paraphrased])` remove um dos dois
- [x] Teste unitário: parsing falha graciosamente em JSON malformado (não mata a run inteira)

#### Riscos & armadilhas

- **LLM "alucinando" ground truth.** Mesmo com instrução estrita, ~10% das respostas vão extrapolar o chunk. → adicionar validação manual de amostra de 20 pairs antes de declarar o benchmark "pronto". Isso é trabalho humano, não automatizável.
- **Custo de geração.** Para 200 chunks × 3 perguntas, são 600 chamadas. Use `gpt-4o-mini`, não `gpt-4o`.
- **Chunks pequenos demais geram perguntas idiotas.** Se `chunk_size < 100`, a geração degrada. Avise no log.

---

### 🔵 Checkpoint 3 — YAML Config + RAG Runner (4–5 dias)

**Objetivo:** mecanismo de experimentação onde **mudar um parâmetro = trocar um YAML**. Zero alteração de código entre experimentos.

**Entrega:**
- `python scripts/run_experiment.py --config configs/exp_001.yaml` executa fim-a-fim
- `data/runs/<experiment_id>/run_results.json` com schema validado
- 2 YAMLs diferentes → 2 outputs diferentes, sem alterar código

#### Schema do YAML

```yaml
# configs/exp_001_chunk256_ada.yaml
experiment_id: "exp_001_chunk256_ada"
description: "Baseline com chunk_size=256 e text-embedding-3-small"
corpus: "data/corpus/relatorio_2024.pdf"
benchmark: "data/benchmark/benchmark_v1_abc123_2026-04-25.json"

chunking:
  chunk_size: 256
  chunk_overlap: 32
  separators: ["\n\n", "\n", ". ", " "]

embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  batch_size: 100

retrieval:
  top_k: 5
  score_threshold: 0.75

generation:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 512
  system_prompt: |
    Você é um assistente que responde perguntas usando APENAS o
    contexto fornecido. Se o contexto for insuficiente, diga
    "Não há informação suficiente no contexto."
```

#### Componentes

**`src/cyber_assessment/config/schema.py`**
```python
from pydantic import BaseModel, Field, field_validator

class ChunkingConfig(BaseModel):
    chunk_size: int = Field(gt=0, le=8192)
    chunk_overlap: int = Field(ge=0)
    separators: list[str] | None = None

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_lt_size(cls, v, info):
        if v >= info.data.get("chunk_size", 0):
            raise ValueError("chunk_overlap must be < chunk_size")
        return v

class EmbeddingConfig(BaseModel):
    provider: Literal["openai"]
    model: str
    batch_size: int = 100

class RetrievalConfig(BaseModel):
    top_k: int = Field(gt=0, le=50)
    score_threshold: float | None = Field(default=None, ge=0, le=1)

class GenerationConfig(BaseModel):
    provider: Literal["openai"]
    model: str
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(gt=0, le=4096)
    system_prompt: str

class ExperimentConfig(BaseModel):
    experiment_id: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    corpus: Path
    benchmark: Path
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
```

**`src/cyber_assessment/rag/retriever.py`**
```python
class Retriever:
    def __init__(self, store: PineconeStore, embedder: Embedder,
                 namespace: str, top_k: int,
                 score_threshold: float | None = None): ...
    def retrieve(self, question: str) -> list[RetrievedChunk]: ...
```

**`src/cyber_assessment/rag/generator.py`**
```python
ANSWER_PROMPT = """{system_prompt}

Contexto:
{context}

Pergunta: {question}

Resposta:"""

class Answerer:
    def answer(self, question: str, context: list[RetrievedChunk]) -> str: ...
```

**`src/cyber_assessment/rag/runner.py`**
```python
def run_experiment(config: ExperimentConfig) -> Path:
    """
    1. Garante ingestão (idempotente)
    2. Carrega benchmark
    3. Para cada QA pair: retrieve → answer
    4. Persiste run_results.json
    Retorna: path do run_results.json
    """
```

**Schema do `run_results.json`:**
```json
{
  "experiment_id": "exp_001_chunk256_ada",
  "config_snapshot": { ... },
  "benchmark_version": "v1_abc123_2026-04-25",
  "started_at": "2026-04-25T10:00:00Z",
  "finished_at": "2026-04-25T10:12:34Z",
  "n_questions": 200,
  "results": [
    {
      "qa_id": "abc123",
      "question": "Qual é o orçamento aprovado para 2024?",
      "expected_answer": "R$ 1.2 bilhão",
      "retrieved_context": [
        {"chunk_id": "doc_p3_5", "text": "...", "score": 0.87}
      ],
      "predicted_answer": "O orçamento aprovado é de R$ 1.2 bilhão.",
      "tokens_used": {"prompt": 450, "completion": 18},
      "latency_ms": 823
    }
  ]
}
```

#### Decisões de design

- **`config_snapshot` no output.** O YAML pode mudar depois. O snapshot garante que o resultado é interpretável daqui a 6 meses.
- **Pydantic falha CEDO.** Erro de campo no YAML aparece antes de qualquer chamada paga.
- **Runner é idempotente em re-execução.** Se `run_results.json` já existe, falha sem `--force`.
- **Latência e tokens por pergunta.** Análise de custo/throughput depois.
- **Sem paralelismo no MVP.** Adicionar `asyncio` é Checkpoint 7. Primeiro funcionar, depois paralelizar.

#### Critérios de aceite

- [ ] YAML mal formado (campo errado) → erro Pydantic claro, sem chamada à API
- [ ] `chunk_overlap >= chunk_size` → erro de validação
- [ ] Run completo de 200 perguntas em <15min com `gpt-4o-mini`
- [ ] `run_results.json` valida contra o schema Pydantic
- [ ] Dois YAMLs diferentes geram dois `run_results.json` diferentes (verificável por diff)
- [ ] Teste unitário: runner com `mock_pinecone + mock_llm` produz JSON válido sem internet

#### Riscos & armadilhas

- **Recuperar 0 chunks** com `score_threshold` muito alto. → quando isso acontecer, registrar `retrieved_context: []` e seguir, não falhar.
- **LLM responde "não sei"** com contexto bom (recall failure no judge). É legítimo. Não tratar como erro.
- **`max_tokens` cortando resposta.** Logar quando `finish_reason == "length"`.

---

### 🔵 Checkpoint 4 — LLM-as-a-Judge (Evaluation Engine) (5–6 dias)

**Objetivo:** avaliar cada `(question, expected_answer, retrieved_context, predicted_answer)` em 3 dimensões usando LLM com prompts estruturados, com **redução de variância** rodando 3× e tirando média.

**Entrega:**
- `python scripts/evaluate_run.py --run data/runs/exp_001/run_results.json`
- Gera `data/runs/exp_001/metrics.json` com scores agregados + raw

#### Métricas

| Métrica | O que mede | Inputs do Judge |
|---|---|---|
| **Faithfulness** | A resposta está ancorada no contexto recuperado? | `retrieved_context` + `predicted_answer` |
| **Answer Relevancy** | A resposta responde à pergunta feita? | `question` + `predicted_answer` |
| **Context Recall** | O contexto recuperado continha a info necessária? | `expected_answer` + `retrieved_context` |

#### Componentes

**`src/cyber_assessment/evaluation/judge.py`**
```python
JUDGE_PROMPTS = {
    "faithfulness": """Você é um avaliador rigoroso de sistemas RAG.

Avalie se a RESPOSTA é fiel ao CONTEXTO. Resposta fiel = todas as
afirmações da resposta podem ser verificadas no contexto.

Contexto:
{retrieved_context}

Resposta:
{predicted_answer}

Escala (1-5):
1 = A resposta contradiz ou ignora completamente o contexto
2 = A maior parte da resposta não tem suporte no contexto
3 = Alguma afirmação tem suporte, outras são extrapolações
4 = Quase toda a resposta tem suporte; uma extrapolação menor
5 = A resposta é completamente derivada do contexto

Responda APENAS em JSON válido:
{{"score": <int 1-5>, "reasoning": "<1-2 frases>"}}
""",
    "answer_relevancy": "...",
    "context_recall": "...",
}

@dataclass
class JudgeScore:
    metric: str
    score: int          # 1-5
    reasoning: str
    raw_response: str
    repetition: int     # 0, 1, 2

class Judge:
    def __init__(self, llm_client, model: str = "gpt-4o-mini",
                 n_repetitions: int = 3): ...

    def score(self, metric: str, **kwargs) -> list[JudgeScore]:
        """Roda n_repetitions, retorna lista (NÃO faz média aqui)."""
```

**`src/cyber_assessment/evaluation/metrics.py`**
```python
@dataclass
class MetricResult:
    metric: str
    mean: float
    std: float
    median: float
    distribution: dict[int, int]   # {1: 2, 2: 5, 3: 30, 4: 80, 5: 83}
    per_question: list[float]      # média das 3 reps por pergunta

@dataclass
class ExperimentMetrics:
    experiment_id: str
    faithfulness: MetricResult
    answer_relevancy: MetricResult
    context_recall: MetricResult
    overall: float                 # média ponderada simples
    n_questions: int
    judge_model: str
    total_judge_calls: int
    total_cost_usd: float

def aggregate(judge_scores: list[JudgeScore]) -> ExperimentMetrics: ...
```

**`src/cyber_assessment/evaluation/reporter.py`**
```python
def render_html_report(experiments: list[ExperimentMetrics],
                       output: Path) -> Path:
    """
    Gera HTML com:
    - Leaderboard (tabela ordenada por overall)
    - Distribuição de scores por métrica (sparkline ou bar)
    - Diff de configs side-by-side
    """
```

#### Decisões de design

- **`temperature=0.0` no judge.** Determinismo > criatividade.
- **JSON mode (`response_format={"type":"json_object"}`).** Reduz parsing failures de ~5% para <1%.
- **3 repetições, mediana > média.** A mediana é robusta a um outlier estranho do judge.
- **Reasoning é guardado.** Quando uma resposta tira nota baixa, o reasoning é diagnóstico — onde RAG falhou?
- **Score 1-5, não 0-1.** Escala ordinal é mais fácil para o LLM calibrar do que probabilidade contínua.
- **`overall` é média simples das 3.** Sem tuning de pesos no MVP — é arbitrário e dá falsa precisão.

#### Critérios de aceite

- [ ] Para `run_results.json` com 200 QA pairs → 3 métricas × 3 reps × 200 = 1800 chamadas judge, completa em <30min
- [ ] Cada `JudgeScore` tem `score ∈ {1,2,3,4,5}` (validado)
- [ ] `metrics.json` tem `mean`, `std`, `median`, `distribution` por métrica
- [ ] **Teste unitário com LLM mockado** valida parsing de JSON válido + JSON malformado (deve marcar repetição como `failed` e seguir)
- [ ] Quando o judge retorna `score: 6` (fora do range), parsing rejeita

#### Riscos & armadilhas

- **JSON malformado** apesar do JSON mode. ~1% acontece. → fallback: regex extrai o `score`. Se falhar, marca a repetição como inválida e usa as 2 restantes.
- **Judge bias para scores altos** (4-5). Conhecido. Não corrija agora — registre no relatório que isso é uma limitação.
- **Custo explode com `n_repetitions=3`.** Para iteração rápida em dev, suporte `--repetitions 1`. Em release, 3.
- **Context Recall é a métrica mais ruidosa** — depende do judge inferir se "expected_answer ⊆ context". Considere usar `gpt-4o` (não mini) só para essa métrica se ruído for alto.

---

### 🔵 Checkpoint 5 — MLflow + Relatório Comparativo (3–4 dias)

**Objetivo:** fechar o ciclo de MLOps. Sem isso, você tem JSONs espalhados; com isso, tem leaderboard navegável.

**Entrega:**
- `mlflow ui` mostra todos os experimentos com params + métricas
- Comparação side-by-side via UI nativa do MLflow
- `reports/comparison.html` gerado automaticamente, linkado como artifact

#### Componentes

**`src/cyber_assessment/tracking/mlflow_logger.py`**
```python
def log_experiment(config: ExperimentConfig,
                   metrics: ExperimentMetrics,
                   run_results_path: Path,
                   config_path: Path) -> str:
    """
    Cria um run no MLflow e retorna run_id.

    Params logados:
      - chunking.chunk_size
      - chunking.chunk_overlap
      - embedding.model
      - retrieval.top_k
      - retrieval.score_threshold
      - generation.model
      - generation.temperature

    Métricas logadas:
      - faithfulness_mean, faithfulness_std
      - answer_relevancy_mean, answer_relevancy_std
      - context_recall_mean, context_recall_std
      - overall_score
      - judge_total_cost_usd
      - run_total_cost_usd
      - n_questions
      - mean_latency_ms

    Artifacts:
      - configs/exp_001.yaml
      - run_results.json
      - metrics.json
      - reports/comparison.html (atualizado a cada run)

    Tags:
      - experiment_id, corpus_hash, benchmark_version, judge_model
    """
```

#### Estrutura do relatório HTML

```
reports/comparison.html
├── Header: "RAG Eval Lab — Comparison Report"
├── Leaderboard (tabela)
│   | exp_id | chunk | top_k | gen_model | faith | relev | recall | overall |
│   |--------|-------|-------|-----------|-------|-------|--------|---------|
│   | 003    | 128   | 10    | gpt-4o-mi | 4.42  | 4.61  | 4.20   | 4.41    |
│   | 001    | 256   | 5     | gpt-4o-mi | 4.30  | 4.58  | 4.10   | 4.33    |
├── Distribuição por métrica (bar charts inline SVG)
├── Diff de configs (3 melhores side-by-side)
└── Worst-case examples (5 piores QA pairs com reasoning do judge)
```

#### Decisões de design

- **Tudo num único experimento MLflow.** `experiment_name = "cyber_assessment_rag_eval"`. Cada YAML run vira um *run* dentro dele. Permite filtros nativos.
- **HTML com `jinja2`, não framework.** Sem React, sem JS pesado. Recrutador abre offline.
- **"Worst-case examples" no relatório.** Insight > números. Mostra que o autor entende o sistema.
- **Custo é métrica.** Quem analisar o leaderboard precisa ver que `gpt-4o` ganha 5% mas custa 10×.

#### Critérios de aceite

- [ ] Após 4 runs, `mlflow ui` lista 4 runs com todos os params/métricas
- [ ] `mlflow ui` permite selecionar 2 runs e ver diff lado a lado
- [ ] `reports/comparison.html` abre offline, tem leaderboard ordenado por `overall`
- [ ] Cada run no MLflow tem `run_results.json` e `config.yaml` como artifacts
- [ ] Tag `corpus_hash` permite filtrar runs feitos no mesmo corpus

#### Riscos & armadilhas

- **MLflow tracking URI default = local file.** Documentar no README, não assumir que o usuário sabe.
- **Re-rodar mesmo `experiment_id`** cria run novo no MLflow (não substitui). Decisão: aceitar e ordenar por timestamp. Avisar no log.
- **`mlflow.log_artifact` em arquivos grandes (>50MB)** trava UI. Limitar `run_results.json` a 50 MB ou comprimir.

---

### 🔵 Checkpoint 6 — Teste Ponta-a-Ponta + README como Produto (2–3 dias)

**Objetivo:** comunicação de engenharia. Um projeto sem README forte não existe para recrutadores.

**Entrega:**
- 4 experimentos rodados, números reais no README
- README com: diagrama, quick start (3 comandos), tabela de resultados, seção "Insights"
- Smoke test E2E que roda em CI ou localmente em <5min

#### Estrutura do README

```markdown
# RAG Evaluation Lab

> Plataforma de experimentação para pipelines RAG com avaliação automática (LLM-as-a-Judge), tracking MLflow e configuração declarativa em YAML.

## Por que este projeto existe
[2-3 parágrafos: problema → abordagem → o que se aprende usando isso]

## Arquitetura
![Pipeline](docs/pipeline.svg)
[Diagrama dos 4 subsistemas; pode ser ASCII se não houver figura]

## Quick Start (3 comandos)
\`\`\`bash
# 1. Setup
cp .env.example .env  # preencher OPENAI_API_KEY, PINECONE_API_KEY
pip install -e .

# 2. Pipeline completo
python scripts/run_full_pipeline.py --config configs/exp_001.yaml

# 3. Visualizar
mlflow ui
\`\`\`

## Resultados (corpus: relatorio_2024.pdf, 200 QA pairs)

| exp_id | chunk_size | top_k | gen_model    | Faith | Relev | Recall | Overall | Cost |
|--------|------------|-------|--------------|-------|-------|--------|---------|------|
| 003    | 128        | 10    | gpt-4o-mini  | 4.42  | 4.61  | 4.20   | **4.41** | $1.85 |
| 001    | 256        | 5     | gpt-4o-mini  | 4.30  | 4.58  | 4.10   | 4.33    | $1.20 |
| 002    | 512        | 5     | gpt-4o-mini  | 4.18  | 4.55  | 3.95   | 4.23    | $1.15 |
| 004    | 256        | 3     | gpt-4o-mini  | 4.35  | 4.52  | 3.78   | 4.22    | $0.95 |

## Insights

### O que aprendi
- **`top_k` importa mais do que `chunk_size`** neste corpus. Aumentar para 10 melhorou recall em 0.42 sem degradar faithfulness.
- **`chunk_size=512` é pior**, contra a intuição. Hipótese: chunks longos diluem a semântica do embedding.
- **Custo de `top_k=10` é só 50% mais alto**: prompts maiores, mas mesmas chamadas LLM. Vale a pena.

### Limitações conhecidas
- Judge é gpt-4o-mini, conhecido por bias positivo. Faithfulness real provavelmente é ~0.3 menor.
- Benchmark gerado por LLM tem ~10% de QA pairs com expected_answer questionável. Validação humana de amostra é recomendada.

## Estrutura
[árvore do repo]

## Tecnologias
[lista da §2 da arquitetura]

## Roadmap
- [ ] Suporte a re-ranking (cross-encoder)
- [ ] Avaliação RAGAS oficial (comparar com nosso judge)
- [ ] Async runner para paralelizar Checkpoint 3
```

#### Tarefas

- [ ] Rodar `exp_001` (chunk=256, top_k=5) — baseline
- [ ] Rodar `exp_002` (chunk=512, top_k=5)
- [ ] Rodar `exp_003` (chunk=128, top_k=10)
- [ ] Rodar `exp_004` (chunk=256, top_k=3)
- [ ] `scripts/run_full_pipeline.py` que faz ingest → run → eval → mlflow log num único comando
- [ ] Smoke test: `pytest tests/test_e2e.py` com mini corpus de 2 páginas, 5 QA pairs
- [ ] Diagrama da arquitetura (reusar o ASCII do Checkpoint 0 ou exportar SVG)
- [ ] Seção Insights com **3+ aprendizados específicos e justificados**
- [ ] Limpeza: remover prints, dead code, TODOs

#### Critérios de aceite

- [ ] `git clone` + 3 comandos do Quick Start funciona em máquina limpa (testar em VM ou container)
- [ ] README aberto no GitHub renderiza tabela e diagrama corretamente
- [ ] Seção Insights tem números reais, não placeholders
- [ ] `pytest` passa sem rede (mocks) em <30s
- [ ] Smoke E2E passa com rede em <5min e custo <$0.50

---

## 5. Cross-cutting Concerns

### Secrets & Config
- `.env` localmente, var de ambiente em CI
- **Nunca** colocar API keys no YAML
- `.env` no `.gitignore`; `.env.example` no repo

### Logging
- `rich` ou `structlog` com nível configurável via `LOG_LEVEL`
- Cada chamada LLM loga: modelo, n_tokens_in, n_tokens_out, latência, custo estimado
- Progress bars (`rich.progress`) em loops longos (ingestão, geração QA, judge)

### Custo
- `tiktoken` para estimar tokens **antes** da call
- Toda chamada → log de custo acumulado por experimento
- Cache de embeddings em parquet (Checkpoint 1)
- Smoke test usa mini-corpus (2 págs, 5 QAs) — custo <$0.10

### Determinismo & reprodutibilidade
- `temperature=0.0` no answerer e no judge
- `seed` parameter onde a API suportar
- `corpus_hash` no benchmark e nos artifacts MLflow
- Runs versionados por timestamp + experiment_id

### Erros transitórios
- `tenacity.retry` com backoff exponencial em todas as chamadas externas
- Max 3 retries; depois falha alto, não silenciosa
- Pinecone tem rate limits — batch upsert de 100 vetores por chamada

### Testes
| Nível | Cobertura | Velocidade |
|---|---|---|
| Unit | `chunker`, `validator`, `config_loader`, `judge_parsing` | <5s, sem rede |
| Integration | `runner` com Pinecone fake, `mlflow_logger` com tmpdir | <30s |
| E2E smoke | corpus de 2 págs, 5 QAs, 1 experimento, full pipeline | <5min, custo <$0.50 |

### Idempotência
- Ingestão: `--rebuild` vs falhar em namespace existente
- Run: falha se `run_results.json` existe; `--force` sobrescreve
- Evaluation: idem
- MLflow: cria run novo (intencional — histórico)

---

## 6. Ordem de Execução & Dependências

```
CP0 (bootstrap)
   │
   ▼
CP1 (ingestion) ◄─────────────┐
   │                           │
   │      ┌─ depende do corpus, não da ingestão
   ▼      │
CP2 (QA generation)            │
   │                           │
   ▼                           │
CP3 (RAG runner) ──────────────┘  (precisa do índice + benchmark)
   │
   ▼
CP4 (evaluation)
   │
   ▼
CP5 (MLflow + report)
   │
   ▼
CP6 (E2E + README)
```

**Paralelismo possível (se >1 dev):**
- CP1 e CP2 em paralelo (ambos só dependem do corpus)
- CP4 pode começar enquanto CP3 finaliza (sobre `run_results.json` parcial)

**Sequência recomendada para 1 dev:** CP0 → CP1 → CP2 → CP3 → CP4 → CP5 → CP6.

---

## 7. Estimativa de Cronograma

| Checkpoint | Estimativa | Acumulado |
|---|---|---|
| CP0 — Bootstrap | 0.5d | 0.5d |
| CP1 — Ingestion | 3–4d | 4.5d |
| CP2 — QA Generation | 4–5d | 9.5d |
| CP3 — RAG Runner | 4–5d | 14.5d |
| CP4 — Evaluation | 5–6d | 20.5d |
| CP5 — MLflow + Report | 3–4d | 24.5d |
| CP6 — E2E + README | 2–3d | 27.5d |
| **Total** | | **~22–28 dias úteis** |

Buffer de 20% recomendado: **~33 dias úteis** (≈ 6.5 semanas).

---

## 8. Definition of Done global

O projeto está "pronto" quando:

1. ✅ `git clone <repo> && cp .env.example .env && pip install -e . && python scripts/run_full_pipeline.py --config configs/exp_001.yaml && mlflow ui` funciona em máquina nova.
2. ✅ README tem **tabela de resultados com 4 experimentos reais** e **3+ insights específicos** sobre o comportamento de RAG no corpus.
3. ✅ `pytest` passa sem rede em <30s.
4. ✅ Smoke E2E passa com rede em <5min, custo <$0.50.
5. ✅ MLflow UI abre, lista runs, permite comparar.
6. ✅ `reports/comparison.html` abre offline e é navegável.
7. ✅ Nenhum segredo no histórico do git (`git log --all -p | grep -i "sk-"` retorna vazio).
8. ✅ Os 6 critérios de aceite de cada Checkpoint estão tickados.

---

## 9. O que está fora de escopo (intencional)

- **Re-ranking** (cross-encoder, ColBERT). Adiciona complexidade que não muda a história "como avalio RAG".
- **Hybrid search** (BM25 + vetor). Idem.
- **Avaliação humana sistemática.** Apenas amostragem de 20 QA pairs no Checkpoint 2.
- **Multi-corpus simultâneo.** O sistema é por-corpus; comparar entre corpora exige normalização.
- **Async / paralelismo.** MVP é sequencial. Adicionar depois se gargalo for evidente.
- **UI customizada.** MLflow UI é suficiente.
- **CI/CD com deploy.** Repo é projeto de portfólio, não serviço.

Esses itens entram no **Roadmap** do README como prova de que foram considerados e descartados conscientemente, não esquecidos.
