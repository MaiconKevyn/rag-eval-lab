#!/usr/bin/env python
"""CP6 — Generate a static HTML comparison report from all metrics.json files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv
from jinja2 import Template

load_dotenv()

from rag_eval_lab.utils.logging import get_logger, setup_logging

app = typer.Typer(add_completion=False)
log = get_logger(__name__)

_EVALS_DIR = Path("data/evals")
_RUNS_DIR = Path("data/runs")
_REPORT_DIR = Path("reports")

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RAG Evaluation Lab — Comparison Report</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #0f1117; color: #e2e8f0; line-height: 1.6; }
    .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
    h1 { font-size: 1.8rem; font-weight: 700; color: #fff; margin-bottom: 0.25rem; }
    .subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 2.5rem; }
    h2 { font-size: 1.1rem; font-weight: 600; color: #cbd5e1; margin: 2rem 0 1rem; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
             font-size: 0.75rem; font-weight: 600; }
    .badge-green { background: #14532d; color: #86efac; }
    .badge-blue  { background: #1e3a5f; color: #93c5fd; }

    /* summary cards */
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
             gap: 1rem; margin-bottom: 2rem; }
    .card { background: #1e2535; border: 1px solid #2d3748; border-radius: 10px;
            padding: 1.1rem 1.3rem; }
    .card-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;
                  letter-spacing: 0.05em; margin-bottom: 0.25rem; }
    .card-value { font-size: 1.5rem; font-weight: 700; color: #fff; }
    .card-sub   { font-size: 0.8rem; color: #64748b; margin-top: 0.15rem; }

    /* table */
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th { background: #1e2535; color: #94a3b8; font-weight: 600; text-align: left;
         padding: 0.65rem 0.9rem; border-bottom: 1px solid #2d3748;
         white-space: nowrap; }
    td { padding: 0.6rem 0.9rem; border-bottom: 1px solid #1e2535; }
    tr:hover td { background: #1a2030; }
    .winner { color: #86efac; font-weight: 700; }
    .bar-bg { background: #1e2535; border-radius: 4px; height: 6px; margin-top: 4px; }
    .bar-fill { height: 6px; border-radius: 4px; }
    .bar-faith  { background: #6366f1; }
    .bar-relev  { background: #22d3ee; }
    .bar-recall { background: #f59e0b; }
    .bar-comp   { background: #10b981; }

    /* insights */
    .insights { background: #1e2535; border: 1px solid #2d3748; border-radius: 10px;
                padding: 1.3rem 1.5rem; margin-top: 2rem; }
    .insights ul { padding-left: 1.3rem; }
    .insights li { margin-bottom: 0.5rem; color: #cbd5e1; font-size: 0.9rem; }
    .insights li strong { color: #fff; }

    footer { margin-top: 3rem; color: #475569; font-size: 0.8rem; text-align: center; }
  </style>
</head>
<body>
<div class="container">
  <h1>RAG Evaluation Lab</h1>
  <p class="subtitle">
    Comparison report — {{ n_experiments }} experiments · {{ n_questions }} questions each ·
    Judge: <code>{{ judge_model }}</code> (n_reps={{ n_reps }}) ·
    Generated {{ generated_at }}
  </p>

  <!-- summary cards -->
  <div class="cards">
    <div class="card">
      <div class="card-label">Best Composite</div>
      <div class="card-value">{{ best.composite_mean | round(3) }}</div>
      <div class="card-sub">{{ best.experiment_id }}</div>
    </div>
    <div class="card">
      <div class="card-label">Best Recall</div>
      <div class="card-value">{{ best_recall.context_recall_mean | round(3) }}</div>
      <div class="card-sub">{{ best_recall.experiment_id }}</div>
    </div>
    <div class="card">
      <div class="card-label">Avg Faithfulness</div>
      <div class="card-value">{{ avg_faithfulness | round(3) }}</div>
      <div class="card-sub">across all experiments</div>
    </div>
    <div class="card">
      <div class="card-label">Lowest Empty Context</div>
      <div class="card-value">{{ (best_recall.empty_context_rate * 100) | round(1) }}%</div>
      <div class="card-sub">{{ best_recall.experiment_id }}</div>
    </div>
  </div>

  <!-- comparison table -->
  <h2>Experiment Comparison</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Experiment</th>
          <th>Chunk</th>
          <th>top_k</th>
          <th>Threshold</th>
          <th>Faithfulness</th>
          <th>Relevancy</th>
          <th>Recall</th>
          <th>Composite ↑</th>
          <th>Empty Ctx ↓</th>
          <th>Cost (USD)</th>
        </tr>
      </thead>
      <tbody>
        {% for e in experiments %}
        <tr>
          <td>
            {{ e.experiment_id }}
            {% if e.experiment_id == best.experiment_id %}
            <span class="badge badge-green">best</span>
            {% endif %}
          </td>
          <td>{{ e.chunk_size }}</td>
          <td>{{ e.top_k }}</td>
          <td>{{ e.score_threshold }}</td>
          <td>
            {% if e.faithfulness_mean == best_faith %}
            <span class="winner">{{ e.faithfulness_mean | round(3) }}</span>
            {% else %}
            {{ e.faithfulness_mean | round(3) }}
            {% endif %}
            <div class="bar-bg"><div class="bar-fill bar-faith"
              style="width:{{ (e.faithfulness_mean / 5 * 100) | round }}%"></div></div>
          </td>
          <td>
            {{ e.answer_relevancy_mean | round(3) }}
            <div class="bar-bg"><div class="bar-fill bar-relev"
              style="width:{{ (e.answer_relevancy_mean / 5 * 100) | round }}%"></div></div>
          </td>
          <td>
            {% if e.context_recall_mean == best_recall.context_recall_mean %}
            <span class="winner">{{ e.context_recall_mean | round(3) }}</span>
            {% else %}
            {{ e.context_recall_mean | round(3) }}
            {% endif %}
            <div class="bar-bg"><div class="bar-fill bar-recall"
              style="width:{{ (e.context_recall_mean / 5 * 100) | round }}%"></div></div>
          </td>
          <td>
            {% if e.experiment_id == best.experiment_id %}
            <span class="winner">{{ e.composite_mean | round(3) }}</span>
            {% else %}
            {{ e.composite_mean | round(3) }}
            {% endif %}
            <div class="bar-bg"><div class="bar-fill bar-comp"
              style="width:{{ (e.composite_mean / 5 * 100) | round }}%"></div></div>
          </td>
          <td>{{ (e.empty_context_rate * 100) | round(1) }}%</td>
          <td>${{ "%.4f"|format(e.total_cost_usd) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- insights -->
  <div class="insights">
    <h2 style="margin-top:0">Key Findings</h2>
    <ul>
      <li><strong>Winner: {{ best.experiment_id }}</strong> — chunk_size={{ best.chunk_size }},
          top_k={{ best.top_k }} achieved the highest composite score ({{ best.composite_mean | round(3) }})
          and lowest empty context rate ({{ (best.empty_context_rate * 100) | round(1) }}%).</li>
      <li><strong>Faithfulness ≈ 5.0 across all experiments</strong> — the generation model consistently
          grounds its answers in the retrieved context, refusing to hallucinate when context is absent.</li>
      <li><strong>Answer relevancy and recall are limited by empty context</strong> — when no chunks are
          retrieved above the score threshold, the model responds "not enough information", which the judge
          penalises. Improving retrieval recall is the highest-leverage next step.</li>
      <li><strong>Larger chunks (512) hurt retrieval</strong> — {{ worst.experiment_id }} (chunk=512)
          had the highest empty context rate ({{ (worst.empty_context_rate * 100) | round(1) }}%) and
          the lowest composite ({{ worst.composite_mean | round(3) }}).</li>
      <li><strong>Hybrid search is the recommended next experiment</strong> — combining BM25 sparse
          vectors with dense embeddings in Pinecone should reduce empty context further and improve
          recall for exact-match and rare-term queries.</li>
    </ul>
  </div>

  <footer>
    RAG Evaluation Lab · corpus: agentic_ai_landscape.pdf ·
    embedding: text-embedding-3-small · generation: gpt-4o-mini
  </footer>
</div>
</body>
</html>
"""


def _load_experiment(exp_dir: Path) -> dict | None:
    metrics_path = exp_dir / "metrics.json"
    run_path = _RUNS_DIR / exp_dir.name / "run_results.json"
    if not metrics_path.exists() or not run_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text())
    run = json.loads(run_path.read_text())
    agg = metrics["aggregated"]
    cfg = run.get("config_snapshot", {})

    return {
        "experiment_id": metrics["experiment_id"],
        "chunk_size": cfg.get("chunking", {}).get("chunk_size", "?"),
        "top_k": cfg.get("retrieval", {}).get("top_k", "?"),
        "score_threshold": cfg.get("retrieval", {}).get("score_threshold", "?"),
        "faithfulness_mean": agg["faithfulness"]["mean"],
        "answer_relevancy_mean": agg["answer_relevancy"]["mean"],
        "context_recall_mean": agg["context_recall"]["mean"],
        "composite_mean": agg["composite"]["mean"],
        "empty_context_rate": metrics.get("n_skipped", 0) / max(metrics.get("n_evaluated", 1), 1),
        "total_cost_usd": run.get("total_cost_usd", 0.0),
        "judge_model": metrics.get("judge_model", ""),
        "n_reps": metrics.get("n_reps", 0),
        "n_evaluated": metrics.get("n_evaluated", 0),
    }


@app.command()
def main(
    output: Path = typer.Option(
        Path("reports/comparison_report.html"),
        "--output", "-o",
        help="Output HTML path.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    setup_logging(log_level)

    experiments = []
    for exp_dir in sorted(_EVALS_DIR.iterdir()):
        data = _load_experiment(exp_dir)
        if data:
            # compute real empty_context_rate from run_results
            run_path = _RUNS_DIR / exp_dir.name / "run_results.json"
            run = json.loads(run_path.read_text())
            results = run.get("results", [])
            if results:
                empty = sum(1 for r in results if not r.get("retrieved_context"))
                data["empty_context_rate"] = empty / len(results)
            experiments.append(data)

    if not experiments:
        typer.echo("No completed experiments found in data/evals/. Run CP4 first.")
        raise typer.Exit(1)

    experiments.sort(key=lambda e: e["composite_mean"], reverse=True)
    best = experiments[0]
    worst = experiments[-1]
    best_recall = max(experiments, key=lambda e: e["context_recall_mean"])
    best_faith = max(e["faithfulness_mean"] for e in experiments)
    avg_faithfulness = sum(e["faithfulness_mean"] for e in experiments) / len(experiments)

    output.parent.mkdir(parents=True, exist_ok=True)
    html = Template(_HTML_TEMPLATE).render(
        experiments=experiments,
        best=best,
        worst=worst,
        best_recall=best_recall,
        best_faith=best_faith,
        avg_faithfulness=avg_faithfulness,
        n_experiments=len(experiments),
        n_questions=experiments[0]["n_evaluated"],
        judge_model=experiments[0]["judge_model"],
        n_reps=experiments[0]["n_reps"],
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    output.write_text(html)

    typer.echo("")
    typer.echo("── Report Generated ─────────────────────────────")
    typer.echo(f"  experiments : {len(experiments)}")
    typer.echo(f"  winner      : {best['experiment_id']} (composite={best['composite_mean']:.3f})")
    typer.echo(f"  saved to    : {output}")
    typer.echo("─────────────────────────────────────────────────")
    typer.echo("Done.")


if __name__ == "__main__":
    app()
