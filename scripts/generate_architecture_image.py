#!/usr/bin/env python
"""Generate a polished architecture diagram image for the README using GPT Image 2."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from openai import OpenAI

from rag_eval_lab.utils.logging import get_logger, setup_logging

load_dotenv()

app = typer.Typer(add_completion=False)
log = get_logger(__name__)

DEFAULT_OUTPUT = Path("assets/architecture.png")

ARCHITECTURE_PROMPT = """\
Create a polished horizontal architecture diagram for a software project named "RAG Evaluation Lab".

The diagram should look like a clean, modern engineering architecture graphic for a GitHub README:
- white or very light background
- subtle blue and teal accents
- crisp typography
- vector-style boxes and arrows
- professional, minimal, technically credible
- no photorealism
- no decorative people or mascots
- output must be easy to read when embedded in a README

Use this exact pipeline and labels:

Title:
RAG Evaluation Lab - Pipeline

Flow from left to right:
1. PDFs / corpus
2. INGESTION
   subtitle: chunker + embedder
3. Pinecone (namespaces)
   bullets:
   - exp_001 -> vectors
   - exp_002 -> vectors
4. QA GENERATION
   subtitle: LLM generates Q&A
5. benchmark_dataset.json
   subtitle: versioned ground truth
6. YAML config
7. RAG RUNNER
   subtitle: retriever + generator
8. run_results.json
   subtitle: {Q, expected_A, ctx, predicted_A}
9. EVALUATION
   subtitle: LLM-as-a-Judge
10. metrics.json
    subtitle: faithfulness, relevancy, recall
11. MLflow Tracking
    subtitle: params + metrics + artifacts
12. UI + reports/comparison.html

Layout requirements:
- show arrows connecting the stages clearly
- keep the JSON and YAML artifacts visually distinct from processing stages
- group ingestion, QA generation, RAG runner, evaluation, and tracking as process blocks
- make Pinecone a storage/service block
- include enough spacing for readability
- keep all text in English
- produce a single self-contained diagram image

Style guidance:
- modern SaaS / systems architecture aesthetic
- rounded rectangles
- thin connector arrows
- consistent iconography if needed, but minimal
- prioritize legibility over artistic flair
"""


@app.command()
def main(
    output: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help="Where to save the generated image.",
    ),
    model: str = typer.Option("gpt-image-2", "--model", help="Image model to use."),
    size: str = typer.Option(
        "1536x1024",
        "--size",
        help="Image size. Common values: 1024x1024, 1536x1024, 1024x1536.",
    ),
    quality: str = typer.Option(
        "medium",
        "--quality",
        help="Image quality: low, medium, or high.",
    ),
    background: str = typer.Option(
        "opaque",
        "--background",
        help="Background mode: opaque, transparent, or auto.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
    prompt_only: bool = typer.Option(
        False,
        "--prompt-only",
        help="Print the prompt and exit without calling the API.",
    ),
) -> None:
    setup_logging(log_level)

    if prompt_only:
        typer.echo(ARCHITECTURE_PROMPT)
        raise typer.Exit(0)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        typer.echo("Error: OPENAI_API_KEY is not set.", err=True)
        raise typer.Exit(1)

    client = OpenAI(api_key=api_key)
    log.info("Generating architecture image with %s (%s, %s)", model, size, quality)

    response = client.images.generate(
        model=model,
        prompt=ARCHITECTURE_PROMPT,
        size=size,
        quality=quality,
        background=background,
        output_format="png",
    )

    image_b64 = response.data[0].b64_json
    if not image_b64:
        typer.echo("Error: image API returned no image payload.", err=True)
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(base64.b64decode(image_b64))

    typer.echo(f"Saved architecture image to {output}")


if __name__ == "__main__":
    app()
