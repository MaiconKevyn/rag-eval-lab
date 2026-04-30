from __future__ import annotations

import json
import re

from pydantic import ValidationError
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from rag_eval_lab.ingestion.chunker import Chunk
from rag_eval_lab.qa_generation.dataset import QAPair
from rag_eval_lab.utils.llm_client import LLMClient
from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

QA_PROMPT = """\
Given the chunk below, generate {n} (question, expected_answer) pairs that test knowledge of the content.

Strict rules:
- The question must be answerable using ONLY the information in the provided chunk
- The expected answer must be faithful to the chunk, with no outside inference
- Vary the question type: factual, comparative, why, how
- If the chunk is too short or generic to produce {n} good questions, generate fewer
- Write both `question` and `expected_answer` in English only
- Do NOT reference the chunk or document as an artifact: never use phrases like
  "in the chunk", "in this chunk", "in the text", "in the passage", "in this document",
  "in the section", "the chunk says", "what is mentioned in", "what is shown in"
- Ask about the KNOWLEDGE contained in the chunk, not about the chunk itself
  BAD:  "What is the title of the chunk?"
  BAD:  "Which organization is named at the top of the document in the chunk?"
  GOOD: "What publication does the OECD release on AI conceptual foundations?"
  GOOD: "What role does memory play in agentic AI systems?"

Return ONLY valid JSON in this schema:
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

Chunk ({chunk_id}):
{chunk_text}
"""

# Regex fallback: extract score-like integer when JSON mode still fails.
_QA_PAIRS_RE = re.compile(r'"qa_pairs"\s*:\s*(\[.*?\])', re.DOTALL)


class QAGenerator:
    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "gpt-4o-mini",
        n_per_chunk: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self._client = llm_client
        self._model = model
        self._n = n_per_chunk
        self._temperature = temperature

    def generate_for_chunk(self, chunk: Chunk) -> list[QAPair]:
        """Call LLM for one chunk, parse JSON, return QAPairs.

        Parsing failures are logged and return an empty list — they must not
        propagate and kill the whole corpus generation run.
        """
        if len(chunk.text) < 80:
            log.debug("Skipping short chunk %s (%d chars)", chunk.chunk_id, len(chunk.text))
            return []

        prompt = QA_PROMPT.format(
            n=self._n,
            chunk_id=chunk.chunk_id,
            chunk_text=chunk.text,
        )
        try:
            result = self._client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                json_mode=True,
                temperature=self._temperature,
            )
            raw = result.text
        except Exception as exc:
            log.warning("LLM call failed for chunk %s: %s", chunk.chunk_id, exc)
            return []

        return self._parse(raw, chunk.chunk_id)

    def _parse(self, raw: str, chunk_id: str) -> list[QAPair]:
        try:
            data = json.loads(raw)
            pairs_data = data.get("qa_pairs", [])
        except json.JSONDecodeError:
            # Regex fallback: try to extract qa_pairs array.
            m = _QA_PAIRS_RE.search(raw)
            if not m:
                log.warning("Malformed JSON for chunk %s — skipping", chunk_id)
                return []
            try:
                pairs_data = json.loads(m.group(1))
            except json.JSONDecodeError:
                log.warning("Regex fallback also failed for chunk %s — skipping", chunk_id)
                return []

        pairs: list[QAPair] = []
        for item in pairs_data:
            # Normalise question_type to our Literal values.
            qt = _normalise_type(item.get("question_type", "factual"))
            item["question_type"] = qt
            item.setdefault("source_chunk_id", chunk_id)
            try:
                pairs.append(QAPair(**item))
            except (ValidationError, TypeError) as exc:
                log.debug("Invalid QA pair in chunk %s: %s", chunk_id, exc)
        return pairs

    def generate_for_corpus(
        self,
        chunks: list[Chunk],
        *,
        max_chunks: int | None = None,
    ) -> list[QAPair]:
        target = chunks[:max_chunks] if max_chunks else chunks
        all_pairs: list[QAPair] = []
        failed = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Generating QA pairs", total=len(target))
            for chunk in target:
                pairs = self.generate_for_chunk(chunk)
                if not pairs:
                    failed += 1
                all_pairs.extend(pairs)
                progress.advance(task)

        log.info(
            "Generated %d QA pairs from %d/%d chunks (%d failed/skipped)",
            len(all_pairs),
            len(target) - failed,
            len(target),
            failed,
        )
        return all_pairs


def _normalise_type(raw: str) -> str:
    """Map LLM free-form type strings to our Literal values."""
    r = raw.strip().lower()
    if r == "why":
        return "why"
    if r == "how":
        return "how"
    if r in ("comparative", "comparison"):
        return "comparative"
    return "factual"
