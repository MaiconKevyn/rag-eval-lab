from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rag_eval_lab.ingestion.embedder import Embedder
from rag_eval_lab.qa_generation.dataset import QAPair
from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

_TRIVIAL_PATTERNS = [
    # Artifact-referential — asking about the chunk/document as an object
    r"\bchunk\b",                                                              # any mention of "chunk"
    r"\bin (the|this) (document|text|passage|section)\b",                     # "in the document", "in this text"
    r"what (does|is|was) (the )?(document|text|passage) (say|mention|describe|cover|state|discuss)\b",
    r"what is mentioned in (the|this) (document|text|passage|section)\b",
    r"what (is|are|was|were) (shown|displayed|stated|listed|presented|given|written) in\b",
    r"(summarize|describe|explain) (the|this) (document|text|chunk|passage|section)\b",
    # Vague / low-signal questions
    r"what is the (main )?(topic|subject|idea|purpose|focus)\b",
    r"what (does|did) (this|the) (author|document|text|passage) (argue|claim|suggest|state|say)\b",
]
_TRIVIAL_RE = re.compile("|".join(_TRIVIAL_PATTERNS), re.IGNORECASE)


@dataclass
class DedupReport:
    n_original: int
    n_removed: int
    n_kept: int
    removed_pairs: list[str] = field(default_factory=list)  # qa_ids removed


class QAValidator:
    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = 0.92,
    ) -> None:
        self._embedder = embedder
        self._threshold = similarity_threshold

    def filter_trivial(self, pairs: list[QAPair]) -> list[QAPair]:
        kept = [p for p in pairs if not _TRIVIAL_RE.search(p.question)]
        removed = len(pairs) - len(kept)
        if removed:
            log.info("Filtered %d trivial question(s)", removed)
        return kept

    def deduplicate(self, pairs: list[QAPair]) -> tuple[list[QAPair], DedupReport]:
        """Greedy cosine-similarity dedup.

        Embeds all questions, then iterates: keep a pair if its similarity to
        every already-kept pair is below the threshold.
        """
        if len(pairs) <= 1:
            return pairs, DedupReport(len(pairs), 0, len(pairs))

        questions = [p.question for p in pairs]
        log.info("Embedding %d questions for dedup…", len(questions))
        vectors = np.array(self._embedder.embed(questions), dtype=np.float32)

        kept_indices: list[int] = [0]
        removed_ids: list[str] = []

        for i in range(1, len(pairs)):
            kept_vecs = vectors[kept_indices]
            sims = cosine_similarity(vectors[i : i + 1], kept_vecs)[0]
            if sims.max() >= self._threshold:
                removed_ids.append(pairs[i].qa_id)
                log.debug(
                    "Dedup: removed '%s' (max_sim=%.3f to kept)",
                    pairs[i].question[:60],
                    sims.max(),
                )
            else:
                kept_indices.append(i)

        kept = [pairs[i] for i in kept_indices]
        report = DedupReport(
            n_original=len(pairs),
            n_removed=len(removed_ids),
            n_kept=len(kept),
            removed_pairs=removed_ids,
        )
        log.info(
            "Dedup: %d → %d pairs (removed %d duplicates, threshold=%.2f)",
            report.n_original,
            report.n_kept,
            report.n_removed,
            self._threshold,
        )
        return kept, report
