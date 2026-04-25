"""Unit tests for the Chunker.

These run offline — no API calls.
"""

from __future__ import annotations

import pytest

from rag_eval_lab.ingestion.chunker import Chunk, Chunker


def _text(n: int, char: str = "a") -> str:
    return char * n


class TestChunkerSplit:
    def test_produces_expected_count(self) -> None:
        """1000-char text, chunk_size=200, overlap=20 → ~6 chunks."""
        chunker = Chunker(chunk_size=200, chunk_overlap=20)
        pages = [(1, _text(1000))]
        chunks = chunker.split(pages, source="test.pdf")
        # Effective step = 200-20 = 180. Expect ceil(1000/180) ~ 6 chunks.
        assert 5 <= len(chunks) <= 7, f"Expected ~6 chunks, got {len(chunks)}"

    def test_chunk_fields_are_populated(self) -> None:
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([(1, _text(300))], source="doc.pdf")
        for c in chunks:
            assert c.text
            assert c.source == "doc.pdf"
            assert c.page == 1
            assert c.chunk_id.startswith("doc_p1_")

    def test_empty_pages_are_skipped(self) -> None:
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        pages = [(1, ""), (2, "   "), (3, _text(200))]
        chunks = chunker.split(pages, source="scan.pdf")
        assert all(c.page == 3 for c in chunks)

    def test_overlap_present_between_consecutive_chunks(self) -> None:
        """Consecutive chunks must share the overlap region."""
        overlap = 20
        chunker = Chunker(chunk_size=100, chunk_overlap=overlap)
        # Use a text with distinct sequential chars so overlap is detectable.
        text = "".join(str(i % 10) for i in range(500))
        chunks = chunker.split([(1, text)], source="num.txt")
        assert len(chunks) >= 2
        # The tail of chunk[i] and the head of chunk[i+1] must overlap.
        for i in range(len(chunks) - 1):
            tail = chunks[i].text[-overlap:]
            head = chunks[i + 1].text[:overlap]
            assert tail == head, (
                f"Overlap mismatch at chunks {i}/{i+1}: tail={tail!r} head={head!r}"
            )

    def test_chunk_id_is_unique(self) -> None:
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        pages = [(1, _text(500)), (2, _text(500))]
        chunks = chunker.split(pages, source="multi.pdf")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_multi_page_source_attribution(self) -> None:
        chunker = Chunker(chunk_size=200, chunk_overlap=20)
        pages = [(1, _text(300)), (3, _text(300))]
        chunks = chunker.split(pages, source="pages.pdf")
        pages_seen = {c.page for c in chunks}
        assert pages_seen == {1, 3}

    def test_source_without_extension(self) -> None:
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([(1, _text(200))], source="myfile")
        assert chunks[0].chunk_id.startswith("myfile_p1_")
