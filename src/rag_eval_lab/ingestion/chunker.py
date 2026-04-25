from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    chunk_id: str       # "<source_stem>_p<page>_<idx>"
    text: str
    source: str         # filename
    page: int
    chunk_index: int


class Chunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str] | None = None,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, pages: list[tuple[int, str]], source: str) -> list[Chunk]:
        """Split (page_number, text) pairs into Chunk objects.

        Empty pages are skipped — they come from scanned PDFs with no OCR layer.
        """
        chunks: list[Chunk] = []
        stem = Path(source).stem

        for page_num, text in pages:
            if not text.strip():
                continue
            splits = self._splitter.split_text(text)
            for idx, split_text in enumerate(splits):
                chunks.append(
                    Chunk(
                        chunk_id=f"{stem}_p{page_num}_{idx}",
                        text=split_text,
                        source=source,
                        page=page_num,
                        chunk_index=idx,
                    )
                )

        return chunks
