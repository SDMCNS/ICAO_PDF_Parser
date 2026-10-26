"""PDF parsing engine tailored for ICAO manuals and technical aviation documents."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF

from .config import ParserConfig
from .models import DocumentMetadata, PageData


class PDFParser:
    """Extracts text, structural hierarchy, and metadata from ICAO PDF documents."""

    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig()

    def parse_document(self, file_path: str | Path) -> Tuple[DocumentMetadata, List[PageData]]:
        """Parse an entire PDF file and return its DocumentMetadata and PageData list."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF document not found: {path}")

        doc = fitz.open(str(path))
        total_pages = len(doc)
        file_size = path.stat().st_size

        # Deduce title & doc number from filename and first pages
        doc_id, title, edition = self._extract_doc_info(path, doc)

        pages_data: List[PageData] = []
        current_chapter: Optional[str] = None

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1

            page_data, current_chapter = self._parse_page(
                page=page,
                page_num=page_num,
                total_pages=total_pages,
                current_chapter=current_chapter
            )
            pages_data.append(page_data)

        doc.close()

        # Count content pages
        indexed_pages = sum(1 for p in pages_data if p.page_type == "content" and len(p.clean_text) > 30)

        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=path.name,
            title=title,
            edition=edition,
            total_pages=total_pages,
            indexed_pages=indexed_pages,
            file_size_bytes=file_size,
            file_mtime=path.stat().st_mtime,
            source_path=str(path.resolve())
        )

        return metadata, pages_data

    def _extract_doc_info(self, path: Path, doc: fitz.Document) -> Tuple[str, str, Optional[str]]:
        """Identify standard ICAO document number, title, and edition."""
        filename = path.stem.lower()

        sample_text = ""
        for i in range(min(3, len(doc))):
            sample_text += doc[i].get_text() + "\n"

        if "9562" in filename or "9562" in sample_text:
            doc_id = "doc_9562"
            title = "ICAO Doc 9562: Airport Economics Manual"
            edition = "Fourth Edition, 2020"
        elif "9750" in filename or "9750" in sample_text:
            doc_id = "doc_9750"
            title = "ICAO Doc 9750: Global Air Navigation Plan for CNS/ATM Systems"
            edition = "Second Edition, 2002"
        else:
            doc_id = f"doc_{re.sub(r'[^a-zA-Z0-9_]', '_', filename)}"
            title = path.stem.replace("_", " ").title()
            edition = None

            match = re.search(r"Doc\s+(\d+)\s*\n+([^\n]+)", sample_text, re.IGNORECASE)
            if match:
                doc_id = f"doc_{match.group(1)}"
                title = f"ICAO Doc {match.group(1)}: {match.group(2).strip()}"

        return doc_id, title, edition

    def _parse_page(
        self,
        page: fitz.Page,
        page_num: int,
        total_pages: int,
        current_chapter: Optional[str]
    ) -> Tuple[PageData, Optional[str]]:
        """Parse a single PDF page with layout and header awareness."""
        raw_text = page.get_text()
        stripped_raw = raw_text.strip()

        if len(stripped_raw) < self.config.min_page_char_count:
            return PageData(
                page_number=page_num,
                raw_text=raw_text,
                clean_text="",
                page_type="blank",
                char_count=0
            ), current_chapter

        # Determine page type
        page_type = "content"
        if page_num <= 3:
            page_type = "cover"
        elif page_num <= 20 and self._is_front_matter(stripped_raw):
            page_type = "front_matter"

        # Extract blocks from page for paragraph boundaries
        # block tuple: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        text_blocks = [b[4].strip() for b in blocks if len(b) > 4 and b[4].strip()]

        running_header, detected_chapter, cleaned_paragraphs = self._process_blocks(
            text_blocks, current_chapter
        )

        detected_sections = self._find_sections(cleaned_paragraphs)
        clean_text = "\n\n".join(cleaned_paragraphs)

        page_data = PageData(
            page_number=page_num,
            raw_text=raw_text,
            clean_text=clean_text,
            page_type=page_type,
            running_header=running_header,
            detected_chapter=detected_chapter or current_chapter,
            detected_sections=detected_sections,
            char_count=len(clean_text)
        )

        return page_data, detected_chapter or current_chapter

    def _is_front_matter(self, text: str) -> bool:
        """Check if page belongs to front matter (TOC, amendments, glossary, foreword)."""
        lower = text.lower()
        front_keywords = [
            "table of contents",
            "amendments",
            "foreword",
            "glossary of terms",
            "list of acronyms",
            "publications;",
            "corrigenda"
        ]
        return any(kw in lower for kw in front_keywords)

    def _process_blocks(
        self,
        blocks: List[str],
        current_chapter: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        """Filter running headers and footers while extracting chapter tracking."""
        running_header: Optional[str] = None
        detected_chapter = current_chapter
        cleaned_blocks: List[str] = []

        re_chapter = re.compile(r"^(Chapter\s+\d+[\.\:]?\s*.*|PART\s+[IVXLCDM]+[\.\:]?\s*.*)", re.IGNORECASE)
        re_page_num = re.compile(r"^(\d+-\d+|\([ivxlcdm]+\)|\d+|[IVXLCDM]+-\d+-\d+)$", re.IGNORECASE)
        re_doc_title = re.compile(r"^(Airport Economics Manual|Global Air Navigation Plan.*)$", re.IGNORECASE)

        for idx, block in enumerate(blocks):
            # Clean non-breaking spaces and unicode replacement
            b_clean = block.replace("\xa0", " ").replace("\ufffd", "'").replace("\r\n", "\n")

            # Check if top block is a header
            first_line = b_clean.split("\n")[0].strip()
            if idx == 0:
                chap_match = re_chapter.match(first_line)
                if chap_match:
                    detected_chapter = chap_match.group(1).strip()
                    running_header = detected_chapter
                    if self.config.strip_running_headers:
                        # If block only contains header line or page number, skip
                        remaining_lines = [l.strip() for l in b_clean.split("\n")[1:] if l.strip() and not re_page_num.match(l.strip())]
                        if not remaining_lines:
                            continue
                        b_clean = "\n".join(remaining_lines)

                elif re_doc_title.match(first_line) or re_page_num.match(first_line):
                    if self.config.strip_running_headers:
                        remaining_lines = [l.strip() for l in b_clean.split("\n")[1:] if l.strip() and not re_page_num.match(l.strip())]
                        if not remaining_lines:
                            continue
                        b_clean = "\n".join(remaining_lines)

            # Check if bottom block is just a lone page number
            if idx == len(blocks) - 1 and re_page_num.match(b_clean.strip()):
                if self.config.strip_running_headers:
                    continue

            # Clean hyphenation within the block
            if self.config.clean_hyphenation:
                b_clean = re.sub(r"(\b\w+)-\s*\n\s*(\w+\b)", r"\1\2", b_clean)

            # Replace single internal newlines with space, preserving sentences
            lines = [l.strip() for l in b_clean.split("\n") if l.strip()]
            b_text = " ".join(lines)
            b_text = re.sub(r"[ \t]{2,}", " ", b_text)

            if len(b_text) > 10:
                cleaned_blocks.append(b_text)

        return running_header, detected_chapter, cleaned_blocks

    def _find_sections(self, paragraphs: List[str]) -> List[str]:
        """Detect section headers in extracted paragraphs."""
        sections = []
        section_pattern = re.compile(r"^(\d+\.\d+(\.\d+)?\s+[A-Z][^\.]{3,60})")
        for p in paragraphs:
            match = section_pattern.match(p.strip())
            if match:
                sections.append(match.group(1).strip())
        return sections
