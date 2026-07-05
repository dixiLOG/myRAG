from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import yaml

from myrag.models import CorpusDocument, CorpusDocumentMeta, DocType, corpus_version_for_documents, relative_posix

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
STYLE_RE = re.compile(r"<style.*?</style>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


def load_markdown_corpus(root: Path) -> list[CorpusDocument]:
    root = root.resolve()
    documents: list[CorpusDocument] = []
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(text)
        body = normalize_markdown(body)
        relative_path = relative_posix(path, root)
        meta = build_metadata(relative_path, frontmatter)
        documents.append(CorpusDocument(meta=meta, text=body))
    return documents


def split_frontmatter(text: str) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()
    frontmatter = yaml.safe_load(match.group(1)) or {}
    return frontmatter, text[match.end():].strip()


def normalize_markdown(text: str) -> str:
    text = STYLE_RE.sub("", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<div[^>]*>", "", text, flags=re.IGNORECASE)
    text = text.replace("</div>", "")
    text = text.replace("&nbsp;", " ")
    text = TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_metadata(relative_path: str, frontmatter: dict) -> CorpusDocumentMeta:
    path = Path(relative_path)
    title = str(frontmatter.get("title") or path.stem)
    tags = normalize_str_list(frontmatter.get("tags"))
    categories = normalize_str_list(frontmatter.get("categories"))
    authors = normalize_str_list(frontmatter.get("authors"))
    doc_type = detect_doc_type(relative_path, categories, tags)
    return CorpusDocumentMeta(
        doc_id=path.with_suffix("").as_posix(),
        title=title,
        source_path=path.as_posix(),
        doc_type=doc_type,
        tags=tags,
        categories=categories,
        created_at=string_or_none(frontmatter.get("date")),
        updated_at=string_or_none(frontmatter.get("updated_at") or frontmatter.get("date")),
        language="zh",
        visibility=str(frontmatter.get("visibility") or "public"),
        series=string_or_none(frontmatter.get("series")),
        authors=authors,
    )


def string_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def normalize_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def detect_doc_type(relative_path: str, categories: list[str], tags: list[str]) -> DocType:
    joined = " ".join([relative_path, *categories, *tags]).lower()
    if "book" in joined or "阅读" in joined or "读书" in joined:
        return "book_note"
    if "material" in joined or "build_your_web" in joined or "github" in joined or "教程" in joined:
        return "tutorial"
    if "essay" in joined or "blog" in joined or "豆腐脑" in joined:
        return "essay"
    if "long" in joined or "长文" in joined:
        return "longform"
    if "note" in joined or "环境" in joined or "随手记" in joined:
        return "note"
    return "unknown"


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = "Introduction"
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if buffer:
                body = "\n".join(buffer).strip()
                if body:
                    sections.append((current_heading, body))
                buffer = []
            current_heading = line.lstrip("#").strip() or "Section"
        else:
            buffer.append(line)
    if buffer:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_heading, body))
    return sections


def corpus_stats(documents: list[CorpusDocument]) -> dict[str, object]:
    by_type: dict[str, int] = {}
    for document in documents:
        by_type[document.meta.doc_type] = by_type.get(document.meta.doc_type, 0) + 1
    return {
        "documents": len(documents),
        "corpus_version": corpus_version_for_documents(documents),
        "doc_types": by_type,
    }
