from pathlib import Path

from myrag.corpus import corpus_stats, load_markdown_corpus


def test_load_markdown_corpus_extracts_metadata() -> None:
    root = Path(__file__).resolve().parents[1] / "examples" / "sample_corpus"
    documents = load_markdown_corpus(root)
    stats = corpus_stats(documents)
    assert len(documents) == 5
    assert stats["documents"] == 5
    assert any(document.meta.doc_id == "blog_essay" for document in documents)
    assert any(document.meta.doc_type == "tutorial" for document in documents)
