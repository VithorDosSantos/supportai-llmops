from pathlib import Path

import pytest

from supportai.data.config import load_data_config
from supportai.data.knowledge_base import (
    KnowledgeBaseError,
    load_knowledge_base,
    parse_document,
    validate_documents,
)

CATEGORIES = load_data_config().filter.categories


def _write(tmp_path: Path, name: str, meta: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(f"---\n{meta}\n---\n\n{body}\n", encoding="utf-8")
    return path


BODY = "# Refund Policy\n\nRefunds go to the original payment method within 5 business days."


def test_project_knowledge_base_is_valid() -> None:
    """A base real: front-matter válido, ids únicos, categorias da triagem, refs existentes."""
    docs = load_knowledge_base(categories=CATEGORIES)
    assert 25 <= len(docs) <= 40
    # Toda categoria da triagem tem documentação (o RAG pode filtrar por ela).
    assert {d.category for d in docs} == set(CATEGORIES)


def test_parse_document_reads_front_matter(tmp_path: Path) -> None:
    path = _write(tmp_path, "r.md", "id: kb-refund\ntitle: Refund Policy\ncategory: x", BODY)
    doc = parse_document(path)
    assert (doc.id, doc.title, doc.category) == ("kb-refund", "Refund Policy", "x")
    assert doc.body.startswith("# Refund Policy")


def test_missing_front_matter_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text(BODY, encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="front-matter"):
        parse_document(path)


def test_validation_catches_duplicates_bad_category_and_broken_refs(tmp_path: Path) -> None:
    a = parse_document(
        _write(tmp_path, "a.md", "id: kb-a\ntitle: Refund Policy\ncategory: billing_payments", BODY)
    )
    b = parse_document(
        _write(
            tmp_path,
            "b.md",
            "id: kb-a\ntitle: Other\ncategory: nope",
            '# Other\n\nFor details see "Missing Doc". ' + "x" * 50,
        )
    )
    with pytest.raises(KnowledgeBaseError) as exc:
        validate_documents([a, b], CATEGORIES)
    message = str(exc.value)
    assert "id duplicado" in message
    assert "categoria desconhecida" in message
    assert "Missing Doc" in message
