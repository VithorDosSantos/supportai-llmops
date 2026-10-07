"""Carregamento e validação da base de conhecimento (Markdown + front-matter YAML).

A base é o "contrato" do RAG (Fase 3): cada documento tem `id` estável (usado
nas citações), `title` e `category` (alinhada às filas da triagem, o que permite
filtrar o retrieval pela categoria prevista). Validar aqui evita descobrir um
front-matter quebrado só na indexação.
"""

import re
from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from supportai.config import PROJECT_ROOT

DEFAULT_KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"

_FRONT_MATTER = re.compile(r"\A---\n(?P<meta>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)
# Referências cruzadas no texto: (see "Refund Policy"), "Cancelling a Subscription"...
_CROSS_REF = re.compile(r'(?:see|follow)\s+"([^"]+)"', re.IGNORECASE)


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^kb-[a-z0-9-]+$")
    title: str = Field(min_length=3)
    category: str
    body: str = Field(min_length=50)
    path: Path

    @property
    def cross_references(self) -> list[str]:
        return _CROSS_REF.findall(self.body)


class KnowledgeBaseError(ValueError):
    """Documento mal formado ou base inconsistente."""


def parse_document(path: Path) -> KnowledgeDocument:
    match = _FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    if match is None:
        raise KnowledgeBaseError(f"{path.name}: front-matter ausente ou mal formado")
    meta = yaml.safe_load(match["meta"])
    if not isinstance(meta, dict):
        raise KnowledgeBaseError(f"{path.name}: front-matter não é um mapeamento")
    return KnowledgeDocument(**meta, body=match["body"].strip(), path=path)


def validate_documents(docs: Iterable[KnowledgeDocument], categories: Iterable[str]) -> None:
    """Regras que envolvem a base inteira, não só um documento."""
    docs = list(docs)
    allowed = set(categories)
    errors: list[str] = []

    seen: dict[str, Path] = {}
    for doc in docs:
        if doc.id in seen:
            errors.append(f"id duplicado {doc.id}: {seen[doc.id].name} e {doc.path.name}")
        seen[doc.id] = doc.path
        if doc.category not in allowed:
            errors.append(f"{doc.path.name}: categoria desconhecida '{doc.category}'")
        if not doc.body.startswith(f"# {doc.title}\n"):
            errors.append(f"{doc.path.name}: o H1 deve ser igual ao title")

    titles = {doc.title for doc in docs}
    for doc in docs:
        for ref in doc.cross_references:
            if ref not in titles:
                errors.append(f"{doc.path.name}: referência a documento inexistente '{ref}'")

    if errors:
        raise KnowledgeBaseError("\n".join(errors))


def load_knowledge_base(
    kb_dir: Path = DEFAULT_KB_DIR, categories: Iterable[str] | None = None
) -> list[KnowledgeDocument]:
    """Carrega todos os `.md` (ordenados por nome, para ordem determinística)."""
    docs = [parse_document(p) for p in sorted(kb_dir.glob("*.md"))]
    if categories is not None:
        validate_documents(docs, categories)
    return docs
