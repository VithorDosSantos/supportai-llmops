"""Download do dataset bruto do Hugging Face, com verificação de integridade.

Revisão fixada + SHA-256: se o autor atualizar o arquivo no HF, o pipeline
falha em vez de treinar silenciosamente em dados diferentes dos documentados.

Uso: `python -m supportai.data.download`
"""

import hashlib
import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from supportai.data.config import DataConfig, load_data_config

logger = logging.getLogger(__name__)


class ChecksumMismatchError(RuntimeError):
    """O arquivo baixado não é o esperado pela configuração."""


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ChecksumMismatchError(f"{path}: sha256 {actual} != esperado {expected}")


def download_raw(cfg: DataConfig) -> Path:
    """Baixa o CSV (com cache do HF) e copia para `paths.raw`."""
    target = cfg.paths.resolve().raw
    if target.exists() and sha256_of(target) == cfg.source.sha256:
        logger.info("arquivo já presente e íntegro: %s", target)
        return target

    cached = Path(
        hf_hub_download(
            repo_id=cfg.source.repo_id,
            filename=cfg.source.filename,
            revision=cfg.source.revision,
            repo_type="dataset",
        )
    )
    verify_checksum(cached, cfg.source.sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, target)
    logger.info("dataset salvo em %s", target)
    return target


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    download_raw(load_data_config())


if __name__ == "__main__":
    main()
