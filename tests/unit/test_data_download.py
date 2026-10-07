import hashlib
from pathlib import Path

import pytest

from supportai.data.download import ChecksumMismatchError, sha256_of, verify_checksum


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    path.write_bytes(b"a,b\n1,2\n")
    assert sha256_of(path, chunk_size=3) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()


def test_verify_checksum_rejects_modified_file(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    path.write_bytes(b"original")
    expected = sha256_of(path)
    verify_checksum(path, expected)
    path.write_bytes(b"tampered")
    with pytest.raises(ChecksumMismatchError):
        verify_checksum(path, expected)
