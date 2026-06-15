from pathlib import Path

import pytest
from pydantic import ValidationError

from simple_agent.tools.file_reader import (
    MAX_FILE_BYTES,
    ReadTextFileArguments,
    read_text_file,
)


def test_reads_allowed_utf8_file(tmp_path):
    (tmp_path / "note.md").write_text("你好", encoding="utf-8")

    result = read_text_file(ReadTextFileArguments(path="note.md"), data_dir=tmp_path)

    assert result == {"path": "note.md", "content": "你好"}


@pytest.mark.parametrize("path", ["../secret.txt", "/tmp/secret.txt", ".env", "script.py"])
def test_rejects_disallowed_paths(tmp_path, path):
    with pytest.raises((ValueError, FileNotFoundError)):
        read_text_file(ReadTextFileArguments(path=path), data_dir=tmp_path)


def test_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "escape.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="超出"):
        read_text_file(ReadTextFileArguments(path="escape.txt"), data_dir=tmp_path)


def test_rejects_large_file(tmp_path):
    (tmp_path / "large.txt").write_bytes(b"x" * (MAX_FILE_BYTES + 1))

    with pytest.raises(ValueError, match="100 KB"):
        read_text_file(ReadTextFileArguments(path="large.txt"), data_dir=tmp_path)


def test_rejects_non_utf8_file(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="UTF-8"):
        read_text_file(ReadTextFileArguments(path="bad.txt"), data_dir=tmp_path)


def test_rejects_empty_path():
    with pytest.raises(ValidationError):
        ReadTextFileArguments(path="")


def test_nested_file(tmp_path):
    nested = tmp_path / "notes"
    nested.mkdir()
    (nested / "one.txt").write_text("one", encoding="utf-8")

    result = read_text_file(
        ReadTextFileArguments(path="notes/one.txt"),
        data_dir=Path(tmp_path),
    )

    assert result["path"] == "notes/one.txt"
