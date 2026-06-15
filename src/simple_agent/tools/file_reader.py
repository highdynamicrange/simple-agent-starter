from pathlib import Path

from pydantic import BaseModel, Field


class ReadTextFileArguments(BaseModel):
    path: str = Field(min_length=1, max_length=500)


ALLOWED_SUFFIXES = {".txt", ".md", ".json", ".csv", ".yaml", ".yml"}
MAX_FILE_BYTES = 100 * 1024


def read_text_file(
    arguments: ReadTextFileArguments,
    *,
    data_dir: Path,
) -> dict[str, str]:
    relative_path = Path(arguments.path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("只允许读取 data 目录内的相对路径。")
    if relative_path.name == ".env" or relative_path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("不允许读取该文件类型。")

    root = data_dir.resolve()
    candidate = (root / relative_path).resolve(strict=True)
    if candidate == root or root not in candidate.parents:
        raise ValueError("文件路径超出 data 目录。")
    if not candidate.is_file():
        raise ValueError("目标不是普通文件。")
    if candidate.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("文件超过 100 KB 限制。")

    try:
        content = candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("文件不是有效的 UTF-8 文本。") from exc

    return {
        "path": candidate.relative_to(root).as_posix(),
        "content": content,
    }
