from pathlib import Path


def load_chunks(script_path: str) -> list[str]:
    lines = Path(script_path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]
