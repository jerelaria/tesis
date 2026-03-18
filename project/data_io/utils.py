from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_image_paths(images_dir: str, extensions: list[str]) -> list[Path]:
    """Return all image paths in the given directory matching the given extensions."""
    folder = (_PROJECT_ROOT / "data" / "raw" / images_dir).resolve()
    paths = []
    for ext in extensions:
        paths.extend(folder.glob(f"*.{ext}"))
    return sorted(paths)