from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(path):
    root = Path(path)
    if root.is_file():
        return root.parent, [root]
    if not root.is_dir():
        raise FileNotFoundError("Input does not exist: %s" % root)
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError("No supported images under: %s" % root)
    return root, images


def safe_stem(path, root):
    relative = path.relative_to(root)
    parts = list(relative.parts)
    parts[-1] = Path(parts[-1]).stem
    return "__".join(parts)
