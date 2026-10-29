"""Where an image dropped on SF EXR Load is copied: input/sf_exr/<name>/ with the file's exact name.
(Browse reads files where they are; only drag-and-drop copies, and only single images.)"""
import os
import re

from .sequence import IMAGE_EXTENSIONS as EXTENSIONS

ROOT = "sf_exr"
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


def folder_for(input_dir: str, name: str) -> tuple[str, str]:
    """(absolute folder, folder name) for one uploaded sequence; the name is reduced to safe characters."""
    root = os.path.realpath(os.path.join(input_dir, ROOT))
    clean = _UNSAFE.sub("_", name).strip(" .") or "upload"
    path = os.path.realpath(os.path.join(root, clean))
    if os.path.dirname(path) != root:
        raise ValueError(f"Bad upload folder name: {name!r}")
    return path, clean


def file_in(folder: str, filename: str) -> str:
    """Absolute path for an uploaded file inside folder. The name is kept as-is so frame numbers survive;
    anything that would leave the folder, or isn't an image format, is refused."""
    name = filename
    if not name or name in (".", "..") or "/" in name or "\\" in name or ":" in name:
        raise ValueError(f"Bad file name: {filename!r}")
    if os.path.splitext(name)[1].lower() not in EXTENSIONS:
        raise ValueError(f"{name}: not an image format SF EXR Load reads")
    path = os.path.realpath(os.path.join(folder, name))
    if os.path.dirname(path) != os.path.realpath(folder):
        raise ValueError(f"Bad file name: {filename!r}")
    return path


def clear(folder: str) -> int:
    """Remove the previous upload's images from folder (only image files directly inside it). Returns the count."""
    if not os.path.isdir(folder):
        return 0
    n = 0
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in EXTENSIONS | {".part"}:
            os.remove(p)
            n += 1
    return n
