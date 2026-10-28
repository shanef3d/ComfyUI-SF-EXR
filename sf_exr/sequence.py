"""Image sequence detection: one file, a frame-numbered sibling set, or an explicit #### / %04d / @ pattern."""
import os
import re
from dataclasses import dataclass, field

# name.1001.exr, name_0001.png, name1001.tif — the last digit run before the extension is the frame
FRAME_RE = re.compile(r"^(?P<head>.*?)(?P<frame>\d+)(?P<tail>\.[A-Za-z0-9]+)$")
TOKEN_RE = re.compile(r"(#+|@+|%0?(\d*)d)")


@dataclass
class Sequence:
    paths: list[str]
    frames: list[int | None]
    pattern: str
    padding: int = 0
    missing: list[int] = field(default_factory=list)

    @property
    def is_sequence(self) -> bool:
        return len(self.paths) > 1 or (self.frames and self.frames[0] is not None)

    def select(self, start: int = 0, end: int = 0, step: int = 1, missing_frames: str = "error") -> "Sequence":
        """Restrict to a frame range; start/end of 0 mean the sequence's own first/last frame."""
        if not self.is_sequence or self.frames[0] is None:
            return self
        lo = start or self.frames[0]
        hi = end or self.frames[-1]
        by_frame = dict(zip(self.frames, self.paths))
        paths, frames, missing = [], [], []
        last = None
        for f in range(lo, hi + 1, max(1, step)):
            if f in by_frame:
                last = by_frame[f]
                paths.append(last)
                frames.append(f)
                continue
            missing.append(f)
            if missing_frames == "hold":
                # a gap at the head of the range holds the next available frame instead
                src = last or next((by_frame[g] for g in sorted(by_frame) if g > f), None)
                if src:
                    paths.append(src)
                    frames.append(f)
        if missing and missing_frames == "error":
            raise ValueError(f"Missing frames {_ranges(missing)} in {self.pattern}. Set missing_frames to 'hold' to reuse the previous frame.")
        if not paths:
            raise ValueError(f"No frames between {lo} and {hi} in {self.pattern}.")
        return Sequence(paths, frames, self.pattern, self.padding, missing)


def _ranges(nums: list[int]) -> str:
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def resolve_path(path: str) -> str:
    """Absolute path, or relative to ComfyUI's input folder."""
    path = os.path.expandvars(os.path.expanduser(path.strip().strip('"').strip("'")))
    if not os.path.isabs(path):
        try:
            import folder_paths
            path = os.path.join(folder_paths.get_input_directory(), path)
        except ImportError:
            path = os.path.abspath(path)
    return os.path.normpath(path)


def _collect(folder: str, regex: re.Pattern) -> tuple[list[str], list[int]]:
    hits = []
    for name in os.listdir(folder):
        m = regex.match(name)
        if m:
            hits.append((int(m.group("frame")), os.path.join(folder, name)))
    hits.sort()
    return [p for _, p in hits], [f for f, _ in hits]


def detect(path: str, mode: str = "auto") -> Sequence:
    """Find the sequence a path belongs to. mode: 'auto' groups numbered siblings, 'single' never does."""
    path = resolve_path(path)
    folder, name = os.path.split(path)

    token = TOKEN_RE.search(name)
    if token:
        tok = token.group(1)
        pad = len(tok) if tok[0] in "#@" else int(token.group(2) or 0)
        head, tail = name[:token.start()], name[token.end():]
        digits = rf"\d{{{pad}}}" if pad > 1 else r"\d+"
        regex = re.compile(rf"^{re.escape(head)}(?P<frame>{digits}){re.escape(tail)}$")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        paths, frames = _collect(folder, regex)
        if not paths:
            raise FileNotFoundError(f"No files match {path}")
        return Sequence(paths, frames, os.path.join(folder, head + "#" * max(pad, 1) + tail), pad)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    m = FRAME_RE.match(name)
    if mode == "single" or not m:
        return Sequence([path], [None], path)

    head, frame, tail = m.group("head"), m.group("frame"), m.group("tail")
    padded = frame.startswith("0") and len(frame) > 1
    digits = rf"\d{{{len(frame)}}}" if padded else r"[1-9]\d*|0"
    regex = re.compile(rf"^{re.escape(head)}(?P<frame>{digits}){re.escape(tail)}$", re.IGNORECASE)
    paths, frames = _collect(folder, regex)
    if len(paths) < 2:
        return Sequence([path], [None], path)
    pad = len(frame) if padded else 0
    full = set(range(frames[0], frames[-1] + 1))
    return Sequence(paths, frames, os.path.join(folder, head + "#" * len(frame) + tail), pad,
                    sorted(full - set(frames)))
