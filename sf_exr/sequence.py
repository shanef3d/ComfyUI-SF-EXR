"""Image sequence detection: one file, a frame-numbered sibling set, or an explicit #### / %04d / @ pattern."""
import bisect
import os
import re
from dataclasses import dataclass, field

from . import access

# Image formats OpenImageIO reads that SF EXR Load handles (browse lists only these)
IMAGE_EXTENSIONS = {".exr", ".sxr", ".dpx", ".cin", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".hdr", ".tga",
                    ".bmp", ".webp"}

# name.1001.exr, name_0001.png, name1001.tif — the last digit run before the extension is the frame
FRAME_RE = re.compile(r"^(?P<head>.*?)(?P<frame>\d+)(?P<tail>\.[A-Za-z0-9]+)$")
TOKEN_RE = re.compile(r"(#+|@+|%0?(\d*)d)")
# A "-" straight after one of these is a minus sign (shot.-010.exr is pre-roll frame -10); after anything
# else it is part of the name (shot-0010.exr is frame 10).
SIGN_AFTER = (".", "_")
# One item of a frame list: 15, -10, 1001-1100, -10--1, first-1050, 1001-last
FRAME_ITEM_RE = re.compile(r"^(-?\d+|first|last)(?:-(-?\d+|first|last))?$", re.IGNORECASE)


def parse_frames(spec: str, first: int, last: int, step: int = 1) -> list[int]:
    """Frames named by a frame list: '' = all, '0', '1001-1100', '15, 20', '1-10, 50-60', 'first-1050'.
    step thins every range (and the whole sequence when spec is empty); single frames are always kept.
    first/last are the sequence's own ends. Returns sorted, unique frames."""
    step = max(1, int(step))

    def end(token: str) -> int:
        t = token.lower()
        return first if t == "first" else last if t == "last" else int(token)

    frames: set[int] = set()
    for raw in (spec or "").split(","):
        item = raw.replace(" ", "")
        if not item:
            continue
        m = FRAME_ITEM_RE.match(item)
        if not m:
            raise ValueError(f"Can't read frames '{raw.strip()}'. Use frames and ranges separated by commas, "
                             "like 1001-1100, 15, 20 or first-1050.")
        a = end(m.group(1))
        if m.group(2) is None:
            frames.add(a)
            continue
        b = end(m.group(2))
        if b < a:
            raise ValueError(f"Frame range '{raw.strip()}' runs backwards; write it as {b}-{a}.")
        frames.update(range(a, b + 1, step))
    if not frames:
        frames.update(range(first, last + 1, step))
    return sorted(frames)


def _frame_digits(width: int, padded: bool, signed: bool) -> str:
    """Regex for a frame number. Padding counts the minus sign, like printf: frame -10 at %04d is -010."""
    if padded:
        pos, neg = rf"\d{{{width}}}", rf"-\d{{{max(width - 1, 1)}}}"
    else:
        pos, neg = r"[1-9]\d*|0", r"-[1-9]\d*"
    return f"(?:{pos}|{neg})" if signed else f"(?:{pos})"


def _signed(head: str) -> bool:
    return not head or head.endswith(SIGN_AFTER)


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

    def select(self, frames: str = "", step: int = 1, missing_frames: str = "error") -> "Sequence":
        """Restrict to a frame list (see parse_frames); empty keeps every frame from first to last."""
        if not self.is_sequence or self.frames[0] is None:
            return self
        return self.take(parse_frames(frames, self.frames[0], self.frames[-1], step), missing_frames)

    def take(self, wanted: list[int], missing_frames: str = "error") -> "Sequence":
        """These exact frames. Missing ones raise, or with 'hold' reuse the nearest earlier frame on disk
        (the first frame for gaps before it)."""
        by_frame = dict(zip(self.frames, self.paths))
        have = sorted(by_frame)
        paths, frames, missing = [], [], []
        for f in wanted:
            if f in by_frame:
                paths.append(by_frame[f])
                frames.append(f)
                continue
            missing.append(f)
            if missing_frames == "hold":
                i = bisect.bisect_left(have, f)
                paths.append(by_frame[have[i - 1] if i else have[0]])
                frames.append(f)
        if missing and missing_frames == "error":
            raise ValueError(f"Missing frames {_ranges(missing)} in {self.pattern}. Set missing_frames to 'hold' to reuse the previous frame.")
        if not paths:
            raise ValueError(f"No frames selected in {self.pattern}.")
        return Sequence(paths, frames, self.pattern, self.padding, missing)


    def chunk(self, size: int, index: int) -> "Sequence":
        """Frames index*size .. (index+1)*size of this selection; size 0 keeps everything."""
        if size <= 0:
            return self
        count = -(-len(self.paths) // size)
        if not 0 <= index < count:
            raise ValueError(f"Chunk {index} is past the end: {len(self.paths)} frames make chunks 0-{count - 1} "
                             f"of {size}. Reset chunk to 0.")
        a, b = index * size, (index + 1) * size
        return Sequence(self.paths[a:b], self.frames[a:b], self.pattern, self.padding, self.missing)

    def chunk_count(self, size: int) -> int:
        return 1 if size <= 0 else -(-len(self.paths) // size)


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
    """Absolute path, or relative to ComfyUI's input folder. Refused outside the allowed folders when ComfyUI
    is shared on the network (see access.py)."""
    path = os.path.expandvars(os.path.expanduser(path.strip().strip('"').strip("'")))
    if not os.path.isabs(path):
        try:
            import folder_paths
            path = os.path.join(folder_paths.get_input_directory(), path)
        except ImportError:
            path = os.path.abspath(path)
    return access.check(os.path.normpath(path))


def _split_name(name: str) -> tuple[str, str, str] | None:
    """(head, frame digits incl. any minus sign, tail) of a numbered file name, using the same sign rule as
    detect(); None for names without a frame number."""
    m = FRAME_RE.match(name)
    if not m:
        return None
    head, frame, tail = m.group("head"), m.group("frame"), m.group("tail")
    if head.endswith("-") and _signed(head[:-1]):
        head, frame = head[:-1], "-" + frame
    return head, frame, tail


def list_dir(folder: str, limit: int = 20_000) -> dict:
    """A folder as the browse dialog shows it: subfolders, numbered sequences collapsed to one pattern entry
    (name.####.exr with first/last/count/missing), and other image files. Only image formats are listed."""
    folder = resolve_path(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    dirs, groups, singles = [], {}, []
    with os.scandir(folder) as it:
        for i, entry in enumerate(it):
            if i >= limit:
                break
            name = entry.name
            if name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    dirs.append(name)
                    continue
            except OSError:
                continue
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            parts = _split_name(name)
            if parts:
                groups.setdefault((parts[0], parts[2].lower()), []).append((int(parts[1]), parts[1], name))
            else:
                singles.append(name)

    entries = [{"type": "dir", "name": d, "path": os.path.join(folder, d)} for d in sorted(dirs, key=str.lower)]
    files = [{"type": "file", "name": n, "path": os.path.join(folder, n)} for n in singles]
    for (head, _), members in groups.items():
        if len(members) == 1:
            n = members[0][2]
            files.append({"type": "file", "name": n, "path": os.path.join(folder, n)})
            continue
        members.sort()
        widths = {len(digits) for _, digits, _ in members}
        tail = os.path.splitext(members[0][2])[1]
        pattern = head + "#" * (widths.pop() if len(widths) == 1 else 1) + tail
        first, last = members[0][0], members[-1][0]
        files.append({"type": "seq", "name": pattern, "path": os.path.join(folder, pattern), "first": first,
                      "last": last, "count": len(members), "missing": last - first + 1 - len(members)})
    entries += sorted(files, key=lambda e: e["name"].lower())
    parent = os.path.dirname(folder)
    return {"dir": folder, "parent": parent if parent != folder and access.allowed(parent) else None,
            "entries": entries}


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
        if pad > 1:
            digits = _frame_digits(pad, True, _signed(head))
        else:
            digits = r"-?\d+" if _signed(head) else r"\d+"
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
    if head.endswith("-") and _signed(head[:-1]):
        head, frame = head[:-1], "-" + frame
    digits_only = frame.lstrip("-")
    padded = digits_only.startswith("0") and len(digits_only) > 1
    regex = re.compile(rf"^{re.escape(head)}(?P<frame>{_frame_digits(len(frame), padded, _signed(head))})"
                       rf"{re.escape(tail)}$", re.IGNORECASE)
    paths, frames = _collect(folder, regex)
    pad = len(frame) if padded else 0
    if len(paths) < 2:
        # A lone numbered frame keeps its number, so saves write shot.0050.exr rather than a counter
        return Sequence([path], [int(frame)], os.path.join(folder, head + "#" * len(frame) + tail), pad)
    full = set(range(frames[0], frames[-1] + 1))
    return Sequence(paths, frames, os.path.join(folder, head + "#" * len(frame) + tail), pad,
                    sorted(full - set(frames)))
