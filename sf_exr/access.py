"""Which folders SF EXR may read, list or write.

ComfyUI reachable only from this machine (the default --listen 127.0.0.1): anywhere. Reachable from other
machines (--listen 0.0.0.0, a LAN address, ...): ComfyUI's input / output / temp folders plus the folders in the
SF_EXR_ROOTS environment variable (separated by ';' on Windows, ':' elsewhere), so a network client can't browse
or read the rest of the server's disks.
"""
import os

ENV = "SF_EXR_ROOTS"
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _listen() -> str:
    try:
        from comfy.cli_args import args
        return args.listen or "127.0.0.1"
    except Exception:  # outside ComfyUI (tests, scripts)
        return "127.0.0.1"


def local_only() -> bool:
    hosts = {h.strip().lower() for h in str(_listen()).split(",") if h.strip()}
    return bool(hosts) and hosts <= _LOCAL_HOSTS


def configured_roots() -> list[str]:
    return [os.path.realpath(os.path.expanduser(r.strip())) for r in os.environ.get(ENV, "").split(os.pathsep)
            if r.strip()]


def comfy_roots() -> list[tuple[str, str]]:
    try:
        import folder_paths
    except ImportError:
        return []
    return [("Input", folder_paths.get_input_directory()), ("Output", folder_paths.get_output_directory()),
            ("Temp", folder_paths.get_temp_directory())]


def roots() -> list[str]:
    return [os.path.realpath(p) for _, p in comfy_roots()] + configured_roots()


def allowed(path: str) -> bool:
    if local_only():
        return True
    p = os.path.normcase(os.path.realpath(path))
    for r in roots():
        r = os.path.normcase(r)
        try:
            if os.path.commonpath([p, r]) == r:
                return True
        except ValueError:  # different drives
            continue
    return False


def check(path: str) -> str:
    if not allowed(path):
        raise PermissionError(f"{path} is outside the folders SF EXR may use while ComfyUI is shared on the network. "
                              f"Add its folder to the {ENV} environment variable on the server.")
    return path
