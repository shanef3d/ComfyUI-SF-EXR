"""SF EXR: load, edit, mask and save multi-layer EXR and image sequences in ComfyUI."""
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .sf_exr import nodes_color, nodes_io, nodes_mask, nodes_merge

WEB_DIRECTORY = "./web"


class SFExrExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return nodes_io.NODES + nodes_color.NODES + nodes_mask.NODES + nodes_merge.NODES


async def comfy_entrypoint() -> SFExrExtension:
    try:
        from server import PromptServer

        from .sf_exr import routes
        routes.register(PromptServer.instance.routes)
    except Exception as e:  # nodes still work from the graph without the browser helpers
        import logging
        logging.getLogger("SF EXR").warning(f"SF EXR browser routes unavailable: {e}")
    return SFExrExtension()


__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
