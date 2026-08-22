import importlib.util
import sys
from types import ModuleType


if importlib.util.find_spec("pycolmap") is None:
    # Depth Anything 3 eagerly imports its optional COLMAP exporter from api.py.
    # Mono depth training does not use that exporter, and pycolmap has no Linux
    # ARM64 wheel. A placeholder lets the API load; accessing any COLMAP type
    # still fails with a clear error.
    class _MissingPycolmap(ModuleType):
        def __getattr__(self, name: str):
            raise AttributeError(
                "pycolmap is required for Depth Anything 3 COLMAP export"
            )

    sys.modules["pycolmap"] = _MissingPycolmap("pycolmap")


from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.model import DepthAnything3Net  # noqa: E402

__all__ = ["DepthAnything3", "DepthAnything3Net"]
