try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = '0.0.0'
    version_tuple = (0, 0, 0)


from .frontends.onnx import design_directives, from_onnx
from .model import AIEModel, from_hls4ml

__all__ = ['__version__', 'version_tuple', 'AIEModel', 'design_directives', 'from_hls4ml', 'from_onnx']
