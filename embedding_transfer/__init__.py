"""Training and inference package for Vietnamese embedding-space transfer."""

__all__ = ["EmbeddingTransferAdapter", "MOCHEGPredictor", "ViVQAPredictor"]


def __getattr__(name):
    # Data preparation and scoring must work without loading model dependencies.
    from importlib import import_module

    modules = {
        "EmbeddingTransferAdapter": "adapter",
        "MOCHEGPredictor": "mocheg",
        "ViVQAPredictor": "vivqa",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f".{modules[name]}", __name__), name)
