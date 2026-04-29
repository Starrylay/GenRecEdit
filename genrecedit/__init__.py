def __getattr__(name):
    if name == "GenRecEditHyperParams":
        from .hparams import GenRecEditHyperParams

        return GenRecEditHyperParams
    if name == "GenRecEditCovarianceCache":
        from .cov_cache import GenRecEditCovarianceCache

        return GenRecEditCovarianceCache
    if name == "GenRecEditModelBundle":
        from .model_bundle import GenRecEditModelBundle

        return GenRecEditModelBundle
    if name == "GenRecEdit":
        from .editor import GenRecEdit

        return GenRecEdit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GenRecEdit",
    "GenRecEditCovarianceCache",
    "GenRecEditHyperParams",
    "GenRecEditModelBundle",
]
