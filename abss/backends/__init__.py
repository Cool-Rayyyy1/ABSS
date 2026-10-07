def load_backend(config):
    if config.model == "flux":
        from .flux import FluxBackend

        return FluxBackend(config)
    from .hunyuan import HunyuanBackend

    return HunyuanBackend(config)
