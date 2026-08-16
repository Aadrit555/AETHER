from aether.core.registry import registry


class Test:
    pass


registry.register(
    "models",
    "linear",
    Test
)

registry.show()