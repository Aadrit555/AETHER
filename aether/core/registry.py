"""
Aether Registry
===============

The registry is responsible for managing all framework components.

Nothing in Aether should instantiate a component directly.
Everything should come through the registry.
"""

from typing import Dict, Type


class Registry:

    def __init__(self):
        self._components: Dict[str, Dict[str, Type]] = {
            "models": {},
            "datasets": {},
            "trainers": {}
        }

    def register(self, category: str, name: str, component):

        category = category.lower()
        name = name.lower()

        if category not in self._components:
            raise ValueError(f"Unknown category '{category}'")

        self._components[category][name] = component

    def create(self, category: str, name: str, **kwargs):

        category = category.lower()
        name = name.lower()

        if category not in self._components:
            raise ValueError(f"Unknown category '{category}'")

        if name not in self._components[category]:
            raise ValueError(
                f"'{name}' not registered under '{category}'"
            )

        cls = self._components[category][name]

        return cls(**kwargs)

    def show(self):

        print()

        print("=" * 50)
        print("Registered Components")
        print("=" * 50)

        for category, items in self._components.items():

            print()

            print(category.upper())

            if not items:
                print("  (none)")
                continue

            for item in items:
                print(f"  ✓ {item}")

        print()


registry = Registry()