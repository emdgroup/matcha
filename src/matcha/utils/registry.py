"""
Taken from: https://github.com/chemprop/chemprop/blob/main/chemprop/utils/registry.py
"""

from typing import Any, Iterable, Type, TypeVar

T = TypeVar("T")


class ClassRegistry(dict[str, Type[T]]):
    """A dictionary-based registry that maps string aliases to classes.

    Provides a decorator-based interface for registering classes under one or
    more aliases. Lookups are case-insensitive.
    """

    def register(self, alias: Any | Iterable[Any] | None = None):
        """Register a class in the registry under one or more aliases.

        Can be used as a decorator. If no alias is provided, the lowercase
        class name is used as the default key.

        :param alias: One or more string aliases to register the class under.
            If ``None``, defaults to the lowercase class name.
        :type alias: Any | Iterable[Any] | None
        :returns: A decorator that registers the class and returns it unchanged.
        :rtype: Callable
        """

        def decorator(cls):
            if alias is None:
                keys = [cls.__name__.lower()]
            elif isinstance(alias, str):
                keys = [alias]
            else:
                keys = alias

            cls.alias = keys[0]
            for k in keys:
                self[k] = cls

            return cls

        return decorator

    __call__ = register
    """Alias for :meth:`register`, allowing the registry instance to be used
    directly as a decorator."""

    def __repr__(self) -> str:  # pragma: no cover
        """Return a developer-friendly string representation of the registry.

        :returns: String showing the class name and contents.
        :rtype: str
        """
        return f"{self.__class__.__name__}: {super().__repr__()}"

    def __str__(self) -> str:  # pragma: no cover
        """Return a human-readable, indented string representation of the registry.

        :returns: Multi-line formatted string of registered classes.
        :rtype: str
        """
        INDENT = 4
        items = [f"{' ' * INDENT}{repr(k)}: {repr(v)}" for k, v in self.items()]

        return "\n".join([f"{self.__class__.__name__} {'{'}", ",\n".join(items), "}"])

    def __getitem__(self, key):
        """Retrieve a registered class by its alias (case-insensitive).

        :param str key: The alias to look up.
        :returns: The class registered under the given alias.
        :rtype: Type[T]
        :raises ValueError: If the key is not found in the registry.
        """
        if key.lower() in self:
            return super().__getitem__(key.lower())
        else:
            valid_keys = ", ".join([repr(k) for k in self.keys()])
            raise ValueError(
                f"'{key}' is not valid, here is a list of valid options: {valid_keys}"
            )
