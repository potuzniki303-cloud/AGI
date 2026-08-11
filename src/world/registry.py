"""Реестр миров — одна точка правды.

Из него берут имена обе двери: make() для своего цикла и register_all()
для gym. Добавил мир сюда — он сразу доступен обоими способами, дописывать
gym-идентификатор руками не надо.

Живёт отдельным модулем, а не в __init__.py, чтобы gym_env мог импортировать
реестр без кольца импортов.
"""

from __future__ import annotations

from .core import World
from .worlds import (
    ForageWorld,
    PatchesWorld,
    SeasonsWorld,
    ShiftWorld,
    TwoFoodsWorld,
)

_REGISTRY: dict[str, type[World]] = {
    "forage": ForageWorld,
    "seasons": SeasonsWorld,
    "patches": PatchesWorld,
    "shift": ShiftWorld,
    "two_foods": TwoFoodsWorld,
}


def register(name: str, cls: type[World]) -> None:
    """Добавить свой мир. После этого он виден и make(), и gym."""
    if not issubclass(cls, World):
        raise TypeError(f"{cls} must subclass World")
    _REGISTRY[name] = cls


def list_worlds() -> list[str]:
    return sorted(_REGISTRY)


def world_class(name: str) -> type[World]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown world: {name!r}. available: {list_worlds()}")
    return _REGISTRY[name]


def make(name: str, **overrides) -> World:
    """Собрать мир по имени. Именованные аргументы идут в его Config.

    Неизвестное имя параметра — ошибка, а не молчаливое игнорирование:
    опечатка в физике должна падать, а не тихо менять эксперимент.
    """
    cls = world_class(name)
    return cls(cls.Config(**overrides))
