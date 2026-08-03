"""Модульные миры для экспериментов с обучением без backprop.

Быстрый старт:

    from world import make, simulate
    from world.stubs import RandomMind

    w = make("forage", size=24, seed=1)
    print(w.describe())                       # размер входа и число действий

    h = simulate(w, lambda rng: RandomMind(w.action_size, rng), steps=2000)
    print(h.summary())

Подключение своего агента — это ровно два метода, см. protocols.Mind:

    class MyAgent:
        def act(self, observation) -> int: ...
        def spawn(self, rng) -> "MyAgent": ...

Мир не знает и не хочет знать, что внутри. Он не трогает веса, не считает
награду и никого не ранжирует. Его работа — быть, кормить и убивать.

Смена мира — одна строка: make("seasons"), make("shift"), make("two_foods").
Смена формы наблюдения — переопредели _build_sensor() или собери сенсор
из sensors.py и присвой world.sensor до reset().
"""

from __future__ import annotations

from .core import Body, StepReport, World, WorldConfig
from .protocols import Mind
from .render import print_frame, to_ascii
from .runner import History, simulate
from .sensors import (
    AntennaSensor,
    ConcatSensor,
    Interoception,
    PatchSensor,
    Sensor,
)
from .worlds import (
    ForageWorld,
    PatchesConfig,
    PatchesWorld,
    SeasonsConfig,
    SeasonsWorld,
    ShiftConfig,
    ShiftWorld,
    TwoFoodsConfig,
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
    """Добавить свой мир в реестр."""
    if not issubclass(cls, World):
        raise TypeError(f"{cls} must subclass World")
    _REGISTRY[name] = cls


def list_worlds() -> list[str]:
    return sorted(_REGISTRY)


def make(name: str, **overrides) -> World:
    """Собрать мир по имени. Именованные аргументы идут в его Config.

    Неизвестное имя параметра — это ошибка, а не молчаливое игнорирование:
    опечатка в физике должна падать, а не тихо менять эксперимент.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown world: {name!r}. available: {list_worlds()}")
    cls = _REGISTRY[name]
    return cls(cls.Config(**overrides))


__all__ = [
    "make",
    "register",
    "list_worlds",
    "simulate",
    "History",
    "World",
    "WorldConfig",
    "Body",
    "StepReport",
    "Mind",
    "Sensor",
    "AntennaSensor",
    "PatchSensor",
    "Interoception",
    "ConcatSensor",
    "ForageWorld",
    "SeasonsWorld",
    "SeasonsConfig",
    "PatchesWorld",
    "PatchesConfig",
    "ShiftWorld",
    "ShiftConfig",
    "TwoFoodsWorld",
    "TwoFoodsConfig",
    "to_ascii",
    "print_frame",
]
