"""Снапшоты и форк. Часть 10.3.

Снапшот = состояние мира + состояние RNG (+ состояние агента, если он умеет
его отдавать; это его дело, мир в него не лезет).

Форк — единственный корректный способ измерить дисперсию поведения при
ФИКСИРОВАННОЙ истории: от одного снапшота запускаются N продолжений с разными
seed агента. Без него неизвестно, что вы наблюдаете — эффект или шум.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

from .world import World


def save(world: World, path: str | Path, agent_state: Any = None) -> None:
    payload = {
        "world": world.state(),
        "agent": agent_state,
        "config": world.cfg,
    }
    Path(path).write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def load(path: str | Path) -> tuple[World, Any]:
    payload = pickle.loads(Path(path).read_bytes())
    world = World(payload["config"])
    world.restore(payload["world"])
    return world, payload["agent"]


def clone(world: World) -> World:
    """Копия мира, продолжающая ту же историю побитово."""
    twin = World(world.cfg)
    twin.restore(pickle.loads(pickle.dumps(world.state())))
    return twin


def fork(world: World, n: int, agent_factory: Callable[[int], Any],
         ticks: int, runner: Callable[[World, Any, int], Any]) -> list[Any]:
    """N продолжений одной истории с разными seed агента.

    `agent_factory(seed)` создаёт агента, `runner(world, agent, ticks)` крутит
    продолжение и возвращает что угодно, что вы хотите сравнить.
    """
    return [runner(clone(world), agent_factory(seed), ticks) for seed in range(n)]
