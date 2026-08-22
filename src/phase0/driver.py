"""Драйвер: качает мир и агента, не давая миру знать про агента. Часть 1.1.

Разделение ответственности здесь принципиальное. Мир тикает сам по себе.
Агент — источник событий. Драйвер лишь даёт обоим процессорное время и
переносит бюджет. Он НЕ спрашивает у агента действие и не передаёт агента
в мир.

Протокол агента (агент реализует его сам, мир о нём не знает):

    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None

`inbox` — то, что мир насыпал с прошлого раза. `outbox` — куда писать, когда
и если созрел. Ноль событий — легальный ответ, и никакой пометки «пропустил
ход» не существует.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from .budget import BudgetProfile
from .events import Event, Queue
from .logs import RunLogger
from .truth import TruthRecord
from .world import World


class Agent(Protocol):
    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None: ...


def run(
    world: World,
    agent: Agent | None = None,
    ticks: int = 1000,
    budget: BudgetProfile | None = None,
    logger: RunLogger | None = None,
    on_tick: Callable[[World, TruthRecord], None] | None = None,
) -> World:
    """Прокрутить мир `ticks` тиков.

    `agent=None` — легальный и важный случай: мир обязан работать при полном
    молчании агента. Это условие завершения шага 0.1.
    """
    profile = budget or BudgetProfile.constant(1000)
    for _ in range(ticks):
        quota = profile.at(world.tick)
        world.set_budget(quota)

        record = world.step()

        sensory = world.inbox.drain()
        if logger is not None:
            logger.truth(record)
            logger.sensor(sensory)
            logger.budget(record.tick, quota)

        if agent is not None:
            before = len(world.outbox)
            agent.step(sensory, world.outbox, quota)
            if logger is not None:
                emitted = world.outbox.peek()[before:]
                logger.motor(list(emitted))

        if on_tick is not None:
            on_tick(world, record)
    return world


def replay(cfg: Any, motor_events: list[Event], ticks: int) -> World:
    """Воспроизвести прогон из конфига и записанного лога действий.

    Это и есть проверка Части 10.1: при фиксированном seed и записанном логе
    действий мир обязан воспроизвестись побитово.
    """
    world = World(cfg)
    by_tick: dict[int, list[Event]] = {}
    for e in motor_events:
        by_tick.setdefault(e.t, []).append(e)

    for _ in range(ticks):
        # Действия кладём ДО тика: в живом прогоне агент писал их по итогам
        # предыдущего тика, и мир видел их на следующем осушении.
        for e in by_tick.get(world.tick, ()):
            world.outbox.put(e)
        world.step()
        world.inbox.drain()
    return world
