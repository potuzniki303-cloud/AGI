"""Сборка бейзлайна и прогон с метриками.

Существует отдельно от driver.py по одной причине: символические бейзлайны
получают привилегированный вход, и место, где это происходит, должно быть
одно и хорошо видное. Мир при этом по-прежнему ничего не знает про агента.
"""

from __future__ import annotations

from typing import Any

from .baselines import BASELINES
from .budget import BudgetProfile
from .config import Config
from .events import Channels
from .logs import RunLogger
from .metrics import RunMetrics
from .world import World


def build(name: str, cfg: Config, channels: Channels, seed: int = 0) -> Any:
    """Создать бейзлайн по имени."""
    if name not in BASELINES:
        raise KeyError(f"неизвестный бейзлайн: {name}. есть: {sorted(BASELINES)}")
    cls = BASELINES[name]
    needs_channels = name in ("greedy_sustained", "greedy_transient",
                          "linear_pixel", "small_rnn_bptt")
    return cls(cfg, channels, seed=seed) if needs_channels else cls(cfg, seed=seed)


def run_baseline(name: str, cfg: Config, ticks: int = 30_000, seed: int = 0,
                 logger: RunLogger | None = None,
                 budget: BudgetProfile | None = None) -> tuple[World, RunMetrics]:
    """Прогнать бейзлайн и собрать метрики."""
    world = World(cfg)
    agent = build(name, cfg, world.channels, seed=seed)
    metrics = RunMetrics()
    profile = budget or BudgetProfile.constant(1000)

    wants_truth = getattr(agent, "symbolic", False)
    wants_energy = hasattr(agent, "note_energy")

    for _ in range(ticks):
        world.set_budget(profile.at(world.tick))

        # Привилегированный вход символических бейзлайнов — ДО тика,
        # отдельным явным вызовом. В inbox он не попадает.
        if wants_truth:
            agent.observe_symbolic(world)

        record = world.step()
        sensory = world.inbox.drain()
        metrics.observe(world, record)

        if logger is not None:
            logger.truth(record)
            logger.sensor(sensory)
            logger.budget(record.tick, profile.at(record.tick))

        if wants_energy:
            agent.note_energy(world.body.energy)

        before = len(world.outbox)
        agent.step(sensory, world.outbox, profile.at(world.tick))
        if logger is not None:
            logger.motor(list(world.outbox.peek()[before:]))

    return world, metrics
