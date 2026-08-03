"""Цикл симуляции и наблюдательская статистика.

Всё, что здесь считается, считается ДЛЯ ЧЕЛОВЕКА. Ни одна из этих величин
не попадает обратно в мир и не влияет ни на одно решение. Это важно держать
в голове при доработке: как только средняя продолжительность жизни начнёт
на что-то влиять, она станет функцией приспособленности, и вся конструкция
схлопнется в обычный генетический алгоритм.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .core import StepReport, World
from .protocols import Mind


@dataclass
class History:
    """Времянные ряды по такту. Только для графиков и глазами посмотреть."""

    population: list[int] = field(default_factory=list)
    food: list[int] = field(default_factory=list)
    births: list[int] = field(default_factory=list)
    deaths: list[int] = field(default_factory=list)
    eaten: list[int] = field(default_factory=list)
    mean_energy: list[float] = field(default_factory=list)
    mean_age: list[float] = field(default_factory=list)
    lifespans: list[int] = field(default_factory=list)
    max_generation: int = 0
    ticks: int = 0
    extinct_at: int | None = None

    def record(self, report: StepReport, world: World) -> None:
        self.population.append(report.population)
        self.food.append(report.food_on_grid)
        self.births.append(report.births)
        self.deaths.append(report.deaths)
        self.eaten.append(report.eaten)
        self.mean_energy.append(report.mean_energy)
        self.mean_age.append(report.mean_age)
        self.ticks = report.tick
        if world.bodies:
            self.max_generation = max(
                self.max_generation, max(b.generation for b in world.bodies)
            )
        if report.extinct and self.extinct_at is None:
            self.extinct_at = report.tick

    def summary(self) -> str:
        if not self.population:
            return "нет данных"
        alive = self.population[-1]
        spans = self.lifespans
        med = float(np.median(spans)) if spans else 0.0
        p90 = float(np.percentile(spans, 90)) if spans else 0.0
        tail = self.population[-min(len(self.population), 200):]
        return (
            f"тактов: {self.ticks}  живых: {alive}  "
            f"популяция(хвост) сред/макс: {np.mean(tail):.1f}/{max(tail)}  "
            f"поколений: {self.max_generation}  "
            f"рождений: {sum(self.births)}  смертей: {sum(self.deaths)}\n"
            f"жизнь медиана/p90/макс: {med:.0f}/{p90:.0f}/"
            f"{max(spans) if spans else 0}  "
            f"еда на поле (сред): {np.mean(self.food):.1f}  "
            + (f"ВЫМЕРЛИ на такте {self.extinct_at}" if self.extinct_at else "")
        )


def simulate(
    world: World,
    mind_factory: Callable[[np.random.Generator], Mind],
    steps: int,
    on_step: Callable[[World, StepReport], None] | None = None,
    stop_on_extinction: bool = True,
) -> History:
    """Заселить мир и крутить его steps тактов.

    mind_factory вызывается для КАЖДОГО начального тела отдельно и должен
    возвращать независимый случайный геном. Дальше мир никого не создаёт —
    все последующие тела появляются через mind.spawn() их родителей.
    """
    world.reset(mind_factory)
    history = History()

    for _ in range(steps):
        report = world.step()
        history.record(report, world)
        if on_step is not None:
            on_step(world, report)
        if world.extinct and stop_on_extinction:
            break

    history.lifespans = list(world._lifespans)
    # Живые на момент остановки — незавершённые жизни; их возраст это нижняя
    # граница, а не длительность, поэтому в lifespans они не попадают.
    return history
