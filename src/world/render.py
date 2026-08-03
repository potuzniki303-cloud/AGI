"""Отрисовка в ASCII. Без зависимостей, работает в терминале и по ssh.

Намеренно не трогает arcade: смотреть на симуляцию нужно на порядок чаще,
чем показывать её кому-то, а окно требует дисплея и тормозит цикл.
"""

from __future__ import annotations

import numpy as np

from .core import World

EMPTY_CH = "·"
FOOD_CH = {1: "*", 2: "%"}
AGENT_CH = "@"
AGENT_ON_FOOD_CH = "&"
CROWD_CH = "#"


def to_ascii(world: World, legend: bool = True) -> str:
    """Кадр мира текстом."""
    size = world.cfg.size
    grid = [[EMPTY_CH] * size for _ in range(size)]

    ys, xs = np.nonzero(world.food)
    for y, x in zip(ys, xs):
        grid[y][x] = FOOD_CH.get(int(world.food[y, x]), "?")

    counts: dict[tuple[int, int], int] = {}
    for b in world.bodies:
        counts[(b.y, b.x)] = counts.get((b.y, b.x), 0) + 1
    for (y, x), n in counts.items():
        if n > 1:
            grid[y][x] = CROWD_CH
        elif world.food[y, x] != 0:
            grid[y][x] = AGENT_ON_FOOD_CH
        else:
            grid[y][x] = AGENT_CH

    body = "\n".join(" ".join(row) for row in grid)
    if not legend:
        return body

    head = (
        f"t={world.tick}  живых={len(world.bodies)}  "
        f"еды={int(np.count_nonzero(world.food))}"
    )
    if world.bodies:
        head += (
            f"  энергия сред={np.mean([b.energy for b in world.bodies]):.1f}"
            f"  поколение макс={max(b.generation for b in world.bodies)}"
        )
    if world.extinct:
        head += "  [ВЫМЕРЛИ]"
    return f"{head}\n{body}"


def print_frame(world: World, clear: bool = True) -> None:
    """Печать кадра с возвратом курсора — годится для живого просмотра."""
    if clear:
        print("\033[H\033[J", end="")
    print(to_ascii(world), flush=True)
