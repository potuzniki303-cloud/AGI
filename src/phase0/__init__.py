"""Мир Фазы 0.

Реализация спецификации `world_spec_phase01.md`. Зависимость одна — numpy.
arcade подтягивается лениво и только для графической игры руками.

Что здесь есть и чего здесь нет — см. Часть 0 спецификации. Коротко: физика,
тело, гомеостат, сенсорный и моторный тракт, событийный контракт, приборная
база и бейзлайны. Ничего из архитектуры агента.

Главный инвариант, который нельзя сломать: в пакете нет функции вида
`get_action(agent)`. Мир не знает про агента и не вызывает его код.
"""

from .budget import BudgetProfile
from .config import Config, Kind, Level, Mode
from .driver import replay, run
from .events import Channels, Event, Motor, Queue
from .harness import build, run_baseline
from .logs import RunLogger
from .metrics import (RunMetrics, adaptation_curve, error_cost_curve,
                      error_cost_summary, noise_floor, savings)
from .snapshot import clone, fork, load, save
from .world import World

__all__ = [
    "BudgetProfile", "Channels", "Config", "Event", "Kind", "Level", "Mode",
    "Motor", "Queue", "RunLogger", "RunMetrics", "World", "adaptation_curve", "error_cost_curve", "error_cost_summary", "noise_floor",
    "build", "clone", "fork", "load", "replay", "run", "run_baseline",
    "save", "savings",
]
