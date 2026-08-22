"""Канал истины. Часть 6.

Это то, ради чего мир пишется самостоятельно, а не берётся готовым.

ЖЁСТКОЕ ПРАВИЛО: ничто отсюда никогда не попадает в inbox_agent. Нарушение
инвалидирует все эксперименты по связыванию. В коде это защищено тем, что
TruthRecord вообще не умеет превращаться в Event, и тестом
test_truth_never_reaches_inbox.

Поле `retinal_span` — ключ ко всей диагностике восприятия: без него нельзя
ответить, есть ли биекция между объектными слотами агента и предметами мира,
а по поведению этого не узнать — хорошее поведение достижимо при полностью
неверном внутреннем представлении.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ItemTruth:
    item_id: int
    x: float
    y: float
    kind: str
    nutritive: float
    visible: bool
    retinal_span: tuple[int, int] | None
    occluded_by: int


@dataclass(slots=True)
class TruthRecord:
    tick: int
    body: dict[str, Any]
    items: list[ItemTruth]
    regime: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    budget: int = 0
    losses: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "body": self.body,
            "items": [asdict(i) for i in self.items],
            "regime": self.regime,
            "events": self.events,
            "budget": self.budget,
            "losses": self.losses,
        }
