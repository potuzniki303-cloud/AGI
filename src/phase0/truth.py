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
    """Правда об одном предмете на этом тике.

    Про `visible` и `visible_receptors`. `retinal_span` — куда предмет попал
    БЫ (его угловой размер, обрезанный полем зрения), `visible_receptors` —
    сколько рецепторов он реально занял. Раньше эти две вещи были слиты, и
    `visible` означал «в поле зрения»: все 35 447 записей с occluded_by >= 0
    были помечены видимыми. На таких данных склеивание, дробление и
    восстановление после окклюзии посчитать нельзя.

    `occlusion`: "none" | "partial" | "full".
    """

    item_id: int
    x: float
    y: float
    kind: str
    nutritive: float
    visible: bool               # видно хотя бы одним рецептором
    in_fov: bool                # попадает в поле зрения (может быть закрыт)
    retinal_span: tuple[int, int] | None
    visible_receptors: int
    occlusion: str
    occluded_by: int


@dataclass(slots=True)
class TruthRecord:
    tick: int
    body: dict[str, Any]
    items: list[ItemTruth]
    regime: dict[str, Any]
    # Кому принадлежит каждый рецептор (item_id или -1). Это и есть ключ к
    # метрикам связывания: имея слоты агента как множества рецепторов, по
    # этому массиву считаются все пять метрик Части 6.
    retina_owner: list[int] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    budget: int = 0
    losses: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "body": self.body,
            "items": [asdict(i) for i in self.items],
            "regime": self.regime,
            "retina_owner": self.retina_owner,
            "events": self.events,
            "budget": self.budget,
            "losses": self.losses,
        }
