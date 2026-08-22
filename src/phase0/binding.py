"""Пять метрик связывания. Часть 6 спецификации.

Это то, ради чего мир пишется самостоятельно. Ответ на вопрос «как формируется
объектный слот с устойчивым object_id» НЕВОЗМОЖНО получить, глядя на поведение:
хорошее поведение достижимо при полностью неверном внутреннем представлении.
Нужно прямое сопоставление слотов агента с `item_id` из канала истины.

Контракт замера. Агент отдаёт свои слоты как отображение
`object_id -> множество индексов рецепторов`, которые слот считает своими.
Больше ничего от него не требуется: ни того, что слот «знает» про предмет,
ни какой-либо семантики. Сопоставление делает прибор, по массиву
`retina_owner` из канала истины.

    tracker = BindingTracker()
    ...
    record = world.step()
    tracker.observe(record, my_agent.slots())      # slots(): {oid: {рецепторы}}
    ...
    print(tracker.report())

Правило сопоставления: слот приписывается тому предмету, которому принадлежит
больше всего его рецепторов. Предмет считается покрытым слотом, если слот
содержит хотя бы один его рецептор.

ВАЖНО про интерпретацию. Метрики считаются только по видимым предметам —
тем, у кого `visible_receptors > 0`. Полностью закрытый предмет не может быть
связан ни с чем, и требовать этого от агента нельзя; зато именно на переходе
«скрылся -> появился» меряется пятая метрика.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .truth import TruthRecord


@dataclass
class BindingReport:
    ticks: int = 0
    slots_seen: int = 0

    # 1. соответствие
    bijection_ticks: int = 0
    slot_precision_num: int = 0     # слотов, попавших ровно в один предмет
    item_recall_num: int = 0        # видимых предметов, покрытых ровно одним слотом
    visible_items_total: int = 0

    # 2. устойчивость идентичности
    identity_episodes: int = 0
    identity_switches: int = 0

    # 3-4. склеивание и дробление
    merge_events: int = 0
    split_events: int = 0

    # 5. восстановление после окклюзии
    occlusion_returns: int = 0
    occlusion_same_id: int = 0

    def as_dict(self) -> dict[str, float | int | str]:
        if self.ticks == 0:
            return {"статус": "нет данных: агент не отдавал слотов"}
        vis = max(1, self.visible_items_total)
        out: dict[str, float | int | str] = {
            "тиков": self.ticks,
            "1. соответствие (доля тиков с биекцией)":
                round(self.bijection_ticks / self.ticks, 3),
            "   точность слотов": round(self.slot_precision_num / max(1, self.slots_seen), 3),
            "   полнота по предметам": round(self.item_recall_num / vis, 3),
            "3. склеиваний на 1000 тиков":
                round(self.merge_events / self.ticks * 1000, 2),
            "4. дроблений на 1000 тиков":
                round(self.split_events / self.ticks * 1000, 2),
        }
        if self.identity_episodes:
            out["2. смен object_id на эпизод видимости"] = round(
                self.identity_switches / self.identity_episodes, 3)
        else:
            out["2. смен object_id на эпизод видимости"] = "нет эпизодов"
        if self.occlusion_returns:
            out["5. восстановление после окклюзии"] = round(
                self.occlusion_same_id / self.occlusion_returns, 3)
            out["   возвращений измерено"] = self.occlusion_returns
        else:
            out["5. восстановление после окклюзии"] = "окклюзий с возвратом не было"
        return out


class BindingTracker:
    """Накопитель пяти метрик. Ничего не возвращает в мир."""

    def __init__(self) -> None:
        self.r = BindingReport()
        self._bound: dict[int, int] = {}          # item_id -> object_id сейчас
        self._last_seen: dict[int, int] = {}      # item_id -> object_id до пропажи
        self._was_visible: set[int] = set()
        self._episode_open: set[int] = set()

    # ------------------------------------------------------------------
    def observe(self, record: TruthRecord,
                slots: Mapping[int, Iterable[int]] | None) -> None:
        if slots is None:
            return
        owner = record.retina_owner
        if not owner:
            return

        visible = {it.item_id for it in record.items if it.visible}
        # Полная окклюзия — отдельный случай: предмет в поле зрения, но невидим.
        occluded_now = {it.item_id for it in record.items
                        if it.in_fov and it.occlusion == "full"}

        slot_items: dict[int, Counter] = {}
        for oid, receptors in slots.items():
            counts: Counter = Counter()
            for i in receptors:
                if 0 <= i < len(owner) and owner[i] >= 0:
                    counts[owner[i]] += 1
            slot_items[oid] = counts

        self.r.ticks += 1
        self.r.slots_seen += len(slot_items)
        self.r.visible_items_total += len(visible)

        # --- 3. склеивание: один слот накрыл несколько предметов ---
        for counts in slot_items.values():
            if len(counts) > 1:
                self.r.merge_events += 1
            elif len(counts) == 1:
                self.r.slot_precision_num += 1

        # --- 4. дробление: один предмет накрыт несколькими слотами ---
        covering: dict[int, list[int]] = defaultdict(list)
        for oid, counts in slot_items.items():
            for item_id in counts:
                covering[item_id].append(oid)
        for item_id in visible:
            n = len(covering.get(item_id, ()))
            if n > 1:
                self.r.split_events += 1
            elif n == 1:
                self.r.item_recall_num += 1

        # --- 1. соответствие: биекция слотов и видимых предметов ---
        assigned = {oid: counts.most_common(1)[0][0]
                    for oid, counts in slot_items.items() if counts}
        targets = list(assigned.values())
        if (visible and len(assigned) == len(visible)
                and len(set(targets)) == len(targets) and set(targets) == visible):
            self.r.bijection_ticks += 1

        # --- 2 и 5: идентичность во времени ---
        current: dict[int, int] = {}
        for oid, item_id in assigned.items():
            current.setdefault(item_id, oid)

        for item_id in visible:
            oid = current.get(item_id)
            if oid is None:
                continue
            if item_id in self._episode_open:
                previous = self._bound.get(item_id)
                if previous is not None and previous != oid:
                    self.r.identity_switches += 1
            else:
                # Эпизод видимости открылся. Если предмет до этого пропадал —
                # это и есть проверка восстановления после окклюзии.
                self.r.identity_episodes += 1
                self._episode_open.add(item_id)
                if item_id in self._last_seen:
                    self.r.occlusion_returns += 1
                    if self._last_seen[item_id] == oid:
                        self.r.occlusion_same_id += 1
            self._bound[item_id] = oid

        # Предметы, которые перестали быть видимыми: закрываем эпизод и
        # запоминаем последний object_id — на случай возвращения.
        for item_id in list(self._episode_open):
            if item_id not in visible:
                self._episode_open.discard(item_id)
                if item_id in self._bound:
                    self._last_seen[item_id] = self._bound.pop(item_id)

        self._was_visible = visible | occluded_now

    def report(self) -> dict[str, float | int | str]:
        return self.r.as_dict()


# ----------------------------------------------------------------------
class ConnectedComponentSlots:
    """Опорный сегментатор: непрерывные куски непустой сетчатки — это слоты.

    Это ПРИБОР, а не кандидат в агенты и не то, что агент должен выучить.
    Нужен ровно для двух вещей: проверить, что метрики вообще считаются, и
    дать нижнюю отметку — сколько выбивает тривиальная сегментация без
    всякой памяти. Идентичность он держит по перекрытию с прошлым тиком,
    то есть самым дешёвым способом из возможных.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self._next_id = 0
        self._prev: dict[int, set[int]] = {}

    def slots(self, luminance) -> dict[int, set[int]]:
        runs: list[set[int]] = []
        current: set[int] = set()
        for i, value in enumerate(luminance):
            if value > self.threshold:
                current.add(i)
            elif current:
                runs.append(current)
                current = set()
        if current:
            runs.append(current)

        out: dict[int, set[int]] = {}
        used: set[int] = set()
        for run in runs:
            best_id, best_overlap = None, 0
            for oid, prev in self._prev.items():
                if oid in used:
                    continue
                overlap = len(run & prev)
                if overlap > best_overlap:
                    best_id, best_overlap = oid, overlap
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            used.add(best_id)
            out[best_id] = run
        self._prev = out
        return out
