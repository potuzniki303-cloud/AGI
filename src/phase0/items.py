"""Предметы-питатели. Часть 2.3.

Ключевое: `kind` — это ВНЕШНОСТЬ, не питательность. Питательность определяет
режим мира (Части 7-9) и может меняться, пока внешность остаётся прежней.
Именно на этом стоит режим B, поэтому связь «вид -> польза» не должна попасть
сюда ни в каком виде.

РЕШЕНИЕ, КОТОРОГО НЕТ В СПЕЦИФИКАЦИИ (зафиксировано как изменение физики).
Спецификация не говорит, каким становится `kind` после респавна. Первая
реализация разыгрывала его заново, и это оказалось храповиком: на уровне A2+
съедобные съедаются и возвращаются монеткой, несъедобные не съедаются никогда
и копятся. Замер: за 30000 тиков поле выродилось до 0 питательных из 12, и
`greedy_symbolic` умирал от голода при формально верных константах Части 3.

Поэтому здесь:
  * состав поля задаётся раскладкой РОВНО пополам и дальше не меняется;
  * при респавне `kind` СОХРАНЯЕТСЯ, меняется только позиция.

Иначе состав поля дрейфует от поведения агента, а в режиме B это сделало бы
`T_adapt` неизмеримым: после переворота менялась бы не только польза, но и
сама доступность нужного вида.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config, Kind

# Уровни, на которых предметы движутся (2.3, ручка сложности A4).
MOVING_LEVELS = frozenset({"A4", "A5"})


@dataclass(slots=True)
class Item:
    item_id: int
    x: float
    y: float
    kind: Kind
    vx: float = 0.0
    vy: float = 0.0
    alive: bool = True
    respawn_at: int = -1  # тик, на котором вернётся; -1 — уже на поле


class ItemField:
    """Все предметы арены плюс их респавн."""

    def __init__(self, cfg: Config, rng_layout: np.random.Generator,
                 blocked=None) -> None:
        self.cfg = cfg
        # blocked(x, y, clearance) -> занята ли точка препятствием.
        self.blocked = blocked or (lambda x, y, c: False)
        self.moving = cfg.difficulty in MOVING_LEVELS
        self.items: list[Item] = []

        # Состав ровно пополам: при нечётном n_items лишний достаётся виду A.
        n_a = (cfg.n_items + 1) // 2
        kinds = [Kind.A] * n_a + [Kind.B] * (cfg.n_items - n_a)
        rng_layout.shuffle(kinds)

        for i, kind in enumerate(kinds):
            x, y = self._free_position(rng_layout, avoid=None)
            item = Item(item_id=i, x=x, y=y, kind=kind)
            if self.moving:
                self._randomize_velocity(item, rng_layout)
            self.items.append(item)

    # ------------------------------------------------------------- позиция
    def _free_position(
        self, rng: np.random.Generator, avoid: tuple[float, float] | None
    ) -> tuple[float, float]:
        w, h = self.cfg.arena
        r = self.cfg.r_item
        x = y = 0.0
        for _ in range(256):
            x = float(rng.uniform(r, w - r))
            y = float(rng.uniform(r, h - r))
            if self.blocked(x, y, r):
                continue
            if avoid is None:
                return x, y
            if math.hypot(x - avoid[0], y - avoid[1]) >= self.cfg.respawn_min_dist:
                return x, y
        return x, y

    def _randomize_velocity(self, item: Item, rng: np.random.Generator) -> None:
        angle = float(rng.uniform(-math.pi, math.pi))
        speed = float(rng.uniform(0.0, self.cfg.item_speed_max))
        item.vx = speed * math.cos(angle)
        item.vy = speed * math.sin(angle)

    # ---------------------------------------------------------------- такт
    def step(self, tick: int, body_pos: tuple[float, float],
             rng_respawn: np.random.Generator) -> None:
        if self.moving:
            self._move(rng_respawn)

        for item in self.items:
            if item.alive or item.respawn_at < 0 or tick < item.respawn_at:
                continue
            item.x, item.y = self._free_position(rng_respawn, avoid=body_pos)
            item.alive = True
            item.respawn_at = -1
            item.vx = item.vy = 0.0
            if self.moving:
                self._randomize_velocity(item, rng_respawn)
            # item.kind намеренно НЕ трогаем: см. шапку модуля.

    def _move(self, rng: np.random.Generator) -> None:
        """Случайное блуждание с ограничением |v| <= item_speed_max,
        отражение от стен."""
        dt = self.cfg.dt
        w, h = self.cfg.arena
        r = self.cfg.r_item
        vmax = self.cfg.item_speed_max
        for item in self.items:
            if not item.alive:
                continue
            item.vx += float(rng.normal(0.0, vmax)) * dt
            item.vy += float(rng.normal(0.0, vmax)) * dt
            sp = math.hypot(item.vx, item.vy)
            if sp > vmax:
                item.vx *= vmax / sp
                item.vy *= vmax / sp
            item.x += item.vx * dt
            item.y += item.vy * dt
            if item.x < r:
                item.x, item.vx = r, -item.vx
            elif item.x > w - r:
                item.x, item.vx = w - r, -item.vx
            if item.y < r:
                item.y, item.vy = r, -item.vy
            elif item.y > h - r:
                item.y, item.vy = h - r, -item.vy
            # Препятствия непроходимы и для предметов. Без этого движущийся
            # предмет заезжает внутрь и «видно» его оттуда быть не может,
            # то есть окклюзия в канале истины начинает врать.
            if self.blocked(item.x, item.y, r):
                self._push_out(item, r)

    def _push_out(self, item: Item, r: float) -> None:
        """Вытолкнуть предмет из препятствия и отразить скорость.

        Направление ищем пробами по кругу: точную нормаль знает только
        препятствие, а ItemField о его форме не осведомлён намеренно —
        он спрашивает лишь «занято ли здесь».
        """
        for scale in (0.5, 1.0, 2.0, 4.0, 8.0):
            for k in range(8):
                angle = k * math.pi / 4.0
                nx, ny = math.cos(angle), math.sin(angle)
                x = item.x + nx * scale * r
                y = item.y + ny * scale * r
                if not self.blocked(x, y, r):
                    item.x, item.y = x, y
                    normal = item.vx * nx + item.vy * ny
                    if normal < 0.0:
                        item.vx -= 2.0 * normal * nx
                        item.vy -= 2.0 * normal * ny
                    return

    def consume(self, item: Item, tick: int) -> None:
        item.alive = False
        item.respawn_at = tick + self.cfg.t_respawn
        item.vx = item.vy = 0.0

    def touching(self, x: float, y: float) -> list[Item]:
        """Предметы в контакте с телом. Поедание мгновенно, без действия
        «съесть»: отдельное действие добавило бы вторую задачу назначения
        заслуг до того, как проверен базовый механизм."""
        reach = self.cfg.r_body + self.cfg.r_item
        return [i for i in self.items
                if i.alive and math.hypot(i.x - x, i.y - y) < reach]

    @property
    def visible_items(self) -> list[Item]:
        return [i for i in self.items if i.alive]

    def composition(self) -> dict[str, int]:
        """Сколько каких видов сейчас на поле. Диагностика: это число обязано
        быть стационарным, иначе физика зависит от поведения агента."""
        out = {"A": 0, "B": 0}
        for i in self.items:
            if i.alive:
                out[i.kind.name] += 1
        return out

    # ------------------------------------------------------------- снапшот
    def state(self) -> dict:
        return {
            "items": [
                (i.item_id, i.x, i.y, int(i.kind), i.vx, i.vy, i.alive, i.respawn_at)
                for i in self.items
            ]
        }

    def restore(self, state: dict) -> None:
        self.items = [
            Item(item_id=r[0], x=r[1], y=r[2], kind=Kind(r[3]), vx=r[4], vy=r[5],
                 alive=r[6], respawn_at=r[7])
            for r in state["items"]
        ]
