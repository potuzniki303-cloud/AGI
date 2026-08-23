"""Непроходимые препятствия. Уровень сложности A5.

Зачем именно они. Взаимное перекрытие предметов даёт окклюзию мимолётную:
предметы движутся и респавнятся, и предмет редко скрыт дольше секунды. Пятая
метрика связывания — восстановление идентичности ПОСЛЕ окклюзии — на таких
данных меряет почти шум. Препятствие стоит на месте, поэтому предмет за ним
скрыт столько, сколько агент туда не заходит, и вопрос «тот же object_id или
новый» становится содержательным.

Препятствия ВИДНЫ, в отличие от стен. Это не непоследовательность: невидимое
препятствие, которое загораживает обзор, означало бы, что предметы пропадают
за пустотой, и связывание проверялось бы на артефакте. Ахроматические (C=0),
то есть цветовая ось по-прежнему разделяет ровно kind A и kind B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config

# Идентификаторы препятствий в проекции сетчатки идут отдельным диапазоном,
# чтобы не путаться с item_id в канале истины.
OBSTACLE_ID_BASE = 100_000


@dataclass(slots=True)
class Obstacle:
    obstacle_id: int
    x: float
    y: float
    radius: float


class ObstacleField:
    """Статичные круглые препятствия."""

    def __init__(self, cfg: Config, rng_layout: np.random.Generator) -> None:
        self.cfg = cfg
        self.obstacles: list[Obstacle] = []
        if cfg.difficulty != "A5" or cfg.n_obstacles <= 0:
            return

        w, h = cfg.arena
        margin = cfg.r_obstacle + cfg.r_body * 2.0
        for i in range(cfg.n_obstacles):
            for _ in range(256):
                x = float(rng_layout.uniform(margin, w - margin))
                y = float(rng_layout.uniform(margin, h - margin))
                if all(math.hypot(x - o.x, y - o.y) > 2 * cfg.r_obstacle + 4.0
                       for o in self.obstacles):
                    break
            self.obstacles.append(
                Obstacle(OBSTACLE_ID_BASE + i, x, y, cfg.r_obstacle))

    def __bool__(self) -> bool:
        return bool(self.obstacles)

    # ------------------------------------------------------------ геометрия
    def blocks(self, x: float, y: float, clearance: float) -> bool:
        """Занята ли точка (для спавна предметов и респавна тела)."""
        return any(math.hypot(x - o.x, y - o.y) < o.radius + clearance
                   for o in self.obstacles)

    def collide(self, body, cfg: Config) -> int:
        """Вытолкнуть тело наружу. Нормальная компонента скорости гасится,
        тангенциальная сохраняется — та же семантика, что у стен."""
        hit = 0
        for o in self.obstacles:
            dx, dy = body.x - o.x, body.y - o.y
            dist = math.hypot(dx, dy)
            reach = o.radius + cfg.r_body
            if dist >= reach:
                continue
            if dist < 1e-9:
                nx, ny = 1.0, 0.0
                dist = 1e-9
            else:
                nx, ny = dx / dist, dy / dist
            body.x = o.x + nx * reach
            body.y = o.y + ny * reach
            normal = body.vx * nx + body.vy * ny
            if normal < 0.0:
                body.vx -= normal * nx
                body.vy -= normal * ny
                hit = 1
        return hit

    # ------------------------------------------------------------- снапшот
    def state(self) -> list[tuple[int, float, float, float]]:
        return [(o.obstacle_id, o.x, o.y, o.radius) for o in self.obstacles]

    def restore(self, rows) -> None:
        self.obstacles = [Obstacle(*row) for row in rows]
