"""Визуализация. Часть 10.4 — не опция, а прибор.

Три обязательных вида:
  1. Мир сверху — тело, предметы, поле зрения, траектория.
  2. Сетчатка — что видит агент, оба канала, с наложенной разметкой
     retinal_span из канала истины. Позволяет ГЛАЗАМИ увидеть расхождение
     восприятия и реальности.
  3. Граф — узлы, типы, активность (данные даёт агент, мир их не знает).

Здесь ASCII-версия: работает без дисплея, попадает в тесты и в логи. Живой
графический вид и игра человеком — в play.py.

Вид «сетчатка» — главный. Если человек не может играть по этой сетчатке,
агент тем более не сможет, и это не его вина.
"""

from __future__ import annotations

import math

import numpy as np

from .truth import TruthRecord
from .world import World

SHADES = " .:-=+*#%@"


def world_view(world: World, width: int = 64, height: int = 32,
               trail: list[tuple[float, float]] | None = None) -> str:
    """Мир сверху. A — заглавная, B — строчная, @ — тело, стрелка — курс."""
    cfg = world.cfg
    grid = [[" "] * width for _ in range(height)]

    def cell(x: float, y: float) -> tuple[int, int]:
        cx = min(width - 1, max(0, int(x / cfg.arena[0] * width)))
        cy = min(height - 1, max(0, int(y / cfg.arena[1] * height)))
        return cx, cy

    if trail:
        for tx, ty in trail[-400:]:
            cx, cy = cell(tx, ty)
            if grid[cy][cx] == " ":
                grid[cy][cx] = "·"

    # Поле зрения: два луча по краям FOV.
    half = math.radians(cfg.fov_deg) / 2.0
    for side in (-half, half):
        ang = world.body.theta + side
        for step in range(1, 70):
            rx = world.body.x + math.cos(ang) * step
            ry = world.body.y + math.sin(ang) * step
            if not (0 <= rx < cfg.arena[0] and 0 <= ry < cfg.arena[1]):
                break
            cx, cy = cell(rx, ry)
            if grid[cy][cx] == " ":
                grid[cy][cx] = ","

    for item in world.items.visible_items:
        cx, cy = cell(item.x, item.y)
        grid[cy][cx] = "A" if item.kind.name == "A" else "b"

    bx, by = cell(world.body.x, world.body.y)
    grid[by][bx] = "@"

    border = "+" + "-" * width + "+"
    rows = [border] + ["|" + "".join(r) + "|" for r in grid] + [border]
    b = world.body
    rows.append(
        f"тик {world.tick}  E={b.energy:.3f}  v={b.speed:5.2f}  "
        f"w={b.omega:+5.2f}  смертей={world.deaths}  "
        f"съедено A/B={world.eaten_counts[list(world.eaten_counts)[0]]}/"
        f"{world.eaten_counts[list(world.eaten_counts)[1]]}"
    )
    return "\n".join(rows)


def retina_view(world: World, record: TruthRecord | None = None) -> str:
    """Сетчатка: транзиентный и устойчивый каналы плюс разметка истины.

    Строка `истина` рисуется ИЗ КАНАЛА ИСТИНЫ и агенту недоступна. Она здесь
    ровно для того, чтобы человек мог сравнить, что видит агент, с тем, что
    есть на самом деле.
    """
    proj = world._last_projection
    n = world.cfg.retina_n

    def bar(values: np.ndarray, lo: float, hi: float) -> str:
        out = []
        for v in values:
            f = 0.0 if hi <= lo else (float(v) - lo) / (hi - lo)
            f = min(1.0, max(0.0, f))
            out.append(SHADES[int(f * (len(SHADES) - 1))])
        return "".join(out)

    lines = [
        "     " + "".join(str((i // 10) % 10) if i % 10 == 0 else " " for i in range(n)),
        "L   |" + bar(proj.l, 0.0, 3.1) + "|  яркость",
        "C   |" + bar(proj.c, -1.0, 1.0) + "|  цвет (A светлый / B тёмный)",
    ]

    g = n // world.cfg.sustained_n
    sus_l = np.repeat(world.retina.sustained[:, 0], g)
    sus_c = np.repeat(world.retina.sustained[:, 1], g)
    lines.append("sL  |" + bar(sus_l, 0.0, 3.1) + "|  устойчивый, яркость")
    lines.append("sC  |" + bar(sus_c, -1.0, 1.0) + "|  устойчивый, цвет")

    owner = ["·"] * n
    if record is not None:
        for it in record.items:
            if it.retinal_span is None or not it.visible:
                continue
            lo, hi = it.retinal_span
            for i in range(max(0, lo), min(n - 1, hi) + 1):
                owner[i] = it.kind
    lines.append("ист |" + "".join(owner) + "|  канал истины (агенту НЕ виден)")
    return "\n".join(lines)


def graph_view(report: dict | None) -> str:
    """Третий обязательный вид. Данные о графе даёт агент — мир в него не лезит,
    поэтому здесь только форматирование того, что дали."""
    if not report:
        return "граф: агент не сообщил состояние (мир в него не заглядывает)"
    parts = [f"узлов: {report.get('nodes', '?')}"]
    if "by_type" in report:
        parts.append("типы: " + ", ".join(f"{k}={v}" for k, v in report["by_type"].items()))
    if "activity" in report:
        parts.append(f"активность: {report['activity']:.3f}")
    return "граф: " + "  ".join(parts)


def frame(world: World, record: TruthRecord | None = None,
          trail: list[tuple[float, float]] | None = None,
          graph_report: dict | None = None) -> str:
    return "\n".join([
        world_view(world, trail=trail),
        "",
        retina_view(world, record),
        "",
        graph_view(graph_report),
    ])
