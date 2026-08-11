"""Отрисовка мира через arcade.

Палитра взята из твоего src/env/grid_world.py, чтобы новые миры выглядели
продолжением старого, а не чужой программой.

Что показано сверх старого рендера — и почему именно это:

  яркость тела   — энергия. Тусклое тело близко к смерти. Видно, кто
                   доедает последнее, а кто жирует, без единой цифры.
  обводка        — поколение. Чем позже родился, тем светлее ободок.
                   Волна светлых тел по экрану это и есть смена поколений.
  цвет еды       — тип. В two_foods их два, и они внешне различимы, потому
                   что агент их тоже различает; что за ними стоит, на вид
                   не определить ни ему, ни тебе.
  красная рамка  — тело, которому сломали сенсомоторику (мир shift).
                   Агенту про это не сообщают, а тебе видно.

Arcade требует дисплей. На headless-машине запускай под виртуальным:

    xvfb-run -a python3 experiments/watch.py --world forage
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .core import World

CELL = 28
WALL = 16
HUD = 26

COLOR_WALL = (18, 7, 26)
COLOR_BG = (46, 18, 60)
COLOR_LINE = (62, 26, 80)
COLOR_FOOD = (96, 214, 118)
COLOR_FOOD_DARK = (46, 140, 70)
COLOR_FOOD2 = (214, 118, 96)
COLOR_FOOD2_DARK = (140, 70, 46)
COLOR_AGENT = (245, 245, 250)
COLOR_TEXT = (200, 180, 220)
COLOR_BROKEN = (230, 80, 80)

FOOD_COLORS = {1: (COLOR_FOOD, COLOR_FOOD_DARK), 2: (COLOR_FOOD2, COLOR_FOOD2_DARK)}


class ArcadeRenderer:
    """Окно arcade поверх мира. Создаётся лениво, при первом кадре."""

    def __init__(self, world: World, visible: bool = True, fps: float = 15.0) -> None:
        import arcade

        self.arcade = arcade
        self.world = world
        self.fps = fps
        self._last_frame = 0.0

        size = world.cfg.size
        self.side = size * CELL + 2 * WALL
        self.window = arcade.Window(
            self.side,
            self.side + HUD,
            f"{type(world).__name__} {size}x{size}",
            visible=visible,
            vsync=False,
        )
        self._text_cache: dict[str, Any] = {}

    # ---------------------------------------------------------------- кадр

    def draw(self, throttle: bool = True) -> np.ndarray | None:
        arcade = self.arcade
        world = self.world
        size = world.cfg.size

        self.window.switch_to()
        self.window.dispatch_events()
        self.window.clear(COLOR_WALL)

        # поле
        arcade.draw_rect_filled(
            arcade.XYWH(self.side / 2, self.side / 2, size * CELL, size * CELL),
            COLOR_BG,
        )

        # сетка
        for i in range(1, size):
            p = WALL + i * CELL
            arcade.draw_line(p, WALL, p, self.side - WALL, COLOR_LINE, 1)
            arcade.draw_line(WALL, p, self.side - WALL, p, COLOR_LINE, 1)

        self._draw_food()
        self._draw_bodies()
        self._draw_hud()

        self.window.flip()

        if throttle:
            wait = 1.0 / self.fps - (time.perf_counter() - self._last_frame)
            if wait > 0:
                time.sleep(wait)
            self._last_frame = time.perf_counter()
            return None

        return np.asarray(arcade.get_image(0, 0, self.side, self.side + HUD))[:, :, :3]

    def _cell_centre(self, x: int, y: int) -> tuple[float, float]:
        # ряд 0 сверху, как в старом рендере
        cx = WALL + x * CELL + CELL / 2
        cy = self.side - WALL - y * CELL - CELL / 2
        return cx, cy

    def _draw_food(self) -> None:
        arcade = self.arcade
        ys, xs = np.nonzero(self.world.food)
        for y, x in zip(ys, xs):
            ftype = int(self.world.food[y, x])
            fill, edge = FOOD_COLORS.get(ftype, (COLOR_FOOD, COLOR_FOOD_DARK))
            cx, cy = self._cell_centre(int(x), int(y))
            arcade.draw_circle_filled(cx, cy, CELL * 0.28, fill)
            arcade.draw_circle_outline(cx, cy, CELL * 0.28, edge, 2)

    def _draw_bodies(self) -> None:
        arcade = self.arcade
        world = self.world
        e_max = max(world.cfg.energy_max, 1e-6)
        gens = [b.generation for b in world.bodies]
        g_max = max(gens) if gens else 0

        for body in world.bodies:
            cx, cy = self._cell_centre(body.x, body.y)

            # яркость по энергии: тусклый = вот-вот умрёт
            k = 0.30 + 0.70 * float(np.clip(body.energy / e_max, 0.0, 1.0))
            fill = tuple(int(c * k) for c in COLOR_AGENT)

            arcade.draw_rect_filled(
                arcade.XYWH(cx, cy, CELL - 8, CELL - 8), fill
            )

            # обводка по поколению
            if g_max > 0:
                t = body.generation / g_max
                edge = (int(80 + 150 * t), int(60 + 120 * t), int(120 + 100 * t))
                arcade.draw_rect_outline(
                    arcade.XYWH(cx, cy, CELL - 8, CELL - 8), edge, 2
                )

            # сломанная сенсомоторика (shift)
            if body.action_map is not None or body.sensor_mask is not None:
                arcade.draw_rect_outline(
                    arcade.XYWH(cx, cy, CELL - 2, CELL - 2), COLOR_BROKEN, 2
                )

    def _draw_hud(self) -> None:
        world = self.world
        bodies = world.bodies
        parts = [
            f"t={world.tick}",
            f"живых={len(bodies)}",
            f"еды={int(np.count_nonzero(world.food))}",
        ]
        if bodies:
            parts.append(f"энергия={np.mean([b.energy for b in bodies]):.0f}")
            parts.append(f"поколение={max(b.generation for b in bodies)}")
        if world.extinct:
            parts.append("ВЫМЕРЛИ")

        self._draw_text(" | ".join(parts), 8, self.side + 6)

    def _draw_text(self, text: str, x: float, y: float) -> None:
        """arcade.Text дорогой в создании, поэтому объекты переиспользуются."""
        key = f"{x}:{y}"
        obj = self._text_cache.get(key)
        if obj is None:
            obj = self.arcade.Text(text, x, y, COLOR_TEXT, 12)
            self._text_cache[key] = obj
        else:
            obj.text = text
        obj.draw()

    def close(self) -> None:
        if self.window is not None:
            self.window.close()
            self.window = None
