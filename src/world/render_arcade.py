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

Два режима.

  ОКОННЫЙ (visible=True, fullscreen=False) — окно под размер мира, всё поле
  видно целиком. Годится примерно до 40x40, дальше окно перестаёт влезать
  в экран.

  ПОЛНОЭКРАННЫЙ (fullscreen=True) — для больших миров, сотни на сотни клеток.
  Экран показывает кусок мира, камера ездит. Рисуется только видимое:
  поле 500x500 это четверть миллиона клеток, и обходить их каждый кадр
  нельзя, поэтому из массива еды вырезается ровно видимое окно.

Управление в полноэкранном режиме:

  WASD, стрелки  — двигать камеру
  Shift          — двигать быстрее
  +  -           — приблизить/отдалить
  F              — следовать за телом (в большом мире иначе никого не найти)
  TAB            — следующее тело
  M              — миникарта
  G              — сетка
  ESC, Q         — выход

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
COLOR_MINIMAP_BG = (28, 12, 40)
COLOR_MINIMAP_VIEW = (200, 180, 220)

FOOD_COLORS = {1: (COLOR_FOOD, COLOR_FOOD_DARK), 2: (COLOR_FOOD2, COLOR_FOOD2_DARK)}

MIN_CELL = 2
MAX_CELL = 64
PAN_CELLS_PER_SEC = 18.0
MINIMAP_MAX = 220


class Camera:
    """Положение камеры в КЛЕТКАХ мира, не в пикселях.

    Клетки, а не пиксели, потому что зум меняет размер клетки, и при
    пиксельной камере любое приближение уводило бы вид в сторону.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, cell: int = CELL) -> None:
        self.x = x
        self.y = y
        self.cell = cell
        self.follow_id: int | None = None

    def clamp_to(self, world: World, view_w: int, view_h: int) -> None:
        """На торе камера ездит по кругу, у стен упирается."""
        size = world.cfg.size
        if world.cfg.boundary == "wrap":
            self.x %= size
            self.y %= size
        else:
            self.x = min(max(self.x, 0.0), max(0.0, size - view_w))
            self.y = min(max(self.y, 0.0), max(0.0, size - view_h))


class ArcadeRenderer:
    """Окно arcade поверх мира. Создаётся лениво, при первом кадре."""

    def __init__(
        self,
        world: World,
        visible: bool = True,
        fps: float = 15.0,
        fullscreen: bool = False,
        cell: int | None = None,
    ) -> None:
        import arcade

        self.arcade = arcade
        self.world = world
        self.fps = fps
        self.fullscreen = fullscreen
        self._last_frame = 0.0
        self._last_pan = time.perf_counter()
        self.show_minimap = True
        self.show_grid = True
        self.closed = False
        self._offset = (0.0, 0.0)  # левый верхний угол поля, считается за кадр

        size = world.cfg.size

        if fullscreen:
            self.window = arcade.Window(
                fullscreen=True, visible=visible, vsync=False,
                title=f"{type(world).__name__} {size}x{size}",
            )
            start_cell = cell if cell is not None else self._fit_cell(size)
            self.camera = Camera(0.0, 0.0, start_cell)
            self.side = None  # в полноэкранном размер окна задаёт экран
        else:
            self.camera = Camera(0.0, 0.0, cell or CELL)
            self.side = size * self.camera.cell + 2 * WALL
            self.window = arcade.Window(
                self.side, self.side + HUD,
                f"{type(world).__name__} {size}x{size}",
                visible=visible, vsync=False,
            )

        self._keys = self._install_key_handler()
        self._text_cache: dict[str, Any] = {}

    def _fit_cell(self, size: int) -> int:
        """Подобрать размер клетки так, чтобы мир влез в экран целиком.

        Если не влезает даже при минимальном размере — берём минимальный и
        дальше живём панорамой.
        """
        usable = min(self.window.width, self.window.height - HUD)
        return int(min(MAX_CELL, max(MIN_CELL, usable // max(1, size))))

    def _install_key_handler(self):
        import pyglet

        keys = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(keys)
        self.window.push_handlers(on_key_press=self._on_key_press)
        return keys

    # ------------------------------------------------------------ ввод

    def _on_key_press(self, symbol: int, modifiers: int) -> bool:
        key = self.arcade.key
        cam = self.camera

        if symbol in (key.ESCAPE, key.Q):
            self.close()
        elif symbol in (key.PLUS, key.EQUAL, key.NUM_ADD):
            cam.cell = int(min(MAX_CELL, max(MIN_CELL, cam.cell + max(1, cam.cell // 4))))
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            cam.cell = int(min(MAX_CELL, max(MIN_CELL, cam.cell - max(1, cam.cell // 4))))
        elif symbol == key.F:
            cam.follow_id = None if cam.follow_id is not None else self._first_body_id()
        elif symbol == key.TAB:
            cam.follow_id = self._next_body_id(cam.follow_id)
        elif symbol == key.M:
            self.show_minimap = not self.show_minimap
        elif symbol == key.G:
            self.show_grid = not self.show_grid
        return False

    def _first_body_id(self) -> int | None:
        return self.world.bodies[0].id if self.world.bodies else None

    def _next_body_id(self, current: int | None) -> int | None:
        ids = [b.id for b in self.world.bodies]
        if not ids:
            return None
        if current is None or current not in ids:
            return ids[0]
        return ids[(ids.index(current) + 1) % len(ids)]

    def _pan(self) -> None:
        """Плавное движение камеры по удерживаемым клавишам."""
        import pyglet

        key = pyglet.window.key
        now = time.perf_counter()
        dt = min(0.2, now - self._last_pan)
        self._last_pan = now

        k = self._keys
        dx = (k[key.D] or k[key.RIGHT]) - (k[key.A] or k[key.LEFT])
        dy = (k[key.S] or k[key.DOWN]) - (k[key.W] or k[key.UP])
        if not dx and not dy:
            return

        # Ручное движение отменяет слежение: иначе камера дёргалась бы
        # между тем, куда её ведут, и телом, за которым она следит.
        self.camera.follow_id = None

        speed = PAN_CELLS_PER_SEC * (3.0 if (k[key.LSHIFT] or k[key.RSHIFT]) else 1.0)
        self.camera.x += dx * speed * dt
        self.camera.y += dy * speed * dt

    def _follow(self, view_w: int, view_h: int) -> None:
        if self.camera.follow_id is None:
            return
        for b in self.world.bodies:
            if b.id == self.camera.follow_id:
                self.camera.x = b.x - view_w / 2.0
                self.camera.y = b.y - view_h / 2.0
                return
        # тело умерло — переключаемся на следующее, а не теряем камеру
        self.camera.follow_id = self._first_body_id()

    # ---------------------------------------------------------------- кадр

    def draw(self, throttle: bool = True) -> np.ndarray | None:
        if self.closed:
            return None

        arcade = self.arcade
        world = self.world
        size = world.cfg.size

        if not self.fullscreen:
            # Мир может расти по ходу прогона (GrowingWorld), поэтому размер
            # окна не константа: подгоняем, если сторона выросла.
            side = size * self.camera.cell + 2 * WALL
            if side != self.side:
                self.side = side
                self.window.set_size(side, side + HUD)

        self.window.switch_to()
        self.window.dispatch_events()
        if self.closed:
            return None

        view_w, view_h = self._viewport_cells()
        self._pan()
        self._follow(view_w, view_h)
        self.camera.clamp_to(world, view_w, view_h)
        self._offset = self._field_offset(view_w, view_h)

        self.window.clear(COLOR_WALL)
        self._draw_field(view_w, view_h)
        self._draw_food(view_w, view_h)
        self._draw_bodies(view_w, view_h)
        # когда мир виден целиком, миникарта дублирует экран — не рисуем
        whole = view_w >= size and view_h >= size
        if self.fullscreen and self.show_minimap and not whole:
            self._draw_minimap(view_w, view_h)
        self._draw_hud(view_w, view_h)

        self.window.flip()

        if throttle:
            wait = 1.0 / self.fps - (time.perf_counter() - self._last_frame)
            if wait > 0:
                time.sleep(wait)
            self._last_frame = time.perf_counter()
            return None

        w, h = self.window.width, self.window.height
        return np.asarray(arcade.get_image(0, 0, w, h))[:, :, :3]

    # ------------------------------------------------------------ геометрия

    def _viewport_cells(self) -> tuple[int, int]:
        """Сколько клеток влезает на экран. Не больше, чем есть в мире."""
        cell = self.camera.cell
        margin = 0 if self.fullscreen else WALL
        w = (self.window.width - 2 * margin) // cell
        h = (self.window.height - HUD - 2 * margin) // cell
        size = self.world.cfg.size
        return max(1, min(int(w), size)), max(1, min(int(h), size))

    def _origin(self) -> tuple[int, int]:
        """Левый верхний угол вида в клетках мира."""
        return int(np.floor(self.camera.x)), int(np.floor(self.camera.y))

    def _field_offset(self, view_w: int, view_h: int) -> tuple[float, float]:
        """Левый верхний угол поля в пикселях.

        Если мир целиком влезает в экран, поле центрируется: мир квадратный,
        экран широкий, и прижатое к краю поле оставляло бы полэкрана пустым.
        """
        cell = self.camera.cell
        margin = 0 if self.fullscreen else WALL
        avail_w = self.window.width - 2 * margin
        avail_h = self.window.height - HUD - 2 * margin
        left = margin + max(0, (avail_w - view_w * cell) // 2)
        top = self.window.height - HUD - margin - max(0, (avail_h - view_h * cell) // 2)
        return left, top

    def _to_screen(self, sx: float, sy: float) -> tuple[float, float]:
        """Экранная позиция клетки вида (sx, sy). Ряд 0 сверху."""
        cell = self.camera.cell
        left, top = self._offset
        return left + sx * cell + cell / 2, top - sy * cell - cell / 2

    def _visible_rows_cols(self, view_w: int, view_h: int):
        """Индексы строк и столбцов мира, попадающие в вид.

        На торе идут по кругу, у стен обрезаются. Именно здесь появляется
        отсечение: дальше по массиву мы ходим только этими индексами, а не
        по всему полю.
        """
        size = self.world.cfg.size
        x0, y0 = self._origin()
        rows = np.arange(y0, y0 + view_h)
        cols = np.arange(x0, x0 + view_w)
        if self.world.cfg.boundary == "wrap":
            return rows % size, cols % size
        return rows[(rows >= 0) & (rows < size)], cols[(cols >= 0) & (cols < size)]

    # ------------------------------------------------------------ рисование

    def _draw_field(self, view_w: int, view_h: int) -> None:
        arcade = self.arcade
        cell = self.camera.cell

        w_px, h_px = view_w * cell, view_h * cell
        left, top = self._offset
        arcade.draw_rect_filled(
            arcade.XYWH(left + w_px / 2, top - h_px / 2, w_px, h_px), COLOR_BG
        )

        if not self.show_grid or cell < 8:
            return
        for i in range(1, view_w):
            x = left + i * cell
            arcade.draw_line(x, top - h_px, x, top, COLOR_LINE, 1)
        for i in range(1, view_h):
            y = top - i * cell
            arcade.draw_line(left, y, left + w_px, y, COLOR_LINE, 1)

    def _draw_food(self, view_w: int, view_h: int) -> None:
        arcade = self.arcade
        rows, cols = self._visible_rows_cols(view_w, view_h)
        if rows.size == 0 or cols.size == 0:
            return

        sub = self.world.food[np.ix_(rows, cols)]
        sy, sx = np.nonzero(sub)
        if sy.size == 0:
            return

        r = max(1.0, self.camera.cell * 0.28)
        outline = self.camera.cell >= 10
        for j, i in zip(sy, sx):
            ftype = int(sub[j, i])
            fill, edge = FOOD_COLORS.get(ftype, (COLOR_FOOD, COLOR_FOOD_DARK))
            cx, cy = self._to_screen(i, j)
            arcade.draw_circle_filled(cx, cy, r, fill)
            if outline:
                arcade.draw_circle_outline(cx, cy, r, edge, 2)

    def _draw_bodies(self, view_w: int, view_h: int) -> None:
        arcade = self.arcade
        world = self.world
        size = world.cfg.size
        wrap = world.cfg.boundary == "wrap"
        x0, y0 = self._origin()
        cell = self.camera.cell
        e_max = max(world.cfg.energy_max, 1e-6)
        gens = [b.generation for b in world.bodies]
        g_max = max(gens) if gens else 0
        box = max(2.0, cell - max(2, cell // 4))
        detail = cell >= 8

        for body in world.bodies:
            dx = (body.x - x0) % size if wrap else body.x - x0
            dy = (body.y - y0) % size if wrap else body.y - y0
            if not (0 <= dx < view_w and 0 <= dy < view_h):
                continue

            cx, cy = self._to_screen(dx, dy)
            k = 0.30 + 0.70 * float(np.clip(body.energy / e_max, 0.0, 1.0))
            fill = tuple(int(c * k) for c in COLOR_AGENT)
            arcade.draw_rect_filled(arcade.XYWH(cx, cy, box, box), fill)

            if detail and g_max > 0:
                t = body.generation / g_max
                edge = (int(80 + 150 * t), int(60 + 120 * t), int(120 + 100 * t))
                arcade.draw_rect_outline(arcade.XYWH(cx, cy, box, box), edge, 2)

            if body.action_map is not None or body.sensor_mask is not None:
                arcade.draw_rect_outline(
                    arcade.XYWH(cx, cy, box + 4, box + 4), COLOR_BROKEN, 2
                )

            if body.id == self.camera.follow_id:
                arcade.draw_rect_outline(
                    arcade.XYWH(cx, cy, box + 8, box + 8), COLOR_MINIMAP_VIEW, 2
                )

    def _draw_minimap(self, view_w: int, view_h: int) -> None:
        """Весь мир целиком в углу. Без неё в поле 500x500 не сориентироваться."""
        arcade = self.arcade
        world = self.world
        size = world.cfg.size

        scale = max(1, int(np.ceil(size / MINIMAP_MAX)))
        px = size // scale
        pad = 12
        left = self.window.width - px - pad
        bottom = pad

        arcade.draw_rect_filled(
            arcade.XYWH(left + px / 2, bottom + px / 2, px + 6, px + 6),
            COLOR_MINIMAP_BG,
        )

        # тела точками; еду на миникарте не рисуем — она сливается в шум
        for body in world.bodies:
            mx = left + (body.x // scale)
            my = bottom + px - (body.y // scale)
            arcade.draw_point(mx, my, COLOR_AGENT, 2)

        # рамка текущего вида
        vx = left + (self.camera.x / scale)
        vy = bottom + px - (self.camera.y / scale)
        vw = max(2.0, view_w / scale)
        vh = max(2.0, view_h / scale)
        arcade.draw_rect_outline(
            arcade.XYWH(vx + vw / 2, vy - vh / 2, vw, vh), COLOR_MINIMAP_VIEW, 1
        )

    def _draw_hud(self, view_w: int, view_h: int) -> None:
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
        if self.fullscreen:
            parts.append(
                f"камера=({int(self.camera.x)},{int(self.camera.y)})"
                f" вид={view_w}x{view_h} из {world.cfg.size}"
            )
            if self.camera.follow_id is not None:
                parts.append(f"следим за #{self.camera.follow_id}")
        self._draw_text(" | ".join(parts), 8, self.window.height - HUD + 6)

        if self.fullscreen:
            self._draw_text(
                "WASD/стрелки — камера, Shift — быстрее, +/− — зум, "
                "F — следовать, TAB — следующий, M — карта, G — сетка, ESC — выход",
                8, 6,
            )

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
        if self.closed:
            return
        self.closed = True
        if self.window is not None:
            self.window.close()
            self.window = None
