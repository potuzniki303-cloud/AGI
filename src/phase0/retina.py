"""Сенсорный тракт: сетчатка P0, транзиентный и устойчивый каналы. Часть 4.

Уровень P0 — одномерная полоса на 32 рецептора, FOV 120°. Он сохраняет всю
задачу связывания целиком (несколько объектов в поле зрения, окклюзия,
идентичность не дана, движение по сетчатке), но объём входа мизерный и на
экране всё видно глазами при отладке.

Про две шкалы времени. Разделение зрения на быстрый транзиентный и медленный
устойчивый канал существует ЗДЕСЬ, на входе, ещё до всякой обработки. Если
разброс временных масштабов действительно даёт преимущество, оно обязано
проявиться уже на этом уровне.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Config, Kind
from .events import Channels, Event
from .items import Item

# Опорное расстояние спада яркости и порог темноты. Оба ПРОИЗВОЛ: спецификация
# задаёт только «яркость в лог-шкале», не саму фотометрию.
D_REF = 16.0
I_FLOOR = 0.05


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Projection:
    """Что попало на сетчатку в этом тике.

    l, c            — (N,) яркость и цветооппонентная ось
    owner           — (N,) item_id ближайшего предмета в рецепторе, -1 если пусто
    spans           — item_id -> (first, last) занятый диапазон рецепторов
    occluded_by     — item_id -> item_id того, кто его перекрыл (или -1)
    """

    __slots__ = ("l", "c", "owner", "spans", "occluded_by")

    def __init__(self, n: int) -> None:
        self.l = np.zeros(n, dtype=np.float64)
        self.c = np.zeros(n, dtype=np.float64)
        self.owner = np.full(n, -1, dtype=np.int64)
        self.spans: dict[int, tuple[int, int]] = {}
        self.occluded_by: dict[int, int] = {}


class Retina:
    """Проекция мира на полосу рецепторов плюс два канала кодирования."""

    def __init__(self, cfg: Config, channels: Channels) -> None:
        self.cfg = cfg
        self.ch = channels
        n = cfg.retina_n
        self.n = n
        self.fov = math.radians(cfg.fov_deg)
        self.half_fov = self.fov / 2.0
        self.bin = self.fov / n

        # Транзиентный канал: опорное значение и тик последнего события.
        self.l_ref = np.zeros((n, cfg.color_channels), dtype=np.float64)
        self.t_last = np.full((n, cfg.color_channels), -10 ** 9, dtype=np.int64)

        # Устойчивый канал: ФНЧ по огрублённой сетке.
        self.sustained = np.zeros((cfg.sustained_n, cfg.color_channels), dtype=np.float64)
        self._alpha_sustained = cfg.dt / (cfg.tau_sustained + cfg.dt)
        if cfg.sustained_n > n or n % cfg.sustained_n:
            raise ValueError("sustained_n должен делить retina_n нацело")
        self._group = n // cfg.sustained_n

        self.dropped_events_total = 0

    # ------------------------------------------------------------ проекция
    def project(self, bx: float, by: float, theta: float,
                items: list[Item]) -> Projection:
        """Нарисовать предметы на полосе. Ближний перекрывает дальнего."""
        proj = Projection(self.n)
        depth = np.full(self.n, np.inf, dtype=np.float64)

        if self.cfg.walls_visible:
            self._paint_walls(proj, depth, bx, by, theta)

        # Дальние первыми: алгоритм художника. Порядок по (расстояние, item_id) —
        # item_id в ключе, чтобы равные расстояния не зависели от порядка в списке
        # и реплей оставался побитовым.
        ordered = sorted(
            items,
            key=lambda it: (-math.hypot(it.x - bx, it.y - by), -it.item_id),
        )
        for item in ordered:
            self._paint_item(proj, depth, item, bx, by, theta)
        return proj

    def _paint_item(self, proj: Projection, depth: np.ndarray, item: Item,
                    bx: float, by: float, theta: float) -> None:
        dx, dy = item.x - bx, item.y - by
        dist = math.hypot(dx, dy)
        r = self.cfg.r_item
        if dist <= 1e-9:
            return

        bearing = _wrap(math.atan2(dy, dx) - theta)
        half_width = math.asin(min(1.0, r / dist)) if dist > r else self.half_fov
        lo_ang, hi_ang = bearing - half_width, bearing + half_width

        if hi_ang < -self.half_fov or lo_ang > self.half_fov:
            return  # целиком за краем поля зрения

        lo = max(0, int(math.floor((lo_ang + self.half_fov) / self.bin)))
        hi = min(self.n - 1, int(math.ceil((hi_ang + self.half_fov) / self.bin)) - 1)
        if hi < lo:
            hi = lo
        if lo > self.n - 1 or hi < 0:
            return

        intensity = 1.0 / (1.0 + dist / D_REF)
        lum = math.log(intensity + I_FLOOR) - math.log(I_FLOOR)
        col = intensity * (1.0 if item.kind is Kind.A else -1.0)

        painted = False
        for i in range(lo, hi + 1):
            if dist < depth[i]:
                prev_owner = int(proj.owner[i])
                if prev_owner >= 0:
                    proj.occluded_by[prev_owner] = item.item_id
                depth[i] = dist
                proj.l[i] = lum
                proj.c[i] = col
                proj.owner[i] = item.item_id
                painted = True

        proj.spans[item.item_id] = (lo, hi)
        if not painted:
            # Предмет в поле зрения, но целиком закрыт.
            proj.occluded_by.setdefault(item.item_id, int(proj.owner[lo]))

    def _paint_walls(self, proj: Projection, depth: np.ndarray,
                     bx: float, by: float, theta: float) -> None:
        """Стены как тусклый ахроматический фон. Выключено по умолчанию:
        спецификация не вводит стены в зрение, и включать их — значит добавить
        ориентиры, которых в ней нет."""
        w, h = self.cfg.arena
        for i in range(self.n):
            ang = theta - self.half_fov + (i + 0.5) * self.bin
            dist = _ray_to_box(bx, by, ang, w, h)
            if not math.isfinite(dist):
                continue
            intensity = 0.25 / (1.0 + dist / D_REF)
            depth[i] = dist
            proj.l[i] = math.log(intensity + I_FLOOR) - math.log(I_FLOOR)
            proj.c[i] = 0.0

    # ------------------------------------------------- транзиентный канал
    def transient_events(self, tick: int, proj: Projection) -> tuple[list[Event], int]:
        """Событийное кодирование (4.4).

        Статичная сцена даёт ноль событий — разреженность возникает физически,
        а не вводится регуляризатором. Возвращает (события, сколько отброшено).
        """
        cfg = self.cfg
        theta_thr = cfg.theta_event
        field = np.stack([proj.l, proj.c], axis=1)
        delta = field - self.l_ref
        mag = np.abs(delta)

        ready = (tick - self.t_last) > cfg.refractory_ticks
        fire = (mag > theta_thr) & ready
        idx = np.argwhere(fire)
        dropped = 0

        cap = int(round(cfg.event_rate_cap * self.ch.n_transient))
        if len(idx) > cap:
            # Приоритет по |ΔL|: сохраняем самые сильные изменения.
            order = np.argsort(-mag[fire], kind="stable")
            keep = order[:cap]
            dropped = len(idx) - cap
            idx = idx[keep]

        events: list[Event] = []
        for i, c in idx:
            i, c = int(i), int(c)
            d = delta[i, c]
            polarity = 1 if d > 0.0 else 0
            events.append(
                Event(
                    t=tick,
                    channel=self.ch.transient(i, c, polarity),
                    value=float(abs(d)),
                    sign=1 if polarity else -1,
                )
            )
            self.l_ref[i, c] = field[i, c]
            self.t_last[i, c] = tick

        self.dropped_events_total += dropped
        return events, dropped

    # ---------------------------------------------------- устойчивый канал
    def sustained_events(self, tick: int, proj: Projection) -> list[Event]:
        """Плотный медленный канал: абсолютная яркость и цвет, которых
        транзиентный канал принципиально не содержит."""
        g = self._group
        coarse_l = proj.l.reshape(-1, g).mean(axis=1)
        coarse_c = proj.c.reshape(-1, g).mean(axis=1)
        target = np.stack([coarse_l, coarse_c], axis=1)
        self.sustained += self._alpha_sustained * (target - self.sustained)

        events = []
        for i in range(self.cfg.sustained_n):
            for c in range(self.cfg.color_channels):
                events.append(
                    Event(tick, self.ch.sustained(i, c), float(self.sustained[i, c]))
                )
        return events

    # ------------------------------------------------------------- снапшот
    def state(self) -> dict:
        return {
            "l_ref": self.l_ref.copy(),
            "t_last": self.t_last.copy(),
            "sustained": self.sustained.copy(),
            "dropped": self.dropped_events_total,
        }

    def restore(self, state: dict) -> None:
        self.l_ref = state["l_ref"].copy()
        self.t_last = state["t_last"].copy()
        self.sustained = state["sustained"].copy()
        self.dropped_events_total = state["dropped"]


def _ray_to_box(x: float, y: float, ang: float, w: float, h: float) -> float:
    """Расстояние от точки внутри прямоугольника до стены вдоль луча."""
    cx, sy = math.cos(ang), math.sin(ang)
    best = math.inf
    if abs(cx) > 1e-12:
        for wall_x in (0.0, w):
            t = (wall_x - x) / cx
            if t > 0:
                yy = y + t * sy
                if -1e-9 <= yy <= h + 1e-9:
                    best = min(best, t)
    if abs(sy) > 1e-12:
        for wall_y in (0.0, h):
            t = (wall_y - y) / sy
            if t > 0:
                xx = x + t * cx
                if -1e-9 <= xx <= w + 1e-9:
                    best = min(best, t)
    return best
