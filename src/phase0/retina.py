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

# Фотометрия переехала в Config (поля `d_ref` и `i_floor`). Держать её
# модульными константами было ошибкой: спецификация требует, чтобы meta.json
# содержал ПОЛНЫЙ снимок констант, а числа, живущие вне Config, в него не
# попадают — то есть защита от тихой подкрутки на них не действует.
# Значения по умолчанию не изменились.


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Projection:
    """Что попало на сетчатку в этом тике.

    l, c        — (N,) яркость и цветооппонентная ось
    owner       — (N,) item_id ближайшего предмета в рецепторе, -1 если пусто
    spans       — item_id -> (first, last): куда предмет ПОПАЛ БЫ, то есть его
                  угловой размер, обрезанный полем зрения
    visible     — item_id -> сколько рецепторов он реально занял
    occluded_by — item_id -> кто его закрыл, -1 если никто
    occlusion   — item_id -> "none" | "partial" | "full"

    Разделение `spans` и `visible` принципиально. Раньше `spans` писался
    безусловно, и `visible` означал «в поле зрения», а не «видно»: за 35 447
    записей с occluded_by >= 0 ВСЕ были помечены видимыми. На таких данных
    три метрики связывания из пяти (склеивание, дробление, восстановление
    после окклюзии) посчитать нельзя.
    """

    __slots__ = ("l", "c", "owner", "spans", "visible", "occluded_by", "occlusion")

    def __init__(self, n: int) -> None:
        self.l = np.zeros(n, dtype=np.float64)
        self.c = np.zeros(n, dtype=np.float64)
        self.owner = np.full(n, -1, dtype=np.int64)
        self.spans: dict[int, tuple[int, int]] = {}
        self.visible: dict[int, int] = {}
        self.occluded_by: dict[int, int] = {}
        self.occlusion: dict[int, str] = {}

    def visible_receptors(self, item_id: int) -> int:
        return self.visible.get(item_id, 0)


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
        self._sustained_ref = np.zeros_like(self.sustained)
        self._alpha_sustained = cfg.dt / (cfg.tau_sustained + cfg.dt)
        if cfg.sustained_n > n or n % cfg.sustained_n:
            raise ValueError("sustained_n должен делить retina_n нацело")
        self._group = n // cfg.sustained_n

        self.dropped_events_total = 0

    # ------------------------------------------------------------ проекция
    def project(self, bx: float, by: float, theta: float,
                items: list[Item], obstacles=()) -> Projection:
        """Нарисовать сцену на полосе. Ближний перекрывает дальнего.

        Препятствия рисуются наравне с предметами: они непроходимы И
        загораживают обзор. Ахроматические (C=0), поэтому цветовая ось
        по-прежнему разделяет ровно kind A и kind B.
        """
        proj = Projection(self.n)
        depth = np.full(self.n, np.inf, dtype=np.float64)

        if self.cfg.walls_visible:
            self._paint_walls(proj, depth, bx, by, theta)

        # Дальние первыми: алгоритм художника. Порядок по (расстояние, item_id) —
        # item_id в ключе, чтобы равные расстояния не зависели от порядка в списке
        # и реплей оставался побитовым.
        drawable = [(it, it.item_id, it.kind, self.cfg.r_item) for it in items]
        drawable += [(ob, ob.obstacle_id, None, ob.radius) for ob in obstacles]
        ordered = sorted(
            drawable,
            key=lambda d: (-math.hypot(d[0].x - bx, d[0].y - by), -d[1]),
        )
        for shape, sid, kind, radius in ordered:
            self._paint_shape(proj, depth, shape, sid, kind, radius, bx, by, theta)

        self._classify_occlusion(proj)
        return proj

    @staticmethod
    def _classify_occlusion(proj: Projection) -> None:
        """Кто кем закрыт — считается ПОСЛЕ всей отрисовки.

        На лету это считать нельзя: предмет, закрашенный первым, может быть
        перекрыт позже, и «частично» от «полностью» уже не отличить.
        """
        for item_id, (lo, hi) in proj.spans.items():
            window = proj.owner[lo:hi + 1]
            seen = int((window == item_id).sum())
            width = hi - lo + 1
            proj.visible[item_id] = seen

            if seen == width:
                proj.occlusion[item_id] = "none"
                proj.occluded_by[item_id] = -1
                continue

            proj.occlusion[item_id] = "full" if seen == 0 else "partial"
            others = window[(window != item_id) & (window >= 0)]
            proj.occluded_by[item_id] = int(np.bincount(others).argmax()) if others.size else -1

    def _paint_shape(self, proj: Projection, depth: np.ndarray, shape,
                     shape_id: int, kind, radius: float,
                     bx: float, by: float, theta: float) -> None:
        dx, dy = shape.x - bx, shape.y - by
        dist = math.hypot(dx, dy)
        r = radius
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

        cfg = self.cfg
        intensity = 1.0 / (1.0 + dist / cfg.d_ref)
        lum = math.log(intensity + cfg.i_floor) - math.log(cfg.i_floor)
        # Препятствие (kind=None) ахроматично: C=0.
        col = 0.0 if kind is None else intensity * (1.0 if kind is Kind.A else -1.0)

        for i in range(lo, hi + 1):
            if dist < depth[i]:
                depth[i] = dist
                proj.l[i] = lum
                proj.c[i] = col
                proj.owner[i] = shape_id

        # Угловой размер: куда объект попал БЫ. Сколько из этого реально
        # видно — считает _classify_occlusion после всей отрисовки.
        proj.spans[shape_id] = (lo, hi)

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
            intensity = 0.25 / (1.0 + dist / self.cfg.d_ref)
            depth[i] = dist
            proj.l[i] = (math.log(intensity + self.cfg.i_floor)
                         - math.log(self.cfg.i_floor))
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
        on_change = self.cfg.sustained_on_change
        for i in range(self.cfg.sustained_n):
            for c in range(self.cfg.color_channels):
                value = float(self.sustained[i, c])
                if on_change:
                    if abs(value - self._sustained_ref[i, c]) <= self.cfg.sustained_theta:
                        continue
                    self._sustained_ref[i, c] = value
                events.append(Event(tick, self.ch.sustained(i, c), value))
        return events

    # ------------------------------------------------------------- снапшот
    def state(self) -> dict:
        return {
            "l_ref": self.l_ref.copy(),
            "t_last": self.t_last.copy(),
            "sustained": self.sustained.copy(),
            "sustained_ref": self._sustained_ref.copy(),
            "dropped": self.dropped_events_total,
        }

    def restore(self, state: dict) -> None:
        self.l_ref = state["l_ref"].copy()
        self.t_last = state["t_last"].copy()
        self.sustained = state["sustained"].copy()
        self._sustained_ref = state["sustained_ref"].copy()
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
