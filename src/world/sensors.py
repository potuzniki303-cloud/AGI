"""Сенсоры — то, как мир превращается в вектор для агента.

Сенсор отделён от мира намеренно: форма наблюдения влияет на обучаемость
сильнее, чем правило пластичности, и её надо уметь менять, не трогая физику.

Главный сенсор здесь — AntennaSensor. Он эгоцентрический и его выходы стоят
в том же порядке, что и действия: сенсор 0 отвечает за то же направление,
что действие 0. Это не косметика. Именно такое выравнивание делает связь
«вижу еду там → иду туда» одной ассоциацией, которую локальное правило
способно поймать. На аллоцентрической сетке таких ассоциаций 25x25 разных,
и ни одна не обобщается.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Направления в порядке действий: 0-север, 1-юг, 2-запад, 3-восток.
# y растёт вниз (как индекс строки в numpy), поэтому север это -1 по y.
CARDINAL = np.array(
    [
        [0.0, -1.0],  # north
        [0.0, 1.0],  # south
        [-1.0, 0.0],  # west
        [1.0, 0.0],  # east
    ],
    dtype=np.float64,
)

DIAGONAL = np.array(
    [
        [1.0, -1.0],  # north-east
        [1.0, 1.0],  # south-east
        [-1.0, 1.0],  # south-west
        [-1.0, -1.0],  # north-west
    ],
    dtype=np.float64,
)


def _directions(n_dirs: int) -> np.ndarray:
    """Единичные векторы направлений. 4 — только стороны света, 8 — с диагоналями."""
    if n_dirs == 4:
        dirs = CARDINAL
    elif n_dirs == 8:
        dirs = np.concatenate([CARDINAL, DIAGONAL])
    else:
        raise ValueError(f"n_dirs must be 4 or 8, got {n_dirs}")
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


class Sensor(ABC):
    """Базовый сенсор. size должен быть известен до первого вызова —
    агенту нужно знать размер входа, чтобы построить сеть."""

    @property
    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def sense(self, world, body) -> np.ndarray: ...

    def __add__(self, other: "Sensor") -> "ConcatSensor":
        return ConcatSensor([self, other])


class ConcatSensor(Sensor):
    """Склейка нескольких сенсоров в один вектор."""

    def __init__(self, parts: list[Sensor]) -> None:
        flat: list[Sensor] = []
        for p in parts:
            flat.extend(p.parts if isinstance(p, ConcatSensor) else [p])
        self.parts = flat

    @property
    def size(self) -> int:
        return sum(p.size for p in self.parts)

    def sense(self, world, body) -> np.ndarray:
        return np.concatenate([p.sense(world, body) for p in self.parts])

    def __repr__(self) -> str:
        return f"ConcatSensor({self.parts!r})"


class AntennaSensor(Sensor):
    """Направленная близость до ближайшего объекта в каждом секторе.

    Для каждого направления d выход это max по объектам от
    proximity * alignment, где proximity = 1/(1+L1-расстояние), а alignment —
    косинус между смещением до объекта и направлением d (отрицательные
    обрезаются нулём).

    Ключевое свойство: движение в сторону объекта увеличивает сенсор этого
    направления. Из-за этого правильная пара «сенсор d — действие d»
    коррелирована сильнее неправильной, и разницу создаёт геометрия мира,
    а не оценка дизайнера.
    """

    def __init__(
        self,
        channel: str = "food",
        n_dirs: int = 4,
        food_type: int | None = None,
        max_range: float | None = None,
    ) -> None:
        if channel not in ("food", "agents"):
            raise ValueError(f"unknown channel: {channel}")
        self.channel = channel
        self.food_type = food_type
        # По умолчанию дальность не ограничена: тело всегда видит ближайшую
        # еду, где бы она ни была. Так мир заведомо проходим, и первый
        # эксперимент проверяет правило обучения, а не разведку.
        # Ограничив дальность, получишь задачу на поиск вслепую — она
        # намного тяжелее, и браться за неё стоит уже с работающим таксисом.
        self.max_range = max_range
        self.dirs = _directions(n_dirs)
        self._size = n_dirs

    @property
    def size(self) -> int:
        return self._size

    def sense(self, world, body) -> np.ndarray:
        ys, xs = world.locate(self.channel, food_type=self.food_type, exclude=body)
        out = np.zeros(self._size, dtype=np.float32)
        if ys.size == 0:
            return out

        dx, dy = world.displacement(body.x, body.y, xs, ys)

        l1 = np.abs(dx) + np.abs(dy)
        keep = l1 > 0
        if self.max_range is not None:
            keep &= l1 <= self.max_range
        if not keep.any():
            return out
        dx, dy, l1 = dx[keep], dy[keep], l1[keep]

        prox = 1.0 / (1.0 + l1)
        norm = np.sqrt(dx * dx + dy * dy)
        ux, uy = dx / norm, dy / norm

        # (n_dirs, n_objects): выравнивание каждого объекта с каждым направлением
        align = self.dirs[:, 0:1] * ux[None, :] + self.dirs[:, 1:2] * uy[None, :]
        np.clip(align, 0.0, None, out=align)

        return (align * prox[None, :]).max(axis=1).astype(np.float32)

    def __repr__(self) -> str:
        suffix = "" if self.food_type is None else f", type={self.food_type}"
        return f"AntennaSensor({self.channel}, {self._size}{suffix})"


class PatchSensor(Sensor):
    """Эгоцентрическое окно (2r+1)x(2r+1) вокруг тела.

    Даёт больше информации, чем антенна, но и вход намного шире, а
    локальному правилу тяжелее найти в нём инвариант. Полезен, когда
    нужна пространственная структура (см. PatchWorld).

    Канал "wall" имеет смысл только при boundary="clamp": на торе стен нет.
    """

    def __init__(
        self,
        radius: int = 2,
        channels: tuple[str, ...] = ("food", "agents"),
        food_type: int | None = None,
    ) -> None:
        if radius < 1:
            raise ValueError("radius must be >= 1")
        for c in channels:
            if c not in ("food", "agents", "wall"):
                raise ValueError(f"unknown channel: {c}")
        self.radius = radius
        self.channels = channels
        self.food_type = food_type
        self._side = 2 * radius + 1
        self._size = self._side * self._side * len(channels)

    @property
    def size(self) -> int:
        return self._size

    def sense(self, world, body) -> np.ndarray:
        r = self.radius
        offs = np.arange(-r, r + 1)
        rows = body.y + offs
        cols = body.x + offs

        if world.cfg.boundary == "wrap":
            rows = rows % world.cfg.size
            cols = cols % world.cfg.size
            oob = None
        else:
            oob_r = (rows < 0) | (rows >= world.cfg.size)
            oob_c = (cols < 0) | (cols >= world.cfg.size)
            oob = oob_r[:, None] | oob_c[None, :]
            rows = np.clip(rows, 0, world.cfg.size - 1)
            cols = np.clip(cols, 0, world.cfg.size - 1)

        idx = np.ix_(rows, cols)
        planes = []
        for c in self.channels:
            if c == "food":
                grid = world.food
                plane = (grid > 0) if self.food_type is None else (grid == self.food_type)
                plane = plane[idx].astype(np.float32)
            elif c == "agents":
                plane = world.occupancy[idx].astype(np.float32)
                # своё тело в центре не показываем: это шум, а не информация
                plane[r, r] = max(0.0, plane[r, r] - 1.0)
            else:  # wall
                plane = np.zeros((self._side, self._side), dtype=np.float32)
                if oob is not None:
                    plane[oob] = 1.0
            if oob is not None and c != "wall":
                plane = plane * (~oob)
            planes.append(plane.ravel())

        return np.concatenate(planes).astype(np.float32)

    def __repr__(self) -> str:
        return f"PatchSensor(r={self.radius}, {self.channels})"


class Interoception(Sensor):
    """Внутреннее состояние тела. Голод — это тоже сенсор, просто направленный внутрь.

    Заметь: energy здесь нормирована и подаётся как обычный вход. Она НЕ
    умножает ничего в правиле обучения. Как только внутренний скаляр начинает
    масштабировать dw, он превращается в награду.
    """

    def __init__(self, fields: tuple[str, ...] = ("energy",)) -> None:
        for f in fields:
            if f not in ("energy", "age"):
                raise ValueError(f"unknown field: {f}")
        self.fields = fields

    @property
    def size(self) -> int:
        return len(self.fields)

    def sense(self, world, body) -> np.ndarray:
        out = np.empty(len(self.fields), dtype=np.float32)
        for i, f in enumerate(self.fields):
            if f == "energy":
                out[i] = body.energy / world.cfg.energy_max
            else:
                ref = world.cfg.max_age or 1000
                out[i] = min(body.age / ref, 1.0)
        return out

    def __repr__(self) -> str:
        return f"Interoception({self.fields})"
