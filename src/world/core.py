"""Ядро: тела, метаболизм, смерть, размножение.

Здесь вся физика мира. Ни одной строки, которая оценивает агента.

Порядок такта намеренно такой:
  1. все тела воспринимают мир ОДНОВРЕМЕННО (по снимку до любых движений)
  2. все тела выбирают действие
  3. действия применяются в случайном порядке
  4. умирают те, у кого кончилась энергия
  5. делятся те, у кого её избыток
  6. отрастает еда

Пункт 1 важен: если бы восприятие шло вперемешку с движением, порядок в
списке давал бы преимущество, то есть в мир протёк бы отбор по позиции
в массиве. Пункт 3 рандомизирован по той же причине.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Callable

import numpy as np

from .protocols import Mind
from .sensors import AntennaSensor, Interoception, Sensor

EMPTY = 0


@dataclass
class WorldConfig:
    """Физика мира. Всё, что здесь есть, — task-blind: по этим числам нельзя
    прочитать, какое поведение считается правильным."""

    size: int = 24
    boundary: str = "wrap"  # "wrap" (тор) | "clamp" (стены)
    allow_stay: bool = False  # добавляет 5-е действие «стоять»

    # метаболизм
    energy_at_birth: float = 40.0
    energy_max: float = 140.0
    cost_per_step: float = 1.0
    cost_per_move: float = 0.0

    # размножение
    repro_threshold: float = 90.0
    repro_cost: float = 0.0
    max_population: int = 150  # ёмкость среды, а не отбор: лишние просто не родятся

    # смерть
    max_age: int | None = None

    # Еда. max_food — не украшение, а главная ручка давления отбора.
    # Случайный ходок получает (плотность * food_energy) за шаг против
    # cost_per_step. При 25 и 1.0 безубыточность — плотность 0.04, то есть
    # 23 клетки из 576. Потолок держим заметно ниже, иначе при просевшей
    # популяции еда накапливается, блуждание становится прибыльным
    # и давление исчезает ровно тогда, когда оно нужнее всего.
    food_energy: float = 25.0
    regrow_per_step: float = 1.2
    max_food: int | None = 16  # потолок на КАЖДЫЙ тип еды, см. _food_cap
    initial_food_density: float = 0.028
    # Гниение: шанс, что несъеденная еда пропадёт за такт. По умолчанию
    # выключено. Включи, если агенты выродятся в стратегию «стоять и ждать,
    # пока еда сама отрастёт под ногами».
    food_decay: float = 0.0

    initial_population: int = 30
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.boundary not in ("wrap", "clamp"):
            raise ValueError("boundary must be 'wrap' or 'clamp'")
        if self.repro_threshold <= self.energy_at_birth:
            raise ValueError("repro_threshold must exceed energy_at_birth")
        if self.energy_max < self.repro_threshold:
            raise ValueError("energy_max must be >= repro_threshold")


@dataclass
class Body:
    """Тело — присутствие агента в мире. Веса живут в mind, мир их не трогает."""

    id: int
    x: int
    y: int
    energy: float
    mind: Mind
    age: int = 0
    born_at: int = 0
    parent_id: int | None = None
    generation: int = 0

    # Пер-теловые модификации, которые применяет ядро.
    # ShiftWorld пользуется ими, чтобы ломать сенсомоторику по ходу жизни.
    action_map: np.ndarray | None = None  # перестановка действий
    sensor_mask: np.ndarray | None = None  # мультипликативная маска наблюдения

    # Произвольные пер-теловые признаки для конкретных миров
    # (например, какой тип еды питателен именно для этого тела).
    traits: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepReport:
    """Наблюдательская сводка за такт. Никуда не возвращается в симуляцию."""

    tick: int
    population: int
    births: int
    deaths: int
    eaten: int
    food_on_grid: int
    mean_energy: float
    mean_age: float
    extinct: bool


class World(ABC):
    """Базовый мир. Конкретные миры переопределяют хуки, а не step()."""

    Config: type[WorldConfig] = WorldConfig

    def __init__(self, cfg: WorldConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else self.Config()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.food = np.zeros((self.cfg.size, self.cfg.size), dtype=np.int16)
        self.occupancy = np.zeros((self.cfg.size, self.cfg.size), dtype=np.int16)
        self.bodies: list[Body] = []

        self.tick = 0
        self.extinct = False
        self._ids = itertools.count()

        # Длительности завершённых жизней. Чисто наблюдательская величина:
        # мир её пишет, но никогда не читает.
        self._lifespans: list[int] = []

        self.sensor: Sensor = self._build_sensor()
        self.action_size = 5 if self.cfg.allow_stay else 4

    # ---------------------------------------------------------------- хуки

    def _build_sensor(self) -> Sensor:
        """Форма наблюдения. Переопредели, если миру нужны другие каналы."""
        return AntennaSensor("food", n_dirs=4) + Interoception(("energy",))

    def _initial_food(self) -> None:
        n = int(self.cfg.initial_food_density * self.cfg.size**2)
        self._scatter_food(n, food_type=1)

    def _regrow(self) -> None:
        """Отрастание еды за такт. Дробная скорость разыгрывается монеткой."""
        rate = self._regrow_rate()
        n = int(rate) + int(self.rng.random() < (rate - int(rate)))
        if n:
            self._scatter_food(n, food_type=1)

    def _regrow_rate(self) -> float:
        return self.cfg.regrow_per_step

    def _decay(self) -> None:
        """Гниение несъеденной еды. При food_decay=0 не делает ничего."""
        if self.cfg.food_decay <= 0.0:
            return
        occupied = self.food != EMPTY
        if not occupied.any():
            return
        rot = occupied & (self.rng.random(self.food.shape) < self.cfg.food_decay)
        self.food[rot] = EMPTY

    def _food_energy(self, body: Body, food_type: int) -> float:
        """Сколько энергии даёт еда этого типа этому телу.

        Принимает body, потому что в TwoFoodsWorld питательность зависит
        от тела: геном не может знать заранее, какой тип съедобен.
        """
        return self.cfg.food_energy

    def _on_birth(self, child: Body, parent: Body) -> None:
        """Вызывается после рождения. Здесь ставятся пер-теловые признаки."""

    def _on_tick(self) -> None:
        """Мировая динамика уровня такта (сезоны и т.п.)."""

    def _perturb(self, body: Body) -> None:
        """Пер-теловые изменения по ходу жизни (повреждения, инверсии)."""

    # -------------------------------------------------------------- запуск

    def reset(self, mind_factory: Callable[[np.random.Generator], Mind]) -> None:
        """Заселить мир. mind_factory создаёт независимые случайные геномы.

        Мир не выбирает, кого заселять: он вызывает фабрику N раз и всё.
        """
        self.rng = np.random.default_rng(self.cfg.seed)
        self.food[:] = 0
        self.occupancy[:] = 0
        self.bodies = []
        self.tick = 0
        self.extinct = False
        self._ids = itertools.count()
        self._lifespans = []

        self._initial_food()

        free = self._free_cells(self.cfg.initial_population)
        for (y, x) in free:
            body = Body(
                id=next(self._ids),
                x=int(x),
                y=int(y),
                energy=self.cfg.energy_at_birth,
                mind=mind_factory(self.rng),
                born_at=0,
            )
            self._on_birth(body, body)
            self.bodies.append(body)
            self.occupancy[body.y, body.x] += 1

    def step(self) -> StepReport:
        if self.extinct:
            return self._report(0, 0, 0)

        # 1-2. одновременное восприятие, затем решения
        actions: list[int] = []
        for body in self.bodies:
            obs = self.observe(body)
            a = int(body.mind.act(obs))
            if not 0 <= a < self.action_size:
                raise ValueError(
                    f"mind returned action {a}, expected [0, {self.action_size})"
                )
            actions.append(a)

        # 3. применение в случайном порядке
        eaten = 0
        for i in self.rng.permutation(len(self.bodies)):
            body = self.bodies[i]
            eaten += self._apply(body, actions[i])
            body.age += 1
            self._perturb(body)

        # 4. смерть
        deaths = self._reap()

        # 5. размножение
        births = self._reproduce()

        # 6. еда
        self._decay()
        self._regrow()

        self.tick += 1
        self._on_tick()

        if not self.bodies:
            self.extinct = True

        return self._report(births, deaths, eaten)

    # ------------------------------------------------------------ механика

    def _apply(self, body: Body, action: int) -> int:
        if body.action_map is not None:
            action = int(body.action_map[action])

        moved = action < 4
        if moved:
            dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[action]
            body.x, body.y = self._move(body.x, body.y, dx, dy)

        body.energy -= self.cfg.cost_per_step
        if moved:
            body.energy -= self.cfg.cost_per_move

        ate = 0
        ftype = int(self.food[body.y, body.x])
        if ftype != EMPTY:
            self.food[body.y, body.x] = EMPTY
            body.energy += self._food_energy(body, ftype)
            ate = 1

        body.energy = min(body.energy, self.cfg.energy_max)
        return ate

    def _move(self, x: int, y: int, dx: int, dy: int) -> tuple[int, int]:
        self.occupancy[y, x] -= 1
        if self.cfg.boundary == "wrap":
            x = (x + dx) % self.cfg.size
            y = (y + dy) % self.cfg.size
        else:
            x = min(max(x + dx, 0), self.cfg.size - 1)
            y = min(max(y + dy, 0), self.cfg.size - 1)
        self.occupancy[y, x] += 1
        return x, y

    def _reap(self) -> int:
        """Смерть. Единственный фильтр в этом мире.

        Заметь: ничего не сравнивается ни с чем. Нет «худших». Есть только
        «энергия кончилась» — локальный факт про одно тело.
        """
        survivors: list[Body] = []
        dead = 0
        for body in self.bodies:
            too_old = self.cfg.max_age is not None and body.age >= self.cfg.max_age
            if body.energy <= 0.0 or too_old:
                self.occupancy[body.y, body.x] -= 1
                self._lifespans.append(body.age)
                dead += 1
            else:
                survivors.append(body)
        self.bodies = survivors
        return dead

    def _reproduce(self) -> int:
        """Деление. Тоже локальное событие: тело смотрит только на свою энергию.

        max_population — ёмкость среды. Когда места нет, деление просто не
        происходит; никто не вытесняется и никто ни с кем не сравнивается.
        """
        births = 0
        for body in list(self.bodies):
            if len(self.bodies) >= self.cfg.max_population:
                break
            if body.energy < self.cfg.repro_threshold:
                continue

            body.energy -= self.cfg.repro_cost
            share = body.energy / 2.0
            body.energy = share

            cx, cy = self._move_from(body.x, body.y)
            child = Body(
                id=next(self._ids),
                x=cx,
                y=cy,
                energy=share,
                mind=body.mind.spawn(self.rng),
                born_at=self.tick,
                parent_id=body.id,
                generation=body.generation + 1,
            )
            self._on_birth(child, body)
            self.bodies.append(child)
            self.occupancy[cy, cx] += 1
            births += 1
        return births

    def _move_from(self, x: int, y: int) -> tuple[int, int]:
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[int(self.rng.integers(4))]
        if self.cfg.boundary == "wrap":
            return (x + dx) % self.cfg.size, (y + dy) % self.cfg.size
        return (
            min(max(x + dx, 0), self.cfg.size - 1),
            min(max(y + dy, 0), self.cfg.size - 1),
        )

    def _food_cap(self, food_type: int) -> int | None:
        """Потолок ДЛЯ ЭТОГО типа еды.

        Потолок обязан быть по типу, а не общим на всю еду. С общим потолком
        мир с несколькими типами заклинивает: тип, который никто не ест,
        накапливается, занимает всю квоту, и съедобный перестаёт отрастать
        насовсем. Получается, что чем лучше агент различает типы, тем вернее
        он себя уморит — давление отбора разворачивается задом наперёд.
        """
        return self.cfg.max_food

    def _scatter_food(self, n: int, food_type: int = 1) -> None:
        cap = self._food_cap(food_type)
        if cap is not None:
            room = cap - int(np.count_nonzero(self.food == food_type))
            n = min(n, max(0, room))
        if n <= 0:
            return
        empty = np.flatnonzero(self.food == EMPTY)
        if empty.size == 0:
            return
        pick = self.rng.choice(empty, size=min(n, empty.size), replace=False)
        ys, xs = np.unravel_index(pick, self.food.shape)
        self.food[ys, xs] = food_type

    def _free_cells(self, n: int) -> list[tuple[int, int]]:
        empty = np.flatnonzero((self.food == EMPTY) & (self.occupancy == 0))
        pick = self.rng.choice(empty, size=min(n, empty.size), replace=False)
        return [divmod(int(i), self.cfg.size) for i in pick]

    # ------------------------------------------------------- API сенсоров

    def observe(self, body: Body) -> np.ndarray:
        obs = self.sensor.sense(self, body)
        if body.sensor_mask is not None:
            obs = obs * body.sensor_mask
        return obs

    @property
    def observation_size(self) -> int:
        return self.sensor.size

    def locate(
        self, channel: str, food_type: int | None = None, exclude: Body | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Координаты объектов канала. Используется сенсорами."""
        if channel == "food":
            mask = self.food > 0 if food_type is None else self.food == food_type
            return np.nonzero(mask)
        if channel == "agents":
            ys = np.fromiter(
                (b.y for b in self.bodies if exclude is None or b.id != exclude.id),
                dtype=np.int64,
            )
            xs = np.fromiter(
                (b.x for b in self.bodies if exclude is None or b.id != exclude.id),
                dtype=np.int64,
            )
            return ys, xs
        raise ValueError(f"unknown channel: {channel}")

    def displacement(
        self, x: int, y: int, xs: np.ndarray, ys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Смещение до объектов. На торе — кратчайшее по кольцу."""
        dx = xs.astype(np.float64) - x
        dy = ys.astype(np.float64) - y
        if self.cfg.boundary == "wrap":
            s = self.cfg.size
            dx = (dx + s // 2) % s - s // 2
            dy = (dy + s // 2) % s - s // 2
        return dx, dy

    # ------------------------------------------------------------- прочее

    def _report(self, births: int, deaths: int, eaten: int) -> StepReport:
        n = len(self.bodies)
        return StepReport(
            tick=self.tick,
            population=n,
            births=births,
            deaths=deaths,
            eaten=eaten,
            food_on_grid=int(np.count_nonzero(self.food)),
            mean_energy=float(np.mean([b.energy for b in self.bodies])) if n else 0.0,
            mean_age=float(np.mean([b.age for b in self.bodies])) if n else 0.0,
            extinct=self.extinct,
        )

    def describe(self) -> str:
        params = ", ".join(
            f"{f.name}={getattr(self.cfg, f.name)!r}" for f in dataclass_fields(self.cfg)
        )
        return f"{type(self).__name__}(\n  obs={self.sensor!r} -> {self.observation_size}\n  actions={self.action_size}\n  {params}\n)"
