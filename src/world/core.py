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
from typing import Any, Callable, Sequence

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
    """Наблюдательская сводка за такт. Никуда не возвращается в симуляцию.

    born/died — журнал событий. Он нужен тому, кто держит агентов снаружи
    (gym-обёртка): мир сообщает «тело 12 породило тело 47» и «тело 3 умерло»,
    а решение, что делать с их разумами, принимает вызывающий. Мир при этом
    не трогает ни одного объекта Mind.
    """

    tick: int
    population: int
    births: int
    deaths: int
    eaten: int
    food_on_grid: int
    mean_energy: float
    mean_age: float
    extinct: bool
    born: list[tuple[int, int]] = field(default_factory=list)  # (родитель, потомок)
    died: list[int] = field(default_factory=list)


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

        # Кэш locate(), включается только на время observe_all(). None = выключен.
        self._sense_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] | None = None

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

    def reset(
        self, mind_factory: Callable[[np.random.Generator], Mind] | None = None
    ) -> None:
        """Заселить мир. mind_factory создаёт независимые случайные геномы.

        Мир не выбирает, кого заселять: он вызывает фабрику N раз и всё.

        mind_factory=None — режим, в котором разумы держит вызывающий
        (см. gym_env.py). Тела создаются без mind, действия приходят снаружи,
        а о рождениях и смертях мир сообщает журналом в StepReport.
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
                mind=mind_factory(self.rng) if mind_factory is not None else None,
                born_at=0,
            )
            self._on_birth(body, body)
            self.bodies.append(body)
            self.occupancy[body.y, body.x] += 1

    def observe_all(self) -> tuple[list[int], list[np.ndarray]]:
        """Наблюдения всех живых тел по одному снимку мира.

        Снимок один на всех намеренно: если бы тела воспринимали мир
        вперемешку с движением, порядок в списке давал бы преимущество,
        то есть в симуляцию протёк бы отбор по позиции в массиве.

        На время съёма включается кэш locate(). Иначе каждое телогоняло бы
        np.nonzero по всему полю заново: при 400 телах на поле 200x200 это
        400 проходов по 40 тысячам клеток за один такт. Кэш безопасен именно
        здесь и только здесь — пока снимаются наблюдения, мир неподвижен по
        построению.
        """
        self._sense_cache = {}
        try:
            return [b.id for b in self.bodies], [self.observe(b) for b in self.bodies]
        finally:
            self._sense_cache = None

    def apply_actions(self, actions: Sequence[int]) -> StepReport:
        """Физика такта по готовым действиям. Ни одного обращения к Mind,
        кроме spawn() при делении — и то лишь если разумы держит сам мир.

        actions идут в том же порядке, что и observe_all().
        """
        if self.extinct:
            return self._report(0, 0, 0)
        if len(actions) != len(self.bodies):
            raise ValueError(
                f"нужно {len(self.bodies)} действий, получено {len(actions)}"
            )

        for a in actions:
            if not 0 <= int(a) < self.action_size:
                raise ValueError(
                    f"action {a} out of range, expected [0, {self.action_size})"
                )

        # применение в случайном порядке — по той же причине, что и снимок
        eaten = 0
        for i in self.rng.permutation(len(self.bodies)):
            body = self.bodies[i]
            eaten += self._apply(body, int(actions[i]))
            body.age += 1
            self._perturb(body)

        deaths, died = self._reap()
        births, born = self._reproduce()

        self._decay()
        self._regrow()

        self.tick += 1
        self._on_tick()

        if not self.bodies:
            self.extinct = True

        report = self._report(births, deaths, eaten)
        report.born = born
        report.died = died
        return report

    def step(self) -> StepReport:
        """Такт, в котором действия спрашиваются у разумов самого мира.

        Удобная обёртка над observe_all()/apply_actions() для случая, когда
        популяцию держит мир. Через gym работает второй путь.
        """
        if self.extinct:
            return self._report(0, 0, 0)

        _, observations = self.observe_all()
        actions = [int(b.mind.act(o)) for b, o in zip(self.bodies, observations)]
        return self.apply_actions(actions)

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

    def _reap(self) -> tuple[int, list[int]]:
        """Смерть. Единственный фильтр в этом мире.

        Заметь: ничего не сравнивается ни с чем. Нет «худших». Есть только
        «энергия кончилась» — локальный факт про одно тело.
        """
        survivors: list[Body] = []
        died: list[int] = []
        for body in self.bodies:
            too_old = self.cfg.max_age is not None and body.age >= self.cfg.max_age
            if body.energy <= 0.0 or too_old:
                self.occupancy[body.y, body.x] -= 1
                self._lifespans.append(body.age)
                died.append(body.id)
            else:
                survivors.append(body)
        self.bodies = survivors
        return len(died), died

    def _reproduce(self) -> tuple[int, list[tuple[int, int]]]:
        """Деление. Тоже локальное событие: тело смотрит только на свою энергию.

        max_population — ёмкость среды. Когда места нет, деление просто не
        происходит; никто не вытесняется и никто ни с кем не сравнивается.
        """
        born: list[tuple[int, int]] = []
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
                # Разум потомка делает сам родитель. Если разумы держит
                # вызывающий (gym), тело рождается пустым, а о событии
                # сообщается журналом — мир и тогда не трогает ни один Mind.
                mind=body.mind.spawn(self.rng) if body.mind is not None else None,
                born_at=self.tick,
                parent_id=body.id,
                generation=body.generation + 1,
            )
            self._on_birth(child, body)
            self.bodies.append(child)
            self.occupancy[cy, cx] += 1
            born.append((body.id, child.id))
        return len(born), born

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
        """Координаты объектов канала. Используется сенсорами.

        Кэшируется, только когда кэш явно включён в observe_all(); вне его
        считается заново, чтобы никакой прямой вызов observe() не получил
        устаревшие координаты.
        """
        cache = self._sense_cache
        # канал агентов зависит от того, кого исключаем, поэтому не кэшируем
        if cache is not None and channel == "food":
            key = ("food", food_type)
            hit = cache.get(key)
            if hit is None:
                hit = self._locate_uncached(channel, food_type, exclude)
                cache[key] = hit
            return hit
        return self._locate_uncached(channel, food_type, exclude)

    def _locate_uncached(
        self, channel: str, food_type: int | None, exclude: Body | None
    ) -> tuple[np.ndarray, np.ndarray]:
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
