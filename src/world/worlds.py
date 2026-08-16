"""Конкретные миры.

Все пять отличаются только физикой ресурса и тела. Ни в одном нет функции
оценки агента. Порядок в файле — порядок экспериментов, который я бы прошёл:

  forage     — базовый. Выживает ли вообще хоть кто-то.
  seasons    — ресурс нестационарен во времени.
  patches    — ресурс структурирован в пространстве.
  shift      — сенсомоторика ломается по ходу жизни.
  two_foods  — что съедобно, геном знать не может в принципе.

Последние два — главные. Они устроены так, что замороженные веса их
физически не решают: ответ либо меняется внутри жизни (shift), либо
разыгрывается заново при каждом рождении (two_foods). Если агент их
проходит, значит он несёт способ найти решение, а не само решение.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .core import EMPTY, Body, World, WorldConfig
from .sensors import AntennaSensor, Interoception, PatchSensor, Sensor


# --------------------------------------------------------------- forage

class ForageWorld(World):
    """Базовый мир: несколько агентов, еда, энергия, смерть, деление.

    Физика целиком:
      - каждый шаг стоит энергии
      - шаг на клетку с едой её съедает и добавляет энергии
      - энергия <= 0 — тела больше нет
      - энергия >= порога — тело делится пополам, потомок с мутацией
      - еда отрастает в случайных пустых клетках с постоянной скоростью

    Больше ничего. Ни счёта, ни цели, ни оценки.
    """


# -------------------------------------------------------------- seasons

@dataclass
class SeasonsConfig(WorldConfig):
    season_period: int = 400  # тактов на полный цикл
    season_amplitude: float = 0.9  # 0 — нет сезонов, 1 — зимой еда не растёт


class SeasonsWorld(World):
    """Скорость отрастания еды колеблется синусоидой.

    Зачем: замороженные веса настраиваются под одну плотность ресурса.
    Здесь плотность меняется медленнее жизни, но быстрее эволюции, так что
    выгодно уметь перестраиваться внутри жизни. Это первая проверка, что
    пластичность вообще даёт преимущество над её отсутствием.

    Период задаётся в тактах и должен быть сопоставим с длиной жизни:
    если он сильно короче, сезон усредняется и превращается в шум;
    если сильно длиннее, каждое поколение живёт в стационарном мире.
    """

    Config = SeasonsConfig
    cfg: SeasonsConfig

    def _regrow_rate(self) -> float:
        phase = 2.0 * math.pi * self.tick / self.cfg.season_period
        factor = 1.0 + self.cfg.season_amplitude * math.sin(phase)
        return max(0.0, self.cfg.regrow_per_step * factor)

    @property
    def season(self) -> float:
        """Фаза сезона в [-1, 1]. Для графиков, в симуляцию не возвращается."""
        return math.sin(2.0 * math.pi * self.tick / self.cfg.season_period)


# -------------------------------------------------------------- patches

@dataclass
class PatchesConfig(WorldConfig):
    patch_radius: int = 2  # разброс новой еды вокруг существующей
    seed_patches: int = 4  # сколько независимых пятен в начале
    stray_probability: float = 0.05  # доля еды, вырастающей где попало
    max_food: int | None = 18
    initial_food_density: float = 0.031


class PatchesWorld(World):
    """Еда растёт кучно: новая появляется рядом с уже существующей.

    Зачем: при равномерной еде оптимальна чистая реактивная стратегия
    «иди на самый сильный сенсор». При кучной еде выгоднее другое поведение —
    после находки задержаться и обыскать окрестность. Это уже требует, чтобы
    поведение зависело от недавней истории, а не только от текущего кадра.

    Здесь стоит попробовать PatchSensor вместо антенны: пятно — это
    пространственная структура, и в окне она видна, а в четырёх скалярах нет.
    """

    Config = PatchesConfig
    cfg: PatchesConfig

    def _build_sensor(self) -> Sensor:
        return AntennaSensor("food", n_dirs=8) + Interoception(("energy",))

    def _initial_food(self) -> None:
        total = int(self.cfg.initial_food_density * self.cfg.size**2)
        if total <= 0:
            return
        centres = self._free_cells(self.cfg.seed_patches)
        per = max(1, total // max(1, len(centres)))
        for (cy, cx) in centres:
            self.food[cy, cx] = 1
            for _ in range(per - 1):
                self._grow_near(cy, cx)

    def _regrow(self) -> None:
        rate = self._regrow_rate()
        n = int(rate) + int(self.rng.random() < (rate - int(rate)))
        for _ in range(n):
            if self.cfg.max_food is not None:
                if int(np.count_nonzero(self.food)) >= self.cfg.max_food:
                    return
            ys, xs = np.nonzero(self.food > 0)
            if ys.size == 0 or self.rng.random() < self.cfg.stray_probability:
                self._scatter_food(1, food_type=1)
                continue
            i = int(self.rng.integers(ys.size))
            self._grow_near(int(ys[i]), int(xs[i]))

    def _grow_near(self, cy: int, cx: int) -> None:
        r = self.cfg.patch_radius
        for _ in range(8):  # несколько попыток найти пустую клетку рядом
            dy = int(self.rng.integers(-r, r + 1))
            dx = int(self.rng.integers(-r, r + 1))
            if self.cfg.boundary == "wrap":
                y, x = (cy + dy) % self.cfg.size, (cx + dx) % self.cfg.size
            else:
                y = min(max(cy + dy, 0), self.cfg.size - 1)
                x = min(max(cx + dx, 0), self.cfg.size - 1)
            if self.food[y, x] == EMPTY:
                self.food[y, x] = 1
                return


# ---------------------------------------------------------------- shift

@dataclass
class ShiftConfig(WorldConfig):
    shift_at_age: int = 60  # возраст, в котором ломается сенсомоторика
    shift_jitter: int = 20  # разброс возраста по телам
    shift_mode: str = "motor"  # "motor" | "sensor" | "both"
    # Потолка возраста здесь нет намеренно: в этом мире читать надо
    # продолжительность жизни, а max_age обрезал бы ровно тот хвост
    # распределения, который показывает, кто пережил поломку.
    max_age: int | None = None


class ShiftWorld(World):
    """Сенсомоторное отображение тела ломается посреди жизни.

    "motor" — действия переставляются (пошёл налево, поехал направо).
    "sensor" — часть сенсорных каналов глохнет насовсем.

    Агенту об этом не сообщается никак: нет ни флага в наблюдении, ни сигнала.
    Он может обнаружить поломку только по тому, что мир перестал отвечать
    как раньше.

    Зачем: это самый прямой тест на то, что агент несёт правило, а не ответ.
    Замороженные веса после перестановки действий продолжают уверенно идти
    не туда и умирают. Живое локальное правило может перестроиться, потому
    что корреляции в замкнутой петле поменялись вместе с миром.

    Ровно этот эксперимент — «восстановление после повреждения» — у
    Najarro & Risi (NeurIPS 2020) главный аргумент в пользу эволюции
    пластичности вместо эволюции весов.
    """

    Config = ShiftConfig
    cfg: ShiftConfig

    def _on_birth(self, child: Body, parent: Body) -> None:
        jitter = int(self.rng.integers(-self.cfg.shift_jitter, self.cfg.shift_jitter + 1))
        child.traits["shift_at"] = max(1, self.cfg.shift_at_age + jitter)
        child.traits["shifted"] = False

    def _perturb(self, body: Body) -> None:
        if body.traits.get("shifted") or body.age < body.traits.get("shift_at", 1 << 30):
            return
        body.traits["shifted"] = True

        if self.cfg.shift_mode in ("motor", "both"):
            # Тождественная перестановка ничего не ломает, а выпадает она
            # в 1 случае из 24. Пересдаём, чтобы поломка была всегда поломкой.
            perm = self.rng.permutation(self.action_size)
            while np.array_equal(perm, np.arange(self.action_size)):
                perm = self.rng.permutation(self.action_size)
            body.action_map = perm
        if self.cfg.shift_mode in ("sensor", "both"):
            mask = np.ones(self.observation_size, dtype=np.float32)
            n_dead = max(1, self.observation_size // 4)
            dead = self.rng.choice(self.observation_size, size=n_dead, replace=False)
            mask[dead] = 0.0
            body.sensor_mask = mask


# ------------------------------------------------------------ two_foods

@dataclass
class TwoFoodsConfig(WorldConfig):
    # toxin_energy = -food_energy подобрано намеренно: тело, которое ест
    # без разбора, в среднем получает от еды ровно ноль и умирает от одной
    # только стоимости шагов. Различающее тело получает +food_energy.
    # Весь разрыв между «живёт» и «нет» лежит в различении типов.
    toxin_energy: float = -25.0
    polarity_scope: str = "body"  # "body" | "world"
    polarity_period: int = 500  # только для scope="world"
    food_ratio: float = 0.5  # доля первого типа при отрастании

    # Одна ошибка должна быть больно, но переживаемо: иначе первая же
    # проба убивает, и учиться внутри жизни физически негде.
    energy_at_birth: float = 60.0
    repro_threshold: float = 120.0
    energy_max: float = 180.0

    # Любому телу полезна лишь половина еды, поэтому ресурса вдвое больше.
    max_food: int | None = 30
    regrow_per_step: float = 1.6
    initial_food_density: float = 0.05


class TwoFoodsWorld(World):
    """Два внешне различимых типа еды. Один питателен, другой ядовит.

    И вот главное: какой из них какой, разыгрывается заново при каждом
    рождении (polarity_scope="body").

    Из этого следует ровно то, ради чего весь проект. Геном не может нести
    ответ — ответ не наследуется, он бросается монеткой на старте жизни.
    Наследоваться может только способ его выяснить: попробовать, и по
    последствиям перестроить связь между «вижу тип A» и «иду туда».

    Наблюдение здесь имеет две отдельные пищевые антенны — типы различимы
    на вид. Что за ними стоит, на вид не определить.

    polarity_scope="world" — мягкий вариант: полярность общая для всех и
    переключается раз в polarity_period тактов. С него проще начать.
    """

    Config = TwoFoodsConfig
    cfg: TwoFoodsConfig

    def __init__(self, cfg: TwoFoodsConfig | None = None) -> None:
        super().__init__(cfg)
        self.world_polarity = 1

    def _build_sensor(self) -> Sensor:
        return (
            AntennaSensor("food", n_dirs=4, food_type=1)
            + AntennaSensor("food", n_dirs=4, food_type=2)
            + Interoception(("energy",))
        )

    def _initial_food(self) -> None:
        n = int(self.cfg.initial_food_density * self.cfg.size**2)
        n1 = int(n * self.cfg.food_ratio)
        self._scatter_food(n1, food_type=1)
        self._scatter_food(n - n1, food_type=2)

    def _regrow(self) -> None:
        rate = self._regrow_rate()
        n = int(rate) + int(self.rng.random() < (rate - int(rate)))
        for _ in range(n):
            ftype = 1 if self.rng.random() < self.cfg.food_ratio else 2
            self._scatter_food(1, food_type=ftype)

    def _food_cap(self, food_type: int) -> int | None:
        # Каждому типу своя квота: иначе несъедобный тип забивает поле
        # и съедобный перестаёт отрастать.
        return None if self.cfg.max_food is None else self.cfg.max_food // 2

    def _on_birth(self, child: Body, parent: Body) -> None:
        if self.cfg.polarity_scope == "body":
            # Монетка на каждое тело. Родитель не передаёт ответ, даже если бы хотел.
            child.traits["nutritious"] = 1 if self.rng.random() < 0.5 else 2

    def _on_tick(self) -> None:
        if self.cfg.polarity_scope == "world" and self.cfg.polarity_period > 0:
            if self.tick % self.cfg.polarity_period == 0:
                self.world_polarity = 3 - self.world_polarity

    def _food_energy(self, body: Body, food_type: int) -> float:
        if self.cfg.polarity_scope == "body":
            good = body.traits.get("nutritious", 1)
        else:
            good = self.world_polarity
        return self.cfg.food_energy if food_type == good else self.cfg.toxin_energy


# --------------------------------------------------------------- bounty

@dataclass
class BountyConfig(WorldConfig):
    bounty_factor: float = 10.0  # во сколько раз больше еды в начале
    bounty_ticks: int = 3000  # за сколько тактов спадает до нормы
    # Стартовать надо сразу изобильно, иначе спад начнётся раньше,
    # чем поле успеет наполниться.
    initial_food_density: float = 0.28


class BountyWorld(World):
    """Сначала еды в bounty_factor раз больше нормы, потом изобилие уходит.

    Множитель падает линейно от bounty_factor до 1 за bounty_ticks тактов и
    дальше остаётся единицей — то есть мир плавно превращается в forage.

    Зачем. Это не новая проверка, а РАЗГОН в forage. Холодный старт — самая
    частая причина, по которой ничего не выходит: вся начальная популяция
    вымирает раньше, чем эволюция успевает нащупать хоть что-то, и ты видишь
    «не работает» там, где на самом деле «не успело начаться». Здесь у первых
    поколений есть фора: в изобильном мире выживает почти кто угодно, зато
    поколения успевают смениться и накопить разнообразие геномов. К моменту,
    когда еды становится мало, отбору уже есть из чего выбирать.

    Важно понимать цену: пока изобилие не спало, ДАВЛЕНИЯ НЕТ. Случайное
    блуждание в начале прибыльно (плотность 0.28 против точки безубыточности
    0.04), и первые сотни тактов ничего не отбирают. Смотреть на результаты
    имеет смысл начиная примерно с bounty_ticks, а не раньше.

    Множитель применяется и к потолку еды, и к скорости отрастания: иначе
    поле не удержало бы изобилие под выеданием.
    """

    Config = BountyConfig
    cfg: BountyConfig

    def _bounty(self) -> float:
        """Множитель изобилия: bounty_factor -> 1.0 линейно."""
        if self.cfg.bounty_ticks <= 0:
            return 1.0
        left = max(0.0, 1.0 - self.tick / self.cfg.bounty_ticks)
        return 1.0 + (self.cfg.bounty_factor - 1.0) * left

    def _food_cap(self, food_type: int) -> int | None:
        if self.cfg.max_food is None:
            return None
        return int(round(self.cfg.max_food * self._bounty()))

    def _regrow_rate(self) -> float:
        return self.cfg.regrow_per_step * self._bounty()

    def _on_tick(self) -> None:
        """Убрать излишек, когда потолок опустился ниже текущего запаса.

        Без этого спад почти не чувствовался бы: потолок ограничивает только
        прирост, а уже выросшая еда лежала бы на поле, пока её не съедят.
        """
        cap = self._food_cap(1)
        if cap is None:
            return
        excess = int(np.count_nonzero(self.food)) - cap
        if excess <= 0:
            return
        ys, xs = np.nonzero(self.food)
        pick = self.rng.choice(len(ys), size=excess, replace=False)
        self.food[ys[pick], xs[pick]] = EMPTY


# -------------------------------------------------------------- growing

@dataclass
class GrowingConfig(WorldConfig):
    start_size: int = 8  # сторона в начале
    growth_every: int = 200  # каждые N тактов сторона +1
    # size из базового конфига — это КОНЕЧНЫЙ размер.
    # Стартовая популяция меньше базовой: 30 тел на 8x8 это 47% занятости
    # поля, и маленький мир превращался бы в давку вместо разгона.
    initial_population: int = 12


class GrowingWorld(World):
    """Мир начинается маленьким и растёт до size.

    Каждые growth_every тактов сторона увеличивается на единицу, пока не
    достигнет cfg.size. Новое пространство появляется по югу и востоку,
    тела остаются на своих координатах.

    Зачем. Разгон тут устроен геометрией, и меняется НЕ то, что кажется.
    Замеры при start_size=8, size=24, growth_every=100:

      сторона   занятость поля телами   средний путь до ближайшей еды
         8               0.47                        2.5
        14               0.15                        5.8
        24               0.05                        6.9

    Плотность еды при этом почти не меняется, и популяция тоже — обе
    держатся своими механизмами: еду подчищает выедание, а численность
    задаёт скорость отрастания (regrow * food_energy / cost), которая от
    размера поля не зависит вовсе. Так что «маленький мир = много еды»
    здесь не работает: еды мало всегда.

    Меняется другое — ДЛИНА ПУТИ до еды, почти втрое. И вот это настоящая
    проверка: правило, настроившееся на близкие градиенты, должно
    продолжать работать, когда идти стало вдвое-втрое дальше. Антенна даёт
    proximity = 1/(1 + L1), то есть на расстоянии 2 сигнал равен 0.33, а на
    расстоянии 7 — уже 0.125. Сенсорный вход слабеет втрое, и веса,
    подогнанные под сильный сигнал, окажутся не в том масштабе. Устойчивости
    к смене масштаба входа не проверяет больше ни один мир.

    Побочно меняется скученность: 47% занятости в начале против 5% в конце.
    Стартовая популяция поэтому уменьшена до 12 — иначе маленький мир был бы
    не разгоном, а давкой.

    Читать надо продолжительность жизни. Популяция останется около 30 при
    любом размере поля, потому что её потолок задаёт еда, а не геометрия.

    Учти: рост меняет топологию тора. Тела, которые были рядом через край,
    после расширения окажутся далеко друг от друга. Это разрыв, но честный —
    мир буквально стал больше.
    """

    Config = GrowingConfig
    cfg: GrowingConfig

    def __init__(self, cfg: GrowingConfig | None = None) -> None:
        cfg = cfg if cfg is not None else self.Config()
        if cfg.start_size > cfg.size:
            raise ValueError("start_size must not exceed size")
        if cfg.start_size <= 0:
            raise ValueError("start_size must be > 0")
        # size в конфиге — конечная цель; живём мы стартовой стороной
        self.final_size = cfg.size
        cfg.size = cfg.start_size
        super().__init__(cfg)

    def reset(self, mind_factory=None) -> None:
        # вернуть стартовый размер: прошлый прогон мог оставить мир выросшим
        self._resize(self.cfg.start_size)
        super().reset(mind_factory)

    def _on_tick(self) -> None:
        if self.cfg.size >= self.final_size:
            return
        if self.cfg.growth_every > 0 and self.tick % self.cfg.growth_every == 0:
            self._resize(self.cfg.size + 1)

    def _resize(self, new_size: int) -> None:
        old = self.cfg.size
        if new_size == old:
            return
        keep = min(old, new_size)
        food = np.zeros((new_size, new_size), dtype=self.food.dtype)
        occupancy = np.zeros((new_size, new_size), dtype=self.occupancy.dtype)
        food[:keep, :keep] = self.food[:keep, :keep]
        occupancy[:keep, :keep] = self.occupancy[:keep, :keep]
        self.food = food
        self.occupancy = occupancy
        self.cfg.size = new_size

    @property
    def growth_left(self) -> int:
        """Сколько ещё расти. Для графиков, в симуляцию не возвращается."""
        return self.final_size - self.cfg.size
