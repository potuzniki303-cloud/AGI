"""Все константы Фазы 0 одним объектом.

Каждая константа помечена уровнем обоснования — ровно как в спецификации:

  ВЫВЕДЕНО   — следует из других решений арифметически, менять только с ними.
  КАЛИБРОВКА — подобрано так, чтобы задача была решаемой, но не тривиальной.
  ПРОИЗВОЛ   — не обосновано ничем, кроме здравого смысла. Ожидается замена.

Пометки не комментарий, а данные: `Config.justification("BASAL")` вернёт
уровень, а `meta.json` пишет их вместе со значениями. Смысл в том, что
если вывод эксперимента зависит от константы с пометкой ПРОИЗВОЛ, это видно
из артефакта, а не из памяти автора.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

DERIVED = "выведено"
CALIB = "калибровка"
ARBITRARY = "произвол"


class Level(IntEnum):
    """Уровень перцептивной лестницы (4.2)."""

    P0 = 0
    P1 = 1
    P2 = 2


class Mode(IntEnum):
    """Режим мира (Части 7-9)."""

    A = 0  # стационарный, калибровочный
    B = 1  # реверсия
    C = 2  # ритм


class Kind(IntEnum):
    """Вид предмета. Внешность, НЕ питательность."""

    A = 0
    B = 1


# Пометки уровня обоснования. Ключ — имя поля Config.
JUSTIFICATION: dict[str, str] = {
    "tick_hz": ARBITRARY,
    "arena": CALIB,
    "walls": ARBITRARY,
    "e_wall_hit": ARBITRARY,
    "r_body": DERIVED,
    "mass": DERIVED,
    "f_max": CALIB,
    "t_max_over_i": CALIB,
    "drag": CALIB,
    "drag_ang": CALIB,
    "n_items": CALIB,
    "r_item": DERIVED,
    "t_respawn": ARBITRARY,
    "respawn_min_dist": ARBITRARY,
    "item_speed_max": CALIB,
    "e_max": DERIVED,
    "e_init": ARBITRARY,
    "basal": CALIB,
    "move_cost": CALIB,
    "turn_cost_ratio": ARBITRARY,
    "e_food": CALIB,
    "e_poison": CALIB,
    "metabolic_compute": "флаг",
    "k_compute": ARBITRARY,
    "node_count_ref": ARBITRARY,
    "percept_level": "флаг",
    "fov_deg": ARBITRARY,
    "retina_n": CALIB,
    "color_channels": ARBITRARY,
    "theta_event": CALIB,
    "refractory_ticks": ARBITRARY,
    "event_rate_cap": ARBITRARY,
    "sustained_n": ARBITRARY,
    "tau_sustained": ARBITRARY,
    "walls_visible": ARBITRARY,
    "fovea_deg": ARBITRARY,
    "periphery_downsample": ARBITRARY,
    "saccade_max_deg": ARBITRARY,
    "saccade_speed": ARBITRARY,
    "e_saccade": ARBITRARY,
    "tau_proprio": ARBITRARY,
    "tau_energy": ARBITRARY,
    "noci_duration": ARBITRARY,
    "motor_decay": "флаг",
    "tau_motor": ARBITRARY,
    "motor_discrete": "флаг",
    "mode": "флаг",
    "difficulty": "флаг",
    "flip_interval": CALIB,
    "rhythm_period": CALIB,
    "rhythm_window": CALIB,
    "clock": "флаг",
    "budget_profile": "флаг",
    "seeds": "флаг",
    "snapshot_every": ARBITRARY,
    "sensor_noise_sigma": ARBITRARY,
}


def _default_seeds() -> dict[str, int]:
    return {
        "world_layout": 0,
        "item_respawn": 0,
        "regime_flips": 0,
        "sensor_noise": 0,
    }


@dataclass(frozen=True)
class Config:
    """Физика мира. Заморожена: менять константу — значит создать новый Config,
    и он попадёт в meta.json целиком. Тихая подкрутка между экспериментами
    невозможна по построению."""

    # --- время ---
    tick_hz: int = 60

    # --- арена ---
    arena: tuple[float, float] = (64.0, 64.0)
    walls: bool = True
    e_wall_hit: float = 0.005

    # --- тело ---
    r_body: float = 1.0
    mass: float = 1.0
    f_max: float = 20.0
    t_max_over_i: float = 8.0
    drag: float = 2.0
    drag_ang: float = 4.0

    # --- предметы ---
    n_items: int = 12
    r_item: float = 1.0
    t_respawn: int = 120
    respawn_min_dist: float = 8.0
    item_speed_max: float = 3.0

    # --- гомеостат ---
    e_max: float = 1.0
    e_init: float = 0.6
    basal: float = 0.02
    move_cost: float = 0.03
    turn_cost_ratio: float = 0.5
    e_food: float = 0.25
    e_poison: float = -0.15
    metabolic_compute: bool = False
    k_compute: float = 0.01
    node_count_ref: int = 1000

    # --- зрение ---
    percept_level: Level = Level.P0
    fov_deg: float = 120.0
    retina_n: int = 32
    color_channels: int = 2
    theta_event: float = 0.15
    refractory_ticks: int = 2
    event_rate_cap: float = 0.20
    sustained_n: int = 8
    tau_sustained: float = 0.100
    # Спецификация не говорит, видны ли стены. Оставлено флагом и выключено:
    # включение добавляет ориентиры, которых в спецификации нет (Часть 12).
    walls_visible: bool = False

    # --- фовеа (P2) ---
    fovea_deg: float = 30.0
    periphery_downsample: int = 4
    saccade_max_deg: float = 45.0
    saccade_speed: float = 300.0
    e_saccade: float = 0.002

    # --- прочие сенсоры ---
    tau_proprio: float = 0.050
    tau_energy: float = 0.500
    noci_duration: int = 30
    sensor_noise_sigma: float = 0.0

    # --- моторика ---
    motor_decay: bool = False
    tau_motor: float = 0.5
    motor_discrete: bool = False

    # --- режимы ---
    mode: Mode = Mode.A
    difficulty: str = "A1"
    flip_interval: tuple[int, int] = (1500, 4500)
    rhythm_period: int = 180
    rhythm_window: float = 0.3

    # --- приборы ---
    clock: str = "fast"
    budget_profile: str | None = None
    seeds: dict[str, int] = field(default_factory=_default_seeds)
    snapshot_every: int = 100_000

    # ---------------------------------------------------------------- выведено
    @property
    def dt(self) -> float:
        """ВЫВЕДЕНО из tick_hz."""
        return 1.0 / self.tick_hz

    @property
    def v_max(self) -> float:
        """ВЫВЕДЕНО: установившаяся скорость при полной тяге, F_max/drag."""
        return self.f_max / self.drag

    @property
    def crossing_time(self) -> float:
        """ВЫВЕДЕНО: за сколько секунд тело пересекает арену на v_max."""
        return self.arena[0] / self.v_max

    @property
    def starve_time_idle(self) -> float:
        """ВЫВЕДЕНО: голодная смерть из полного бака в покое, секунд."""
        return self.e_max / self.basal

    @property
    def starve_time_moving(self) -> float:
        """ВЫВЕДЕНО: то же на полной тяге."""
        return self.e_max / (self.basal + self.move_cost)

    @property
    def required_feed_rate(self) -> float:
        """ВЫВЕДЕНО: секунд на предмет, чтобы держаться в нуле в покое."""
        return self.e_food / self.basal

    @property
    def item_density(self) -> float:
        """ВЫВЕДЕНО: предметов на у.е.^2."""
        return self.n_items / (self.arena[0] * self.arena[1])

    def __post_init__(self) -> None:
        if self.tick_hz <= 0:
            raise ValueError("tick_hz must be > 0")
        if self.arena[0] <= 0 or self.arena[1] <= 0:
            raise ValueError("arena must be positive")
        if not 0.0 <= self.e_init <= self.e_max:
            raise ValueError("e_init must lie in [0, e_max]")
        if self.difficulty not in ("A1", "A2", "A3", "A4", "A5"):
            raise ValueError(f"unknown difficulty: {self.difficulty}")
        if self.difficulty == "A5":
            raise NotImplementedError(
                "A5 требует непроходимых препятствий, дающих окклюзию — "
                "их в геометрии мира пока нет. Молча работать как A4 нельзя: "
                "уровень называется по тому, что он проверяет."
            )
        if self.clock not in ("fast", "realtime"):
            raise ValueError("clock must be 'fast' or 'realtime'")
        if self.flip_interval[0] > self.flip_interval[1]:
            raise ValueError("flip_interval must be (lo, hi) with lo <= hi")
        if self.color_channels != 2:
            raise ValueError("Фаза 0 определена только для 2 цветовых каналов")
        if self.percept_level is not Level.P0:
            raise NotImplementedError(
                "Фаза 0 реализует уровень P0. P1/P2 — отдельный шаг лестницы "
                "(4.2), браться за них следует после того, как на P0 что-то "
                "работает."
            )

    # ---------------------------------------------------------------- сервис
    @staticmethod
    def justification(name: str) -> str:
        return JUSTIFICATION.get(name, "не помечено")

    def snapshot(self) -> dict[str, Any]:
        """Полный снимок констант вместе с пометками — для meta.json."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, IntEnum):
                value = value.name
            elif isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, dict):
                value = dict(value)
            out[f.name] = {"value": value, "justification": self.justification(f.name)}
        derived = {
            "dt": self.dt,
            "v_max": self.v_max,
            "crossing_time_s": self.crossing_time,
            "starve_time_idle_s": self.starve_time_idle,
            "starve_time_moving_s": self.starve_time_moving,
            "required_feed_rate_s_per_item": self.required_feed_rate,
            "item_density": self.item_density,
        }
        out["_derived"] = {"value": derived, "justification": DERIVED}
        return out

    def replace(self, **changes: Any) -> "Config":
        return dataclasses.replace(self, **changes)

    def arbitrary_constants(self) -> list[str]:
        """Имена всех констант с пометкой ПРОИЗВОЛ.

        Спецификация требует: если вывод эксперимента зависит от такой
        константы, эксперимент недействителен до её обоснования. Список
        нужен, чтобы это было чем проверять, а не только помнить.
        """
        return sorted(k for k, v in JUSTIFICATION.items() if v == ARBITRARY)
