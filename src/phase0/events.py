"""Событие и два неблокирующих канала. Часть 1 спецификации.

Здесь живёт главный архитектурный инвариант Фазы 0:

    Мир всегда принимает действие, но никогда его не требует.

В этом модуле (и во всём пакете) НЕТ и не должно появиться функции вида
`get_action(agent)`. Мир не знает про агента вообще: он пишет в `inbox` и
читает из `outbox`. Кто и когда положит событие в `outbox` — не его дело.
Проверка при ревью: если появился код, который на каждом тике идёт снимать
состояние агента ради действия, контракт нарушен, даже если снаружи всё
выглядит асинхронно.

Молчание агента — легальное состояние. Мир обязан работать вечно при пустом
`outbox`, и на это есть тест.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from .config import Config, Level


@dataclass(slots=True, frozen=True)
class Event:
    """Единица обмена в обе стороны.

    t       — логический тик эмиссии
    channel — идентификатор канала в своём пространстве (сенсорном/моторном)
    value   — полезная нагрузка
    sign    — только для транзиентных сенсорных событий: +1 / -1, иначе 0
    """

    t: int
    channel: int
    value: float
    sign: int = 0

    def as_row(self) -> tuple[int, int, float, int]:
        return (self.t, self.channel, self.value, self.sign)


class Channels:
    """Раскладка сенсорного пространства каналов для уровня P0.

    Пространство плоское: один int на канал, без вложенности. Так агенту не
    нужно знать структуру, чтобы принимать события, но при желании он может
    её восстановить — раскладка публична и стабильна.

    P0:
      [0, 128)     TRANSIENT  32 рецептора x 2 цвета x 2 полярности
      [128, 144)   SUSTAINED  8 рецепторов x 2 цвета
      [144, 148)   PROPRIO    v_forward, v_lateral, omega, |v|
      [148, 150)   INTERO     ENERGY, NOCI

    Про полярность. Спецификация (4.4) велит эмитить channel=(i,c) со знаком,
    а таблица 4.2 считает 128 каналов, то есть с расщеплением ON/OFF. Здесь
    сделано и то и другое: полярность входит в идентификатор канала (чтобы
    размерность совпадала с таблицей), и одновременно проставлен `sign`.
    Агент волен пользоваться любым из двух представлений.
    """

    def __init__(self, cfg: Config) -> None:
        if cfg.percept_level is not Level.P0:
            raise NotImplementedError("Channels пока описывает только P0")
        self.cfg = cfg
        n, c = cfg.retina_n, cfg.color_channels
        s = cfg.sustained_n

        self.transient_start = 0
        self.n_transient = n * c * 2
        self.sustained_start = self.transient_start + self.n_transient
        self.n_sustained = s * c
        self.proprio_start = self.sustained_start + self.n_sustained
        self.n_proprio = 4
        self.intero_start = self.proprio_start + self.n_proprio
        self.n_intero = 2
        self.n_total = self.intero_start + self.n_intero

        self.ENERGY = self.intero_start
        self.NOCI = self.intero_start + 1
        self.V_FORWARD = self.proprio_start
        self.V_LATERAL = self.proprio_start + 1
        self.OMEGA = self.proprio_start + 2
        self.SPEED = self.proprio_start + 3

    # ------------------------------------------------------------ адресация
    def transient(self, receptor: int, color: int, polarity: int) -> int:
        """polarity: 0 = OFF (сигнал упал), 1 = ON (сигнал вырос)."""
        return self.transient_start + (receptor * self.cfg.color_channels + color) * 2 + polarity

    def sustained(self, receptor: int, color: int) -> int:
        return self.sustained_start + receptor * self.cfg.color_channels + color

    def decode_transient(self, channel: int) -> tuple[int, int, int]:
        """Обратное к transient(): канал -> (рецептор, цвет, полярность)."""
        idx = channel - self.transient_start
        polarity = idx % 2
        rest = idx // 2
        return rest // self.cfg.color_channels, rest % self.cfg.color_channels, polarity

    def decode_sustained(self, channel: int) -> tuple[int, int]:
        idx = channel - self.sustained_start
        return divmod(idx, self.cfg.color_channels)

    def group_of(self, channel: int) -> str:
        if channel < self.sustained_start:
            return "TRANSIENT"
        if channel < self.proprio_start:
            return "SUSTAINED"
        if channel < self.intero_start:
            return "PROPRIO"
        return "INTERO"

    def describe(self) -> dict[str, tuple[int, int]]:
        """Границы групп: имя -> (начало, количество). Пишется в meta.json."""
        return {
            "TRANSIENT": (self.transient_start, self.n_transient),
            "SUSTAINED": (self.sustained_start, self.n_sustained),
            "PROPRIO": (self.proprio_start, self.n_proprio),
            "INTERO": (self.intero_start, self.n_intero),
        }


class Motor:
    """Пространство моторных каналов (Часть 5). Отдельное от сенсорного."""

    THRUST = 0
    TURN = 1
    SACCADE_X = 2  # только P2
    SACCADE_Y = 3  # только P2

    NAMES = {THRUST: "thrust", TURN: "turn",
             SACCADE_X: "saccade_x", SACCADE_Y: "saccade_y"}

    @staticmethod
    def count(cfg: Config) -> int:
        return 4 if cfg.percept_level is Level.P2 else 2


class Queue:
    """Неблокирующая очередь событий.

    Намеренно тупая: список плюс осушение. Никаких блокировок, никаких
    таймаутов, никакого «подожди, пока появится». Обе стороны в любой момент
    могут обнаружить пустоту, и это нормальное состояние, а не ошибка.
    """

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[Event] = []

    def put(self, event: Event) -> None:
        self._items.append(event)

    def extend(self, events: Iterable[Event]) -> None:
        self._items.extend(events)

    def drain(self) -> list[Event]:
        """Забрать всё и очистить. Возвращает список в порядке поступления."""
        out = self._items
        self._items = []
        return out

    def peek(self) -> tuple[Event, ...]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"Queue({len(self._items)} events)"
