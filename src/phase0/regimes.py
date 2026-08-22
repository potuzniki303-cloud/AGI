"""Режимы мира: A стационарный, B реверсия, C ритм. Части 7-9.

Питательность живёт ЗДЕСЬ, а не в предмете. Это не педантизм: весь режим B
построен на том, что внешность предмета не меняется, а правило меняется.
Если бы `nutritive` было полем Item, режим B пришлось бы реализовывать
подменой внешности, и эксперимент проверял бы не то.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, Kind, Mode


@dataclass(slots=True)
class RegimeState:
    """То, что пишется в канал истины и не попадает агенту."""

    mode: str
    flag_state: int
    ticks_since_flip: int
    next_flip: int
    phase: float = 0.0
    window_open: bool = True


class Regime:
    """Правило «что сейчас питательно»."""

    def __init__(self, cfg: Config, rng_flips: np.random.Generator) -> None:
        self.cfg = cfg
        self._rng = rng_flips
        self.flag_state = 0
        self.last_flip_tick = 0
        self.next_flip = self._draw_interval() if cfg.mode is Mode.B else -1
        self.flip_count = 0

    def _draw_interval(self) -> int:
        lo, hi = self.cfg.flip_interval
        return int(self._rng.integers(lo, hi + 1))

    # ---------------------------------------------------------------- такт
    def step(self, tick: int) -> bool:
        """Продвинуть режим. Возвращает True, если случился переворот.

        Переворот происходит без предупреждения: ни сигнала, ни изменения
        внешности. Единственный способ узнать — съесть и получить NOCI.
        """
        if self.cfg.mode is not Mode.B:
            return False
        if tick >= self.last_flip_tick + self.next_flip:
            self.flag_state ^= 1
            self.last_flip_tick = tick
            self.next_flip = self._draw_interval()
            self.flip_count += 1
            return True
        return False

    # --------------------------------------------------------- питательность
    def nutritive(self, kind: Kind, tick: int) -> float:
        """Сколько энергии даёт предмет данного вида прямо сейчас."""
        cfg = self.cfg
        if cfg.mode is Mode.C:
            return cfg.e_food if self._window_open(tick) else 0.0

        effective = kind
        if cfg.mode is Mode.B and self.flag_state:
            effective = Kind.B if kind is Kind.A else Kind.A

        if cfg.mode is Mode.B:
            # Режим B устроен как A3: один вид питателен, другой ядовит.
            return cfg.e_food if effective is Kind.A else cfg.e_poison

        level = cfg.difficulty
        if level == "A1":
            return cfg.e_food
        if level == "A2":
            return cfg.e_food if kind is Kind.A else 0.0
        # A3, A4, A5
        return cfg.e_food if kind is Kind.A else cfg.e_poison

    def _window_open(self, tick: int) -> bool:
        phase = (tick % self.cfg.rhythm_period) / self.cfg.rhythm_period
        return phase < self.cfg.rhythm_window

    def phase(self, tick: int) -> float:
        if self.cfg.mode is not Mode.C:
            return 0.0
        return (tick % self.cfg.rhythm_period) / self.cfg.rhythm_period

    def state(self, tick: int) -> RegimeState:
        return RegimeState(
            mode=self.cfg.mode.name,
            flag_state=self.flag_state,
            ticks_since_flip=tick - self.last_flip_tick,
            next_flip=self.next_flip,
            phase=self.phase(tick),
            window_open=self._window_open(tick) if self.cfg.mode is Mode.C else True,
        )

    # ------------------------------------------------------------- снапшот
    def snapshot(self) -> dict:
        return {
            "flag_state": self.flag_state,
            "last_flip_tick": self.last_flip_tick,
            "next_flip": self.next_flip,
            "flip_count": self.flip_count,
        }

    def restore(self, state: dict) -> None:
        self.flag_state = state["flag_state"]
        self.last_flip_tick = state["last_flip_tick"]
        self.next_flip = state["next_flip"]
        self.flip_count = state["flip_count"]
