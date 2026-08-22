"""Именованные потоки случайности. Часть 10.1.

Один поток на подсистему, чтобы изменение одной не сдвигало последовательности
остальных: добавил режим — не поехали позиции предметов. Поток агента сюда не
входит и входить не должен, он полностью независим (иначе форк по seed агента
при фиксированной истории мира становится невозможен).
"""

from __future__ import annotations

from typing import Any

import numpy as np

STREAMS = ("world_layout", "item_respawn", "regime_flips", "sensor_noise")


class RngBundle:
    """Набор независимых генераторов, умеющий сохранять и восстанавливать
    своё состояние побитово."""

    __slots__ = ("_gens", "_seeds")

    def __init__(self, seeds: dict[str, int]) -> None:
        missing = [s for s in STREAMS if s not in seeds]
        if missing:
            raise ValueError(f"нет seed для потоков: {missing}")
        self._seeds = {name: int(seeds[name]) for name in STREAMS}
        # SeedSequence по имени потока: одинаковый seed в разных потоках
        # не даёт одинаковых последовательностей.
        self._gens = {
            name: np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([self._seeds[name], idx]))
            )
            for idx, name in enumerate(STREAMS)
        }

    def __getattr__(self, name: str) -> np.random.Generator:
        try:
            return self.__getitem__(name)
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name: str) -> np.random.Generator:
        return self._gens[name]

    @property
    def seeds(self) -> dict[str, int]:
        return dict(self._seeds)

    def state(self) -> dict[str, Any]:
        return {
            "seeds": dict(self._seeds),
            "streams": {name: gen.bit_generator.state for name, gen in self._gens.items()},
        }

    def restore(self, state: dict[str, Any]) -> None:
        self._seeds = {k: int(v) for k, v in state["seeds"].items()}
        for name, bg_state in state["streams"].items():
            self._gens[name].bit_generator.state = bg_state
