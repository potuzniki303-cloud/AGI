"""Моторный тракт с удержанием последнего действия. Часть 1.3 и Часть 5.

Action-hold — это семантика зажатой клавиши, а не подачи вектора на тик.
Именно она делает бездействие физически осмысленным: агент, который молчит
на полном ходу, продолжает лететь и врежется в стену. Молчание имеет
последствия — значит, оно является выбором, а не пропуском хода.
"""

from __future__ import annotations

from .config import Config
from .events import Event, Motor


class MotorTract:
    """Удерживаемые значения моторных каналов."""

    __slots__ = ("cfg", "values", "_decay", "_n")

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._n = Motor.count(cfg)
        # Нейтраль: не толкаюсь, не поворачиваю.
        self.values = [0.0] * self._n
        self._decay = 1.0 - cfg.dt / cfg.tau_motor if cfg.motor_decay else 1.0

    def apply(self, event: Event) -> bool:
        """Принять моторное событие. Возвращает False, если канал неизвестен —
        мир не падает от мусора в outbox, но и не делает вид, что принял."""
        ch = event.channel
        if not 0 <= ch < self._n:
            return False
        value = float(event.value)
        if value != value:  # nan
            return False
        value = max(-1.0, min(1.0, value))
        if self.cfg.motor_discrete:
            value = float(round(value))
        self.values[ch] = value
        return True

    def decay(self) -> None:
        """MOTOR_DECAY: удерживаемое значение ползёт к нулю.

        По умолчанию выключено. Включать только как отдельный исследуемый
        фактор — оно меняет цену бездействия, а значит и смысл всех метрик.
        """
        if self._decay == 1.0:
            return
        for i in range(self._n):
            self.values[i] *= self._decay

    @property
    def thrust(self) -> float:
        return self.values[Motor.THRUST]

    @property
    def turn(self) -> float:
        return self.values[Motor.TURN]

    def state(self) -> list[float]:
        return list(self.values)

    def restore(self, values: list[float]) -> None:
        self.values = list(values)
