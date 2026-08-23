"""Линейка для предсказания сенсорного потока.

Это ПРИБОР, а не агент. Здесь нет ни узлов, ни правила обучения — только
тривиальный эталон, с которым сравнивают, и способ посчитать ошибку.

Зачем. Вопрос «предсказывает ли агент поток лучше тривиального предсказателя
«следующий кадр равен текущему» нельзя задать, пока тривиальный предсказатель
не реализован и не измерен. Мерка должна существовать раньше того, что ею
меряют, иначе первое же число окажется не с чем сравнить.

ВАЖНАЯ ОГОВОРКА, которую видно только замером. Неподвижное тело в статичном
мире предсказывать НЕЧЕГО: на A1-A3 предметы стоят, устойчивый канал
постоянен, и персистентный предсказатель даёт ошибку около нуля. Задача
становится непустой, только если что-то движется — предметы (A4) или само
тело. Это надо решить до того, как строить предсказывающего агента, иначе он
будет соревноваться с эталоном на задаче, где эталон идеален.

Протокол предсказателя (реализует автор, мир о нём не знает):

    def predict(self, frame: np.ndarray) -> np.ndarray: ...
    def observe(self, frame: np.ndarray) -> None: ...   # необязательно
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .events import Channels, Event
from .world import World


class TrivialPredictor:
    """«Следующий кадр равен текущему». Эталон, который надо превзойти.

    Он не так глуп, как кажется: на медленно меняющемся входе персистентность
    очень сильна, и переиграть её труднее, чем ожидаешь. Именно поэтому она и
    выбрана эталоном.
    """

    name = "persistence"

    def predict(self, frame: np.ndarray) -> np.ndarray:
        return frame.copy()

    def observe(self, frame: np.ndarray) -> None:
        pass


class MeanPredictor:
    """«Следующий кадр равен скользящему среднему». Второй эталон.

    Нужен, чтобы отличить «агент выучил динамику» от «агент выучил, что вход
    почти постоянен». Если агент бьёт персистентность, но не бьёт среднее,
    он поймал уровень сигнала, а не его движение.
    """

    name = "running_mean"

    def __init__(self, tau: float = 0.2) -> None:
        self.alpha = tau
        self.state: np.ndarray | None = None

    def predict(self, frame: np.ndarray) -> np.ndarray:
        return frame.copy() if self.state is None else self.state.copy()

    def observe(self, frame: np.ndarray) -> None:
        if self.state is None:
            self.state = frame.copy()
        else:
            self.state += self.alpha * (frame - self.state)


@dataclass
class PredictionScore:
    """Ошибка предсказания. Только для человека, в мир не возвращается."""

    ticks: int = 0
    sq_error: float = 0.0
    sq_signal: float = 0.0
    abs_change: float = 0.0
    per_tick: list[float] = field(default_factory=list)

    def add(self, predicted: np.ndarray, actual: np.ndarray,
            previous: np.ndarray) -> None:
        err = float(np.sum((predicted - actual) ** 2))
        self.ticks += 1
        self.sq_error += err
        self.sq_signal += float(np.sum(actual ** 2))
        self.abs_change += float(np.sum(np.abs(actual - previous)))
        self.per_tick.append(err)

    @property
    def mse(self) -> float:
        return self.sq_error / max(1, self.ticks)

    @property
    def normalised(self) -> float:
        """Ошибка, делённая на энергию сигнала. 0 — идеально, 1 — как нуль."""
        return self.sq_error / max(1e-12, self.sq_signal)

    @property
    def change_rate(self) -> float:
        """Насколько вообще шевелится вход. Если около нуля, предсказывать
        нечего и любое сравнение предсказателей бессмысленно."""
        return self.abs_change / max(1, self.ticks)

    def row(self) -> dict[str, float]:
        return {
            "MSE": round(self.mse, 6),
            "норм. ошибка": round(self.normalised, 5),
            "движение входа": round(self.change_rate, 4),
        }


def sustained_frame(events: list[Event], channels: Channels,
                    buffer: np.ndarray) -> np.ndarray:
    """Собрать вектор устойчивого канала из событий тика."""
    for e in events:
        if channels.sustained_start <= e.channel < channels.proprio_start:
            buffer[e.channel - channels.sustained_start] = e.value
    return buffer


def score_predictor(predictor, cfg: Config, ticks: int = 20_000,
                    driver=None) -> PredictionScore:
    """Прогнать предсказателя по устойчивому каналу.

    `driver(world, tick)` — то, что двигает тело. None означает полную
    неподвижность: агент только смотрит. Смотри оговорку в шапке модуля.
    """
    world = World(cfg)
    channels = world.channels
    size = channels.n_sustained
    buffer = np.zeros(size, dtype=np.float64)
    score = PredictionScore()

    world.step()
    frame = sustained_frame(world.inbox.drain(), channels, buffer).copy()

    for tick in range(ticks):
        predicted = np.asarray(predictor.predict(frame), dtype=np.float64)
        if driver is not None:
            driver(world, tick)
        world.step()
        previous = frame
        frame = sustained_frame(world.inbox.drain(), channels, buffer).copy()
        score.add(predicted, frame, previous)
        if hasattr(predictor, "observe"):
            predictor.observe(frame)
    return score
