"""Переходник между агентом и контрактом мира.

Твой Agent говорит forward()/mutate(), мир спрашивает act()/spawn(). Здесь
только перевод, никакой логики обучения.

Про torch: этот файл его не импортирует и импортировать не должен. Мир не
знает, на чём написан агент, и это стоит сохранить — переходник работает с
любым объектом, у которого есть forward() и mutate().

Отдельно про сэмплинг. Он живёт ЗДЕСЬ, а не в мире, и это принципиально:
выбор действия из логитов — часть политики, то есть агента. Если сэмплировать
за агента внутри мира, кусок политики окажется в руках дизайнера. Температуру
и способ выбора решаешь ты.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _identity(observation: np.ndarray) -> Any:
    return observation


class MindAdapter:
    """Оборачивает агента с интерфейсом forward()/mutate() в контракт мира.

    agent      — твой объект: forward(obs) -> логиты, mutate() -> потомок
    n_actions  — сколько первых логитов считать действиями
    to_input   — как превратить np.float32 в то, что ест forward().
                 Для torch: lambda o: torch.from_numpy(o).to(device)
    temperature— 0 означает argmax. Больше нуля — softmax-сэмплинг.

    Про температуру: если последний слой у тебя под tanh, логиты лежат в
    (-1, 1), и softmax по ним почти равномерный. Либо снижай температуру
    до ~0.1, либо не сжимай выход действий tanh-ом. Детерминированный argmax
    в рекуррентной сети залипает в неподвижную точку, так что совсем без
    шума тоже нельзя.
    """

    def __init__(
        self,
        agent: Any,
        n_actions: int,
        rng: np.random.Generator,
        to_input: Callable[[np.ndarray], Any] = _identity,
        temperature: float = 0.25,
    ) -> None:
        self.agent = agent
        self.n_actions = n_actions
        self.rng = rng
        self.to_input = to_input
        self.temperature = float(temperature)

    def act(self, observation: np.ndarray) -> int:
        logits = self.agent.forward(self.to_input(observation))
        logits = np.asarray(_to_numpy(logits), dtype=np.float64)[: self.n_actions]

        # Разошедшиеся веса дают nan/inf. Молча превращать это в действие 0
        # нельзя: получится агент, который «умеет ходить на север», хотя на
        # самом деле он сломан. Лучше падать громко.
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError(
                f"агент выдал нечисловые логиты: {logits}. "
                "Скорее всего, правило пластичности разогнало веса."
            )

        if self.temperature <= 0.0:
            return int(np.argmax(logits))

        z = logits / self.temperature
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(self.rng.choice(self.n_actions, p=p))

    def spawn(self, rng: np.random.Generator) -> "MindAdapter":
        return MindAdapter(
            self.agent.mutate(),
            self.n_actions,
            rng,
            self.to_input,
            self.temperature,
        )


def _to_numpy(x: Any) -> Any:
    """Снять torch.Tensor до numpy, не импортируя torch."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return x
