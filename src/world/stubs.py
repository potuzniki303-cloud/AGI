"""Заглушки для проверки самого мира. Это НЕ агенты.

Нужны ровно для одного: убедиться, что физика работает и что мир вообще
проходим, до того как в него подключится настоящий Mind. В них нет ни весов,
ни пластичности, и добавлять их сюда не надо — этот файл про тесты мира.

GreedyMind стоит отдельного слова. Он смотрит на антенны и идёт к самому
сильному сигналу. Это не то, что должен выучить агент, — это верхняя отметка
на линейке. Если твой агент выходит на уровень GreedyMind в forage, значит
правило пластичности нашло таксис.

Важно, как он ведёт себя в двух последних мирах — он там играет роль генома,
который несёт готовый ответ:

  shift      — держится, пока тело не сломали, потом превращается в
               случайного ходока и умирает. Читать надо не популяцию
               (её держит скорость отрастания еды), а хвост
               распределения жизней.
  two_foods  — выживает примерно на половинной популяции, потому что
               полярность разыгрывается при каждом рождении и «иди к типу 1»
               оказывается верным ровно у половины тел. Это и есть цена
               зашитого ответа. Агент, который выясняет полярность внутри
               жизни, должен выходить на уровень forage, а не на половину.
"""

from __future__ import annotations

import numpy as np


class RandomMind:
    """Случайные действия. Нижняя отметка на линейке."""

    def __init__(self, n_actions: int, rng: np.random.Generator) -> None:
        self.n_actions = n_actions
        self.rng = rng

    def act(self, observation: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))

    def spawn(self, rng: np.random.Generator) -> "RandomMind":
        return RandomMind(self.n_actions, rng)


class StillMind:
    """Не двигается (нужен allow_stay=True). Проверяет, что сидеть невыгодно."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    def act(self, observation: np.ndarray) -> int:
        return 4 if self.n_actions > 4 else 0

    def spawn(self, rng: np.random.Generator) -> "StillMind":
        return StillMind(self.n_actions)


class GreedyMind:
    """Идёт к самому сильному пищевому сенсору. Захардкожено, ничего не учит.

    Работает только с сенсором, у которого первые 4 значения — пищевые
    антенны в порядке действий (то есть с дефолтным сенсором forage/seasons).
    """

    def __init__(self, n_actions: int, rng: np.random.Generator, noise: float = 0.1) -> None:
        self.n_actions = n_actions
        self.rng = rng
        self.noise = noise

    def act(self, observation: np.ndarray) -> int:
        if self.rng.random() < self.noise:
            return int(self.rng.integers(self.n_actions))
        antenna = observation[:4]
        if not np.any(antenna > 0):
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(antenna))

    def spawn(self, rng: np.random.Generator) -> "GreedyMind":
        return GreedyMind(self.n_actions, rng, self.noise)
