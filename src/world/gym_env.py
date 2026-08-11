"""Gym-обёртка над мирами.

Обращение такое же, как к старому GridWorld:

    import gymnasium as gym
    from world.gym_env import register_all

    register_all()
    env = gym.make("Life/Forage-v0", size=24, render_mode="human")
    obs, info = env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(actions)

Дальше — три места, где этот мир не помещается в gym, и что с ними сделано.

1. НАГРАДЫ НЕТ. gym требует поле reward, и оно всегда 0.0. Так же было в
   твоём старом GridWorld. Заполнить его чем-нибудь осмысленным означало бы
   ровно то, чего весь проект избегает: придумать скаляр «хорошо/плохо» и
   начать его максимизировать. Поле есть, потому что этого требует подпись
   step(); смысла в нём нет.

2. ТЕЛ МНОГО, И ИХ ЧИСЛО МЕНЯЕТСЯ. gym рассчитан на одного агента, а тут
   популяция, в которой каждый такт кто-то умирает и рождается. Поэтому
   наблюдение — это последовательность переменной длины (spaces.Sequence),
   а действий надо подавать столько же, сколько пришло наблюдений, в том же
   порядке. Соответствие «какое наблюдение чьё» лежит в info["ids"].

3. РАЗУМЫ ДЕРЖИТ ВЫЗЫВАЮЩИЙ. Мир не хранит агентов и не вызывает у них
   ничего — даже spawn(). Про рождения и смерти он сообщает журналом:
   info["born"] = [(родитель, потомок), ...] и info["died"] = [id, ...].
   Что делать с разумами, решаешь ты. Это более строгая граница, чем в
   simulate(): там мир хотя бы дёргает spawn(), здесь не дёргает ничего.

Типичный цикл целиком:

    obs, info = env.reset(seed=1)
    minds = {i: make_mind(rng) for i in info["ids"]}

    while True:
        actions = [minds[i].act(o) for i, o in zip(info["ids"], obs)]
        obs, _, terminated, truncated, info = env.step(actions)
        for parent, child in info["born"]:
            minds[child] = minds[parent].spawn(rng)
        for i in info["died"]:
            del minds[i]
        if terminated or truncated:
            break

terminated=True означает вымирание популяции — единственный способ, которым
этот мир заканчивается сам. truncated=True — исчерпан max_episode_steps,
если ты его задал.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import World
from .registry import list_worlds, world_class

# Имена сценариев в gym собираются из того же реестра, что и make():
# добавил мир в реестр — gym-идентификатор появился сам, руками ничего
# дописывать не надо.
DEFAULT_PREFIX = "Life"


def gym_id(world_name: str, prefix: str = DEFAULT_PREFIX, version: int = 0) -> str:
    """forage -> Life/Forage-v0, two_foods -> Life/TwoFoods-v0"""
    camel = "".join(part.capitalize() for part in world_name.split("_"))
    return f"{prefix}/{camel}-v{version}"


def register_all(prefix: str = DEFAULT_PREFIX, max_episode_steps: int | None = None) -> list[str]:
    """Зарегистрировать все миры реестра как сценарии gym.

    Возвращает список идентификаторов. Повторный вызов безопасен.
    """
    ids = []
    for name in list_worlds():
        ident = gym_id(name, prefix)
        if ident not in gym.registry:
            gym.register(
                id=ident,
                entry_point="world.gym_env:LifeEnv",
                kwargs={"world_name": name},
                max_episode_steps=max_episode_steps,
            )
        ids.append(ident)
    return ids


class LifeEnv(gym.Env):
    """Мир как gym-среда.

    world_name  — имя из реестра ("forage", "seasons", ...)
    render_mode — "human" (окно arcade), "rgb_array" (кадр numpy),
                  "ansi" (текст), None
    остальные именованные аргументы уходят в Config мира.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 15,
    }

    def __init__(
        self,
        world_name: str = "forage",
        render_mode: str | None = None,
        **world_kwargs: Any,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        cls = world_class(world_name)
        self.world_name = world_name
        self.world: World = cls(cls.Config(**world_kwargs))

        self.observation_space = spaces.Sequence(
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.world.observation_size,),
                dtype=np.float32,
            )
        )
        self.action_space = spaces.Sequence(spaces.Discrete(self.world.action_size))

        # Одиночные пространства: по ним агент строит сеть. Их удобнее
        # спрашивать, чем разбирать Sequence.
        self.single_observation_space = self.observation_space.feature_space
        self.single_action_space = self.action_space.feature_space

        self._renderer = None
        self._ids: list[int] = []

    # ------------------------------------------------------------ gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.world.cfg.seed = seed

        # mind_factory=None: тела рождаются без разумов, их держит вызывающий
        self.world.reset(None)

        ids, obs = self.world.observe_all()
        self._ids = ids
        if self.render_mode == "human":
            self.render()
        return tuple(obs), self._info(ids, born=[], died=[])

    def step(
        self, actions: Sequence[int]
    ) -> tuple[tuple[np.ndarray, ...], float, bool, bool, dict[str, Any]]:
        report = self.world.apply_actions(list(actions))

        ids, obs = self.world.observe_all()
        self._ids = ids

        if self.render_mode == "human":
            self.render()

        info = self._info(ids, born=report.born, died=report.died)
        info["report"] = report

        # reward всегда 0.0 — см. пункт 1 в шапке модуля
        return tuple(obs), 0.0, bool(report.extinct), False, info

    def render(self) -> Any:
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            from .render import to_ascii

            return to_ascii(self.world)

        from .render_arcade import ArcadeRenderer

        if self._renderer is None:
            self._renderer = ArcadeRenderer(
                self.world,
                visible=(self.render_mode == "human"),
                fps=self.metadata["render_fps"],
            )
        return self._renderer.draw(throttle=(self.render_mode == "human"))

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -------------------------------------------------------------- прочее

    def _info(
        self, ids: list[int], born: list[tuple[int, int]], died: list[int]
    ) -> dict[str, Any]:
        bodies = self.world.bodies
        return {
            "ids": ids,
            "born": born,
            "died": died,
            "tick": self.world.tick,
            "population": len(bodies),
            "food": int(np.count_nonzero(self.world.food)),
            "energy": np.array([b.energy for b in bodies], dtype=np.float32),
            "age": np.array([b.age for b in bodies], dtype=np.int64),
            "generation": np.array([b.generation for b in bodies], dtype=np.int64),
        }

    def __repr__(self) -> str:
        return f"LifeEnv({self.world_name}, obs={self.world.observation_size}, act={self.world.action_size})"
