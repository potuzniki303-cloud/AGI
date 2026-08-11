"""Модульные миры для экспериментов с обучением без backprop.

Быстрый старт:

    from world import make, simulate
    from world.stubs import RandomMind

    w = make("forage", size=24, seed=1)
    print(w.describe())                       # размер входа и число действий

    h = simulate(w, lambda rng: RandomMind(w.action_size, rng), steps=2000)
    print(h.summary())

Подключение своего агента — это ровно два метода, см. protocols.Mind:

    class MyAgent:
        def act(self, observation) -> int: ...
        def spawn(self, rng) -> "MyAgent": ...

Мир не знает и не хочет знать, что внутри. Он не трогает веса, не считает
награду и никого не ранжирует. Его работа — быть, кормить и убивать.

Смена мира — одна строка: make("seasons"), make("shift"), make("two_foods").
Смена формы наблюдения — переопредели _build_sensor() или собери сенсор
из sensors.py и присвой world.sensor до reset().

Второй путь — через gym, обращение как к старому GridWorld:

    import gymnasium as gym
    from world import register_all

    register_all()                       # Life/Forage-v0, Life/Seasons-v0, ...
    env = gym.make("Life/Forage-v0", size=24, render_mode="human")
    obs, info = env.reset(seed=1)

Тут разумы держит вызывающий, а мир сообщает о рождениях и смертях журналом
в info. Подробности и готовый цикл — в gym_env.py.

gymnasium подтягивается только если ты его попросил: сам по себе мир от него
не зависит, как и от torch.
"""

from __future__ import annotations

from .core import Body, StepReport, World, WorldConfig
from .protocols import Mind
from .render import print_frame, to_ascii
from .runner import History, simulate
from .sensors import (
    AntennaSensor,
    ConcatSensor,
    Interoception,
    PatchSensor,
    Sensor,
)
from .worlds import (
    ForageWorld,
    PatchesConfig,
    PatchesWorld,
    SeasonsConfig,
    SeasonsWorld,
    ShiftConfig,
    ShiftWorld,
    TwoFoodsConfig,
    TwoFoodsWorld,
)

from .registry import list_worlds, make, register, world_class


def __getattr__(name: str):
    """Ленивый доступ к gym-обёртке.

    Импортировать gymnasium при `import world` нельзя: мир самодостаточен и
    не должен требовать gym от того, кто пользуется собственным циклом.
    """
    if name in ("LifeEnv", "register_all", "gym_id"):
        from . import gym_env

        return getattr(gym_env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "make",
    "register",
    "list_worlds",
    "world_class",
    "gym_id",
    "register_all",
    "simulate",
    "History",
    "World",
    "WorldConfig",
    "Body",
    "StepReport",
    "Mind",
    "Sensor",
    "AntennaSensor",
    "PatchSensor",
    "Interoception",
    "ConcatSensor",
    "ForageWorld",
    "SeasonsWorld",
    "SeasonsConfig",
    "PatchesWorld",
    "PatchesConfig",
    "ShiftWorld",
    "ShiftConfig",
    "TwoFoodsWorld",
    "TwoFoodsConfig",
    "to_ascii",
    "print_frame",
]
