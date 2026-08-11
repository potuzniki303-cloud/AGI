"""Тесты gym-обёртки.

Основное, что тут проверяется помимо механики: обёртка не протаскивает в мир
ничего лишнего. Награда обязана оставаться нулём, а мир не должен трогать
разумы — в gym-режиме их держит вызывающий, и мир не вызывает у них даже
spawn().

    python3 -m pytest tests/test_gym_env.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

gym = pytest.importorskip("gymnasium")

from world import list_worlds, register_all  # noqa: E402
from world.gym_env import LifeEnv, gym_id  # noqa: E402
from world.stubs import GreedyMind, RandomMind  # noqa: E402

ALL_WORLDS = list_worlds()


@pytest.fixture(scope="module", autouse=True)
def _registered():
    register_all()


def drive(env, steps, mind_cls=RandomMind, seed=0):
    """Прогон с ведением популяции разумов на стороне вызывающего."""
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    n_act = env.unwrapped.world.action_size
    minds = {i: mind_cls(n_act, rng) for i in info["ids"]}

    for _ in range(steps):
        actions = [minds[i].act(o) for i, o in zip(info["ids"], obs)]
        obs, reward, terminated, truncated, info = env.step(actions)
        for parent, child in info["born"]:
            minds[child] = minds[parent].spawn(rng)
        for i in info["died"]:
            minds.pop(i, None)
        assert reward == 0.0
        if terminated or truncated:
            break
    return obs, info, minds


# ------------------------------------------------------------ регистрация

def test_gym_id_naming():
    assert gym_id("forage") == "Life/Forage-v0"
    assert gym_id("two_foods") == "Life/TwoFoods-v0"


def test_all_worlds_registered():
    ids = register_all()
    assert len(ids) == len(ALL_WORLDS)
    for ident in ids:
        assert ident in gym.registry


def test_register_all_is_idempotent():
    a = register_all()
    b = register_all()
    assert a == b


def test_new_world_gets_gym_id_automatically():
    """Реестр — одна точка правды: добавил мир, gym-идентификатор появился сам."""
    from world import ForageWorld, register

    register("smoke_world", ForageWorld)
    try:
        assert "Life/SmokeWorld-v0" in register_all()
        env = gym.make("Life/SmokeWorld-v0", size=10)
        env.close()
    finally:
        from world.registry import _REGISTRY

        _REGISTRY.pop("smoke_world", None)


# -------------------------------------------------------------- gym API

@pytest.mark.parametrize("name", ALL_WORLDS)
def test_make_and_run(name):
    env = gym.make(gym_id(name), size=14, seed=1)
    obs, info, minds = drive(env, 150)
    assert len(minds) == info["population"] == len(obs)
    env.close()


@pytest.mark.parametrize("name", ALL_WORLDS)
def test_observations_match_space(name):
    env = gym.make(gym_id(name), size=14, seed=1)
    obs, info = env.reset(seed=1)
    single = env.unwrapped.single_observation_space
    for o in obs:
        assert o.shape == single.shape
        assert o.dtype == np.float32
    assert env.observation_space.contains(tuple(obs))
    env.close()


def test_reward_is_always_zero():
    """В этом мире награды нет. Поле есть только потому, что его требует gym."""
    env = gym.make(gym_id("forage"), size=14, seed=1)
    obs, info = env.reset(seed=1)
    rng = np.random.default_rng(0)
    minds = {i: GreedyMind(4, rng) for i in info["ids"]}
    for _ in range(200):
        actions = [minds[i].act(o) for i, o in zip(info["ids"], obs)]
        obs, reward, term, trunc, info = env.step(actions)
        assert reward == 0.0 and isinstance(reward, float)
        for p, c in info["born"]:
            minds[c] = minds[p].spawn(rng)
        for i in info["died"]:
            minds.pop(i, None)
        if term:
            break
    env.close()


def test_ids_align_with_observations():
    env = gym.make(gym_id("forage"), size=14, seed=1)
    obs, info = env.reset(seed=1)
    assert len(info["ids"]) == len(obs)
    world = env.unwrapped.world
    assert info["ids"] == [b.id for b in world.bodies]
    # энергия в info идёт в том же порядке, что и наблюдения
    assert np.allclose(info["energy"], [b.energy for b in world.bodies])
    env.close()


def test_birth_and_death_log_is_consistent():
    """Журнал должен полностью объяснять изменение состава популяции."""
    env = gym.make(gym_id("forage"), size=16, seed=3)
    obs, info = env.reset(seed=3)
    rng = np.random.default_rng(3)
    minds = {i: GreedyMind(4, rng) for i in info["ids"]}

    for _ in range(300):
        before = set(info["ids"])
        actions = [minds[i].act(o) for i, o in zip(info["ids"], obs)]
        obs, _, term, trunc, info = env.step(actions)
        after = set(info["ids"])

        born = {c for _, c in info["born"]}
        died = set(info["died"])
        assert after == (before | born) - died, "журнал не объясняет состав"
        assert born.isdisjoint(before), "новорождённый не может быть старым id"
        assert died <= before, "умереть может только тот, кто жил"

        for p, c in info["born"]:
            assert p in before
            minds[c] = minds[p].spawn(rng)
        for i in info["died"]:
            minds.pop(i, None)
        if term:
            break
    env.close()


def test_wrong_number_of_actions_raises():
    env = gym.make(gym_id("forage"), size=14, seed=1)
    obs, info = env.reset(seed=1)
    with pytest.raises(ValueError, match="действий"):
        env.step([0] * (len(obs) + 3))
    env.close()


def test_out_of_range_action_raises():
    env = gym.make(gym_id("forage"), size=14, seed=1)
    obs, info = env.reset(seed=1)
    with pytest.raises(ValueError, match="out of range"):
        env.step([99] * len(obs))
    env.close()


def test_terminated_on_extinction():
    # ни еды, ни отрастания — популяция обязана вымереть
    env = gym.make(
        gym_id("forage"), size=10, seed=1, initial_population=4,
        initial_food_density=0.0, max_food=0, regrow_per_step=0.0,
        energy_at_birth=4.0,
    )
    obs, info = env.reset(seed=1)
    terminated = False
    for _ in range(30):
        obs, _, terminated, _, info = env.step([0] * len(obs))
        if terminated:
            break
    assert terminated
    assert info["population"] == 0
    env.close()


def test_deterministic_given_seed():
    def run():
        env = gym.make(gym_id("forage"), size=14)
        _, info, _ = drive(env, 120, GreedyMind, seed=5)
        env.close()
        return info["tick"], info["population"], info["food"]

    assert run() == run()


def test_world_never_touches_minds_in_gym_mode():
    """В gym-режиме мир не должен вызывать у разумов вообще ничего.

    Это более строгая граница, чем в simulate(): там мир дёргает spawn(),
    здесь популяцию ведёт вызывающий, и мир не касается ни одного Mind."""
    touched: set[str] = set()

    class Spy:
        def act(self, obs):
            touched.add("act")
            return 0

        def spawn(self, rng):
            touched.add("spawn")
            return Spy()

        def __getattr__(self, item):
            touched.add(item)
            raise AttributeError(item)

    env = gym.make(gym_id("forage"), size=14, seed=1)
    obs, info = env.reset(seed=1)
    # разумы существуют, но миру не передаются
    minds = {i: Spy() for i in info["ids"]}
    for _ in range(100):
        actions = [minds[i].act(o) for i, o in zip(info["ids"], obs)]
        obs, _, term, _, info = env.step(actions)
        for p, c in info["born"]:
            minds[c] = minds[p].spawn(np.random.default_rng(0))
        for i in info["died"]:
            minds.pop(i, None)
        if term:
            break

    for body in env.unwrapped.world.bodies:
        assert body.mind is None, "мир завёл себе разум в gym-режиме"
    env.close()


def test_direct_construction_without_gym_make():
    env = LifeEnv("seasons", size=12, seed=1)
    obs, info = env.reset(seed=1)
    assert len(obs) == info["population"]
    env.close()


def test_config_kwargs_reach_the_world():
    env = gym.make(gym_id("forage"), size=17, regrow_per_step=3.5)
    assert env.unwrapped.world.cfg.size == 17
    assert env.unwrapped.world.cfg.regrow_per_step == 3.5
    env.close()


def test_bad_config_key_raises():
    with pytest.raises(TypeError):
        gym.make(gym_id("forage"), no_such_knob=1)


def test_bad_render_mode_raises():
    with pytest.raises(ValueError, match="render_mode"):
        LifeEnv("forage", render_mode="hologram")


# ------------------------------------------------------------- отрисовка

def test_ansi_render():
    env = gym.make(gym_id("forage"), size=10, seed=1, render_mode="ansi")
    env.reset(seed=1)
    frame = env.render()
    assert isinstance(frame, str)
    assert "живых=" in frame
    env.close()


def test_render_none_returns_none():
    env = LifeEnv("forage", size=10, seed=1)
    env.reset(seed=1)
    assert env.render() is None
    env.close()
