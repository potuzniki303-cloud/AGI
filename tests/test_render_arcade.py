"""Тесты отрисовки через arcade.

Arcade требует дисплей, поэтому тесты сами пропускаются там, где окна нет.
Чтобы прогнать их на headless-машине:

    xvfb-run -a python3 -m pytest tests/test_render_arcade.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

arcade = pytest.importorskip("arcade")
gym = pytest.importorskip("gymnasium")


def _display_available() -> bool:
    try:
        w = arcade.Window(64, 64, "probe", visible=False)
        w.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _display_available(), reason="нет дисплея (запусти под xvfb-run)"
)

from world import make, register_all  # noqa: E402
from world.gym_env import gym_id  # noqa: E402
from world.render_arcade import ArcadeRenderer  # noqa: E402
from world.stubs import GreedyMind  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _registered():
    register_all()


def test_renderer_produces_nonempty_frame():
    world = make("forage", size=12, seed=1)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    for _ in range(30):
        world.step()

    r = ArcadeRenderer(world, visible=False)
    frame = r.draw(throttle=False)
    r.close()

    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    # не однотонная заливка: на поле есть еда и тела
    assert frame.std() > 1.0


def test_frame_size_follows_world_size():
    for size in (8, 14):
        world = make("forage", size=size, seed=1)
        world.reset(lambda rng: GreedyMind(world.action_size, rng))
        r = ArcadeRenderer(world, visible=False)
        frame = r.draw(throttle=False)
        r.close()
        assert frame.shape[1] == r.side
        assert frame.shape[0] == r.side + 26  # поле + строка состояния


def test_two_food_types_are_visually_distinct():
    """В two_foods типы обязаны различаться на вид — агент их тоже различает."""
    world = make("two_foods", size=10, seed=1, initial_population=0)
    world.reset(None)
    world.food[:] = 0

    world.food[2, 2] = 1
    r = ArcadeRenderer(world, visible=False)
    only_first = r.draw(throttle=False).copy()

    world.food[2, 2] = 2
    only_second = r.draw(throttle=False).copy()
    r.close()

    assert not np.array_equal(only_first, only_second), "типы еды выглядят одинаково"


def test_energy_changes_brightness():
    world = make("forage", size=8, seed=1, initial_population=0)
    world.reset(None)
    world.food[:] = 0

    from world.core import Body

    body = Body(id=0, x=4, y=4, energy=world.cfg.energy_max, mind=None)
    world.bodies = [body]
    world.occupancy[4, 4] = 1

    r = ArcadeRenderer(world, visible=False)
    bright = r.draw(throttle=False).astype(np.int64).sum()
    body.energy = 1.0
    dim = r.draw(throttle=False).astype(np.int64).sum()
    r.close()

    assert bright > dim, "яркость тела не зависит от энергии"


def test_gym_rgb_array_mode():
    env = gym.make(gym_id("forage"), size=12, seed=1, render_mode="rgb_array")
    obs, info = env.reset(seed=1)
    rng = np.random.default_rng(0)
    minds = {i: GreedyMind(4, rng) for i in info["ids"]}
    for _ in range(20):
        actions = [minds[i].act(o) for i, o in zip(info["ids"], obs)]
        obs, _, term, _, info = env.step(actions)
        for p, c in info["born"]:
            minds[c] = minds[p].spawn(rng)
        for i in info["died"]:
            minds.pop(i, None)
        if term:
            break
    frame = env.render()
    env.close()
    assert isinstance(frame, np.ndarray) and frame.ndim == 3


def test_close_is_idempotent():
    world = make("forage", size=8, seed=1)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    r = ArcadeRenderer(world, visible=False)
    r.draw(throttle=False)
    r.close()
    r.close()
