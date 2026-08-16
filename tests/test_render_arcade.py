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
from world.render_arcade import CELL, WALL, ArcadeRenderer  # noqa: E402
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


def test_fullscreen_renders_large_world():
    """Мир, который в обычное окно не влезает, должен рисоваться."""
    world = make("forage", size=200, seed=1, initial_population=40,
                 max_food=400, regrow_per_step=20.0)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    for _ in range(20):
        world.step()

    r = ArcadeRenderer(world, visible=False, fullscreen=True)
    frame = r.draw(throttle=False)
    view_w, view_h = r._viewport_cells()
    r.close()

    assert frame.std() > 1.0
    assert 0 < view_w <= 200 and 0 < view_h <= 200
    # окно определяется экраном, а не миром: 200 клеток по 28px не влезли бы
    assert frame.shape[1] < 200 * 28


def test_camera_pans_and_wraps():
    world = make("forage", size=60, seed=1, initial_population=10)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    r = ArcadeRenderer(world, visible=False, fullscreen=True, cell=16)
    try:
        view_w, view_h = r._viewport_cells()
        r.camera.x, r.camera.y = 10.0, 10.0
        r.draw(throttle=False)
        assert r._origin() == (10, 10)

        # на торе камера уезжает по кругу, а не упирается
        r.camera.x = 61.0
        r.camera.clamp_to(world, view_w, view_h)
        assert 0 <= r.camera.x < 60
    finally:
        r.close()


def test_camera_clamps_at_walls():
    world = make("forage", size=60, seed=1, boundary="clamp", initial_population=5)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    r = ArcadeRenderer(world, visible=False, fullscreen=True, cell=16)
    try:
        view_w, view_h = r._viewport_cells()
        r.camera.x, r.camera.y = 500.0, -50.0
        r.camera.clamp_to(world, view_w, view_h)
        assert 0 <= r.camera.x <= max(0, 60 - view_w)
        assert r.camera.y == 0.0
    finally:
        r.close()


def test_zoom_changes_visible_area():
    world = make("forage", size=120, seed=1, initial_population=10)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    r = ArcadeRenderer(world, visible=False, fullscreen=True, cell=8)
    try:
        wide = r._viewport_cells()
        r.camera.cell = 32
        narrow = r._viewport_cells()
        assert narrow[0] < wide[0] and narrow[1] < wide[1]
    finally:
        r.close()


def test_camera_moves_the_picture():
    """Сдвиг камеры обязан менять кадр — иначе панорама только на бумаге."""
    world = make("patches", size=80, seed=2, initial_population=40,
                 max_food=300, regrow_per_step=10.0)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    for _ in range(60):
        world.step()

    r = ArcadeRenderer(world, visible=False, fullscreen=True, cell=20)
    try:
        r.camera.x, r.camera.y = 0.0, 0.0
        here = r.draw(throttle=False).copy()
        r.camera.x, r.camera.y = 40.0, 40.0
        there = r.draw(throttle=False).copy()
        assert not np.array_equal(here, there)
    finally:
        r.close()


def test_follow_tracks_a_body():
    world = make("forage", size=80, seed=1, initial_population=20)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    r = ArcadeRenderer(world, visible=False, fullscreen=True, cell=20)
    try:
        target = world.bodies[0]
        r.camera.follow_id = target.id
        for _ in range(15):
            world.step()
            if not world.bodies:
                break
            r.draw(throttle=False)

        alive = {b.id: b for b in world.bodies}
        if target.id in alive:
            view_w, view_h = r._viewport_cells()
            b = alive[target.id]
            size = world.cfg.size
            # Экранная позиция, а не сырая разность: мир это тор, и тело у
            # края обёрнуто — камера при этом стоит правильно.
            x0, y0 = r._origin()
            sx, sy = (b.x - x0) % size, (b.y - y0) % size
            assert abs(sx - view_w / 2) <= 1.5, f"по горизонтали: {sx}"
            assert abs(sy - view_h / 2) <= 1.5, f"по вертикали: {sy}"
    finally:
        r.close()


def test_follow_survives_target_death():
    """Если тело, за которым следим, умерло, камера не должна зависнуть."""
    world = make("forage", size=40, seed=1, initial_population=6)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    r = ArcadeRenderer(world, visible=False, fullscreen=True, cell=16)
    try:
        r.camera.follow_id = 999999  # такого тела нет
        r.draw(throttle=False)
        assert r.camera.follow_id in [b.id for b in world.bodies] + [None]
    finally:
        r.close()


def test_windowed_mode_unaffected_by_camera():
    """Оконный режим должен остаться прежним: всё поле видно целиком."""
    world = make("forage", size=12, seed=1)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    r = ArcadeRenderer(world, visible=False, fullscreen=False)
    try:
        assert r._viewport_cells() == (12, 12)
        frame = r.draw(throttle=False)
        assert frame.shape[1] == 12 * CELL + 2 * WALL
    finally:
        r.close()


def test_gym_fullscreen_mode():
    env = gym.make(gym_id("forage"), size=100, seed=1, initial_population=20,
                   max_food=300, regrow_per_step=10.0, render_mode="fullscreen")
    obs, info = env.reset(seed=1)
    rng = np.random.default_rng(0)
    minds = {i: GreedyMind(4, rng) for i in info["ids"]}
    env.unwrapped.metadata["render_fps"] = 1000
    for _ in range(10):
        actions = [minds[i].act(o) for i, o in zip(info["ids"], obs)]
        obs, _, term, _, info = env.step(actions)
        for p, c in info["born"]:
            minds[c] = minds[p].spawn(rng)
        for i in info["died"]:
            minds.pop(i, None)
        if term:
            break
    assert env.unwrapped._renderer is not None
    assert env.unwrapped._renderer.fullscreen
    assert not env.unwrapped.closed_by_user
    env.close()


def test_close_is_idempotent():
    world = make("forage", size=8, seed=1)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    r = ArcadeRenderer(world, visible=False)
    r.draw(throttle=False)
    r.close()
    r.close()
