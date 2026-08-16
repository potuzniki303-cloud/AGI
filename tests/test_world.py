"""Тесты мира.

Половина из них проверяет механику, половина — те инварианты, ради которых
всё затевалось. Второе важнее: механическую поломку видно сразу, а вот мир,
в котором незаметно завёлся отбор по функции приспособленности, снаружи
выглядит совершенно нормально.

Отдельно стоит test_two_foods_is_solvable. Мир, который убивает всех,
неотличим от мира, который просто сложный, если не проверить, что
компетентное поведение в нём выживает. Один такой баг здесь уже был:
общая квота еды на все типы приводила к тому, что несъедобный тип
накапливался и съедобный переставал отрастать.

    python3 -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from world import ForageWorld, make, simulate  # noqa: E402
from world.core import Body, WorldConfig  # noqa: E402
from world.sensors import AntennaSensor, Interoception, PatchSensor  # noqa: E402
from world.stubs import GreedyMind, RandomMind  # noqa: E402

ALL_WORLDS = [
    "forage", "seasons", "patches", "shift", "two_foods", "bounty", "growing",
]


def random_factory(world):
    return lambda rng: RandomMind(world.action_size, rng)


def greedy_factory(world):
    return lambda rng: GreedyMind(world.action_size, rng)


# ----------------------------------------------------------------- базовое

@pytest.mark.parametrize("name", ALL_WORLDS)
def test_world_runs(name):
    w = make(name, seed=1)
    h = simulate(w, random_factory(w), steps=200)
    assert h.ticks > 0
    assert len(h.population) == h.ticks


@pytest.mark.parametrize("name", ALL_WORLDS)
def test_observation_size_matches_sensor(name):
    w = make(name, seed=1)
    w.reset(random_factory(w))
    obs = w.observe(w.bodies[0])
    assert obs.shape == (w.observation_size,)
    assert obs.dtype == np.float32


@pytest.mark.parametrize("name", ALL_WORLDS)
def test_deterministic_given_seed(name):
    def run():
        w = make(name, seed=7)
        h = simulate(w, random_factory(w), steps=150)
        return h.population, h.food

    assert run() == run()


def test_unknown_world_raises():
    with pytest.raises(KeyError):
        make("no_such_world")


def test_unknown_config_key_raises():
    # Опечатка в физике должна падать, а не тихо менять эксперимент.
    with pytest.raises(TypeError):
        make("forage", regrow_per_stepp=3.0)


# -------------------------------------------------------------- метаболизм

def test_energy_drains_and_kills():
    w = make("forage", seed=1, initial_population=4, initial_food_density=0.0,
             max_food=0, regrow_per_step=0.0, energy_at_birth=5.0)
    w.reset(random_factory(w))
    for _ in range(10):
        w.step()
    assert w.extinct
    assert not w.bodies


def test_eating_adds_energy():
    w = make("forage", seed=1, initial_population=1, initial_food_density=0.0,
             max_food=0, regrow_per_step=0.0)
    w.reset(lambda rng: RandomMind(w.action_size, rng))
    body = w.bodies[0]
    before = body.energy
    # кладём еду точно туда, куда тело шагнёт на север
    target_y = (body.y - 1) % w.cfg.size
    w.food[target_y, body.x] = 1

    class North:
        def act(self, obs): return 0
        def spawn(self, rng): return North()

    body.mind = North()
    w.step()
    assert body.energy == pytest.approx(
        before - w.cfg.cost_per_step + w.cfg.food_energy
    )
    assert w.food[target_y, body.x] == 0


def test_energy_capped():
    w = make("forage", seed=1)
    w.reset(random_factory(w))
    for _ in range(300):
        w.step()
        for b in w.bodies:
            assert b.energy <= w.cfg.energy_max + 1e-6


# ------------------------------------------------------------ размножение

def test_reproduction_needs_threshold_and_uses_spawn():
    spawned = []

    class Mind:
        def act(self, obs): return 0
        def spawn(self, rng):
            child = Mind()
            spawned.append(child)
            return child

    w = make("forage", seed=1, initial_population=1, initial_food_density=0.0,
             max_food=0, regrow_per_step=0.0)
    w.reset(lambda rng: Mind())
    body = w.bodies[0]

    body.energy = w.cfg.repro_threshold - 1.0
    w.step()
    assert len(w.bodies) == 1, "деление ниже порога"

    body.energy = w.cfg.repro_threshold + w.cfg.cost_per_step
    w.step()
    assert len(w.bodies) == 2
    assert len(spawned) == 1
    assert w.bodies[1].mind is spawned[0]
    assert w.bodies[1].parent_id == body.id
    assert w.bodies[1].generation == body.generation + 1


def test_reproduction_splits_energy():
    class Mind:
        def act(self, obs): return 0
        def spawn(self, rng): return Mind()

    w = make("forage", seed=1, initial_population=1, initial_food_density=0.0,
             max_food=0, regrow_per_step=0.0)
    w.reset(lambda rng: Mind())
    body = w.bodies[0]
    body.energy = w.cfg.repro_threshold + w.cfg.cost_per_step
    total_before = w.cfg.repro_threshold
    w.step()
    assert sum(b.energy for b in w.bodies) == pytest.approx(total_before)


def test_population_cap_blocks_birth_without_killing():
    w = make("forage", seed=1, initial_population=6, max_population=6)
    w.reset(random_factory(w))
    for b in w.bodies:
        b.energy = w.cfg.repro_threshold + 10.0
    before = len(w.bodies)
    w.step()
    # Никого не вытеснили ради новорождённого — просто места не нашлось.
    assert len(w.bodies) == before


# --------------------------------------------------------- бухгалтерия мира

@pytest.mark.parametrize("name", ALL_WORLDS)
def test_occupancy_matches_bodies(name):
    w = make(name, seed=3)
    w.reset(random_factory(w))
    for _ in range(200):
        w.step()
        if w.extinct:
            break
        grid = np.zeros_like(w.occupancy)
        for b in w.bodies:
            grid[b.y, b.x] += 1
        assert np.array_equal(grid, w.occupancy)


@pytest.mark.parametrize("name", ALL_WORLDS)
def test_bodies_stay_in_bounds(name):
    w = make(name, seed=4)
    w.reset(random_factory(w))
    for _ in range(200):
        w.step()
        for b in w.bodies:
            assert 0 <= b.x < w.cfg.size
            assert 0 <= b.y < w.cfg.size


def test_clamp_boundary_holds():
    w = make("forage", seed=1, boundary="clamp", size=6)
    w.reset(random_factory(w))
    for _ in range(300):
        w.step()
        for b in w.bodies:
            assert 0 <= b.x < 6 and 0 <= b.y < 6


def test_per_type_food_cap():
    """Регрессия: общая квота на все типы заклинивала мир.

    Несъедобный тип копился, занимал всю квоту, и съедобный переставал
    отрастать насовсем."""
    w = make("two_foods", seed=1, max_food=20)
    w.reset(random_factory(w))
    for _ in range(400):
        w.step()
        assert int((w.food == 1).sum()) <= 10
        assert int((w.food == 2).sum()) <= 10


# ------------------------------------------------------------------ сенсоры

def _lone_body(w, x, y):
    return Body(id=0, x=x, y=y, energy=10.0, mind=None)


def test_antenna_points_at_food():
    """Сенсор d должен отвечать за то же направление, что действие d.

    Если это выравнивание сломать, связь «вижу там — иду туда» перестанет
    быть одной ассоциацией, и локальному правилу будет не за что зацепиться.
    Порядок действий: 0 север, 1 юг, 2 запад, 3 восток; y растёт вниз.
    """
    w = ForageWorld(WorldConfig(size=11, seed=1, initial_population=0,
                                initial_food_density=0.0, max_food=0))
    w.food[:] = 0
    sensor = AntennaSensor("food", n_dirs=4)
    body = _lone_body(w, 5, 5)

    for action, (fx, fy) in enumerate([(5, 2), (5, 8), (2, 5), (8, 5)]):
        w.food[:] = 0
        w.food[fy, fx] = 1
        out = sensor.sense(w, body)
        assert int(np.argmax(out)) == action, f"направление {action}: {out}"


def test_antenna_grows_when_approaching():
    """Ключевое свойство: приближение усиливает сенсор этого направления.

    Именно на этой асимметрии держится вся идея — правильная пара
    сенсор/действие коррелирована сильнее неправильной, и разницу создаёт
    геометрия мира, а не оценка снаружи."""
    w = ForageWorld(WorldConfig(size=11, seed=1, initial_population=0,
                                initial_food_density=0.0, max_food=0))
    w.food[:] = 0
    w.food[1, 5] = 1  # еда на севере
    sensor = AntennaSensor("food", n_dirs=4)

    values = [sensor.sense(w, _lone_body(w, 5, y))[0] for y in (8, 6, 4, 2)]
    assert values == sorted(values), f"сенсор не растёт при приближении: {values}"


def test_antenna_max_range():
    w = ForageWorld(WorldConfig(size=21, seed=1, initial_population=0,
                                initial_food_density=0.0, max_food=0))
    w.food[:] = 0
    w.food[2, 10] = 1  # 8 клеток на север от центра
    near = AntennaSensor("food", n_dirs=4, max_range=3.0)
    far = AntennaSensor("food", n_dirs=4, max_range=20.0)
    body = _lone_body(w, 10, 10)
    assert near.sense(w, body).sum() == 0.0
    assert far.sense(w, body).sum() > 0.0


def test_antenna_filters_food_type():
    w = make("two_foods", seed=1)
    w.food[:] = 0
    w.food[2, 5] = 2
    body = _lone_body(w, 5, 5)
    a1 = AntennaSensor("food", n_dirs=4, food_type=1).sense(w, body)
    a2 = AntennaSensor("food", n_dirs=4, food_type=2).sense(w, body)
    assert a1.sum() == 0.0
    assert a2[0] > 0.0


def test_patch_sensor_shape_and_self_exclusion():
    w = make("forage", seed=1, initial_population=3)
    w.reset(random_factory(w))
    sensor = PatchSensor(radius=2, channels=("food", "agents"))
    assert sensor.size == 5 * 5 * 2
    body = w.bodies[0]
    out = sensor.sense(w, body)
    assert out.shape == (sensor.size,)
    # своё тело в центре канала агентов не отображается
    centre = 25 + (2 * 5 + 2)
    others = sum(1 for b in w.bodies[1:] if (b.y, b.x) == (body.y, body.x))
    assert out[centre] == pytest.approx(float(others))


def test_interoception_reports_energy():
    w = make("forage", seed=1, initial_population=1)
    w.reset(random_factory(w))
    body = w.bodies[0]
    body.energy = w.cfg.energy_max / 2.0
    out = Interoception(("energy",)).sense(w, body)
    assert out[0] == pytest.approx(0.5)


def test_sensor_composition():
    s = AntennaSensor("food", 4) + Interoception(("energy",))
    assert s.size == 5
    s2 = s + AntennaSensor("agents", 8)
    assert s2.size == 13
    assert len(s2.parts) == 3  # склейка не вкладывается рекурсивно


# ------------------------------------------------------- границы дизайнера

@pytest.mark.parametrize("name", ALL_WORLDS)
def test_world_only_calls_act_and_spawn(name):
    """Мир не должен трогать ничего, кроме двух методов контракта.

    Если мир полезет в веса или в геном, он перестанет быть средой и
    станет учителем."""
    touched: set[str] = set()

    class Spy:
        def __init__(self, n): self.n = n
        def act(self, obs):
            touched.add("act")
            return 0
        def spawn(self, rng):
            touched.add("spawn")
            return Spy(self.n)
        def __getattr__(self, item):
            touched.add(item)
            raise AttributeError(item)

    w = make(name, seed=2)
    simulate(w, lambda rng: Spy(w.action_size), steps=150)
    assert touched <= {"act", "spawn"}, f"мир полез в {touched - {'act', 'spawn'}}"


def test_step_report_carries_no_per_agent_score():
    """В сводке не должно быть ничего, что ранжирует отдельные тела."""
    w = make("forage", seed=1)
    w.reset(random_factory(w))
    report = w.step()
    for field in vars(report):
        assert "reward" not in field
        assert "fitness" not in field
        assert "score" not in field


def test_observations_are_simultaneous():
    """Все тела воспринимают мир до того, как хоть кто-то сдвинулся.

    Иначе порядок в списке давал бы преимущество — то есть в мир протёк бы
    отбор по позиции в массиве."""
    w = make("forage", seed=1, initial_population=8)
    seen: list[int] = []

    class Mind:
        def __init__(self, w): self.w = w
        def act(self, obs):
            seen.append(int(np.count_nonzero(self.w.food)))
            return 0
        def spawn(self, rng): return Mind(self.w)

    w.reset(lambda rng: Mind(w))
    w.step()
    assert len(set(seen)) == 1, "мир менялся между восприятиями тел"


# -------------------------------------------------------------- решаемость

def test_forage_rewards_competence():
    """Компетентное поведение должно выигрывать у случайного.

    Если разрыва нет, в мире нет давления, и эволюции не за что зацепиться."""
    wr = make("forage", seed=5)
    hr = simulate(wr, random_factory(wr), steps=1500)
    wg = make("forage", seed=5)
    hg = simulate(wg, greedy_factory(wg), steps=1500)

    assert hr.extinct_at is not None, "случайные ходоки не должны выживать"
    assert hg.extinct_at is None, "мир должен быть проходим"
    assert hg.max_generation >= 5, "поколения должны сменяться"


def test_two_foods_is_solvable_only_by_discriminating():
    """Различающий тип выживает, неразличающий — нет.

    Ровно этот разрыв и есть смысл мира: ответ в геном не помещается,
    потому что при каждом рождении он разыгрывается заново."""

    class Oracle:  # знает ответ; проверка проходимости, а не агент
        def __init__(self, rng): self.rng = rng
        def act(self, obs):
            a = obs[0:4]
            return int(np.argmax(a)) if np.any(a > 0) else int(self.rng.integers(4))
        def spawn(self, rng): return Oracle(rng)

    class Blind:  # идёт к любой еде, тип игнорирует
        def __init__(self, rng): self.rng = rng
        def act(self, obs):
            a = np.maximum(obs[0:4], obs[4:8])
            return int(np.argmax(a)) if np.any(a > 0) else int(self.rng.integers(4))
        def spawn(self, rng): return Blind(rng)

    kw = dict(seed=3, polarity_scope="world", polarity_period=0)
    ho = simulate(make("two_foods", **kw), lambda rng: Oracle(rng), steps=1500)
    hb = simulate(make("two_foods", **kw), lambda rng: Blind(rng), steps=1500)

    assert ho.extinct_at is None, "различающий тип должен выживать"
    assert hb.extinct_at is not None, "неразличающий не должен"


def test_shift_actually_breaks_bodies():
    w = make("shift", seed=1, shift_at_age=10, shift_jitter=0, shift_mode="motor")
    w.reset(greedy_factory(w))
    for _ in range(30):
        w.step()
        if w.extinct:
            break
    shifted = [b for b in w.bodies if b.traits.get("shifted")]
    assert shifted, "ни одно тело не пострадало"
    for b in shifted:
        assert b.action_map is not None
        assert not np.array_equal(b.action_map, np.arange(w.action_size)), \
            "тождественная перестановка ничего не ломает"


def test_shift_shortens_life():
    hs = simulate(make("shift", seed=5), greedy_factory(make("shift")), steps=1500)
    hf = simulate(make("forage", seed=5), greedy_factory(make("forage")), steps=1500)
    assert np.percentile(hs.lifespans, 90) < np.percentile(hf.lifespans, 90)


def test_bounty_fades_to_normal():
    """Изобилие должно спадать ровно до базового потолка и там остаться."""
    w = make("bounty", seed=1, bounty_factor=10.0, bounty_ticks=300, max_food=16)
    w.reset(random_factory(w))
    start_cap = w._food_cap(1)
    assert start_cap == pytest.approx(160, abs=1)

    caps = []
    for _ in range(500):
        w.step()
        caps.append(w._food_cap(1))

    assert caps[0] > caps[150] > caps[299], "потолок должен падать"
    assert caps[-1] == 16, "после bounty_ticks должен стоять базовый потолок"
    assert min(caps) == 16, "потолок не должен уходить ниже базового"


def test_bounty_trims_excess_food():
    """Когда потолок опускается, лишняя еда обязана исчезать сама.

    Иначе спад почти не чувствуется: потолок ограничивает только прирост,
    а уже выросшая еда лежала бы на поле, пока её кто-нибудь не съест.

    Механизм проверяется напрямую, без тел: пустой мир вымирает на первом
    же такте, а после вымирания step() выходит сразу и хуки не работают."""
    w = make("bounty", seed=1, bounty_factor=10.0, bounty_ticks=200, max_food=16)
    w.reset(random_factory(w))

    w.food[:] = 0
    w._scatter_food(160, food_type=1)
    assert int(np.count_nonzero(w.food)) > 100, "в начале должно быть изобильно"

    w.tick = 1000  # изобилие давно кончилось
    w._on_tick()
    assert int(np.count_nonzero(w.food)) == 16, "излишек не убран"


def test_bounty_food_never_exceeds_current_cap():
    """Инвариант на живом прогоне: запас никогда не выше текущего потолка."""
    w = make("bounty", seed=2, bounty_factor=10.0, bounty_ticks=400, max_food=16)
    w.reset(greedy_factory(w))
    for _ in range(700):
        w.step()
        if w.extinct:
            break
        assert int(np.count_nonzero(w.food)) <= w._food_cap(1)


def test_bounty_with_zero_ticks_is_plain_forage():
    w = make("bounty", seed=1, bounty_ticks=0, max_food=16)
    assert w._food_cap(1) == 16
    assert w._regrow_rate() == pytest.approx(w.cfg.regrow_per_step)


def test_growing_reaches_final_size():
    w = make("growing", seed=1, start_size=6, size=12, growth_every=50)
    w.reset(random_factory(w))
    assert w.cfg.size == 6
    assert w.food.shape == (6, 6)

    sizes = []
    for _ in range(600):
        w.step()
        sizes.append(w.cfg.size)
        # массивы обязаны идти в ногу со стороной
        assert w.food.shape == (w.cfg.size, w.cfg.size)
        assert w.occupancy.shape == (w.cfg.size, w.cfg.size)

    assert sizes[-1] == 12, "мир должен дорасти до size"
    assert sizes == sorted(sizes), "сторона не должна уменьшаться по ходу"
    assert w.growth_left == 0


def test_growing_keeps_food_and_bodies_on_resize():
    """Расширение не должно терять ни еду, ни тела."""
    w = make("growing", seed=1, start_size=6, size=10, growth_every=10**9)
    w.reset(random_factory(w))
    w.food[:] = 0
    w.food[1, 1] = 1
    w.food[4, 5 % 6] = 1
    before_food = int(np.count_nonzero(w.food))
    before_ids = [b.id for b in w.bodies]
    before_pos = [(b.y, b.x) for b in w.bodies]

    w._resize(9)

    assert w.food.shape == (9, 9)
    assert int(np.count_nonzero(w.food)) == before_food
    assert w.food[1, 1] == 1
    assert [b.id for b in w.bodies] == before_ids
    assert [(b.y, b.x) for b in w.bodies] == before_pos, "тела не должны сдвигаться"


def test_growing_resets_back_to_start_size():
    """Второй прогон обязан начинаться с маленького мира, а не с выросшего."""
    w = make("growing", seed=1, start_size=6, size=12, growth_every=50)
    w.reset(random_factory(w))
    for _ in range(600):
        w.step()
    assert w.cfg.size == 12

    w.reset(random_factory(w))
    assert w.cfg.size == 6
    assert w.food.shape == (6, 6)
    assert w.occupancy.shape == (6, 6)


def test_growing_rejects_bad_sizes():
    with pytest.raises(ValueError, match="start_size"):
        make("growing", start_size=30, size=10)
    with pytest.raises(ValueError, match="start_size"):
        make("growing", start_size=0, size=10)


def test_growing_distance_to_food_increases():
    """Суть мира: путь до еды растёт вместе со стороной."""
    def mean_distance(world):
        ys, xs = np.nonzero(world.food)
        if ys.size == 0 or not world.bodies:
            return None
        out = []
        for b in world.bodies:
            dx, dy = world.displacement(b.x, b.y, xs, ys)
            out.append((np.abs(dx) + np.abs(dy)).min())
        return float(np.mean(out))

    w = make("growing", seed=1, start_size=8, size=24, growth_every=100)
    w.reset(greedy_factory(w))
    early, late = [], []
    for t in range(1700):
        w.step()
        d = mean_distance(w)
        if d is None:
            continue
        if w.cfg.size <= 10:
            early.append(d)
        elif w.cfg.size >= 22:
            late.append(d)

    assert early and late
    assert np.mean(late) > np.mean(early), "в большом мире идти должно дальше"


def test_seasons_modulate_regrowth():
    w = make("seasons", seed=1, season_period=100, season_amplitude=0.9)
    rates = []
    w.reset(random_factory(w))
    for _ in range(100):
        rates.append(w._regrow_rate())
        w.step()
    assert max(rates) > min(rates) * 3


def test_patches_are_clustered():
    """Еда в PatchesWorld должна лежать кучнее, чем при равномерном посеве."""
    def mean_nearest(world):
        ys, xs = np.nonzero(world.food)
        pts = np.stack([ys, xs], axis=1).astype(float)
        d = np.abs(pts[:, None, :] - pts[None, :, :]).sum(-1)
        np.fill_diagonal(d, np.inf)
        return float(d.min(axis=1).mean())

    wp = make("patches", seed=1)
    wp.reset(random_factory(wp))
    wf = make("forage", seed=1, max_food=18, initial_food_density=0.031)
    wf.reset(random_factory(wf))
    assert mean_nearest(wp) < mean_nearest(wf)


# ------------------------------------------------------------- конфигурация

def test_config_validation():
    with pytest.raises(ValueError):
        make("forage", size=0)
    with pytest.raises(ValueError):
        make("forage", boundary="donut")
    with pytest.raises(ValueError):
        make("forage", energy_at_birth=100.0, repro_threshold=50.0)


def test_bad_action_is_rejected():
    class Bad:
        def act(self, obs): return 99
        def spawn(self, rng): return Bad()

    w = make("forage", seed=1)
    w.reset(lambda rng: Bad())
    with pytest.raises(ValueError, match="expected"):
        w.step()


def test_allow_stay_adds_action():
    assert make("forage", allow_stay=False).action_size == 4
    assert make("forage", allow_stay=True).action_size == 5


# ----------------------------------------------------------- переходник

class FakeAgent:
    """Имитирует интерфейс src/agent: forward() -> логиты, mutate() -> потомок.

    Нужен потому, что torch в этом контейнере не ставится, а проверить
    перевод forward/mutate -> act/spawn всё равно надо."""

    def __init__(self, logits, generation=0):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.generation = generation
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.logits

    def mutate(self):
        return FakeAgent(self.logits, self.generation + 1)


class FakeTensor:
    """Утиный torch.Tensor: detach/cpu/numpy."""

    def __init__(self, arr): self._arr = np.asarray(arr)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self._arr


def test_adapter_argmax():
    from world.adapters import MindAdapter

    m = MindAdapter(FakeAgent([0.1, 0.9, 0.2, 0.3]), 4,
                    np.random.default_rng(0), temperature=0.0)
    assert m.act(np.zeros(5, dtype=np.float32)) == 1


def test_adapter_samples_within_range():
    from world.adapters import MindAdapter

    m = MindAdapter(FakeAgent([0.0, 0.0, 0.0, 0.0]), 4,
                    np.random.default_rng(0), temperature=1.0)
    acts = {m.act(np.zeros(5, dtype=np.float32)) for _ in range(200)}
    assert acts <= {0, 1, 2, 3}
    assert len(acts) > 1, "при равных логитах выбор должен быть случайным"


def test_adapter_uses_only_first_n_logits():
    from world.adapters import MindAdapter

    # хвост логитов — скрытое состояние, действием он быть не должен
    m = MindAdapter(FakeAgent([0.1, 0.2, 0.3, 0.4, 9.9, 9.9]), 4,
                    np.random.default_rng(0), temperature=0.0)
    assert m.act(np.zeros(5, dtype=np.float32)) == 3


def test_adapter_spawn_calls_mutate():
    from world.adapters import MindAdapter

    m = MindAdapter(FakeAgent([1.0, 0, 0, 0]), 4, np.random.default_rng(0))
    child = m.spawn(np.random.default_rng(1))
    assert child.agent.generation == 1
    assert child.agent is not m.agent
    assert child.temperature == m.temperature


def test_adapter_unwraps_tensor_like():
    from world.adapters import MindAdapter

    class TensorAgent(FakeAgent):
        def forward(self, x): return FakeTensor(self.logits)

    m = MindAdapter(TensorAgent([0.0, 0.0, 5.0, 0.0]), 4,
                    np.random.default_rng(0), temperature=0.0)
    assert m.act(np.zeros(5, dtype=np.float32)) == 2


def test_adapter_rejects_non_finite_logits():
    """Разошедшиеся веса должны падать, а не превращаться в действие 0.

    Иначе получится агент, который «умеет ходить на север», хотя он сломан."""
    from world.adapters import MindAdapter

    m = MindAdapter(FakeAgent([np.nan, 0, 0, 0]), 4, np.random.default_rng(0))
    with pytest.raises(FloatingPointError, match="нечисловые"):
        m.act(np.zeros(5, dtype=np.float32))


def test_adapter_runs_in_world():
    from world.adapters import MindAdapter

    w = make("forage", seed=1)
    h = simulate(
        w,
        lambda rng: MindAdapter(
            FakeAgent(np.zeros(w.action_size)), w.action_size, rng, temperature=1.0
        ),
        steps=200,
    )
    assert h.ticks > 0
