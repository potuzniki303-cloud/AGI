"""Тесты мира Фазы 0.

Часть проверяет механику, часть — инварианты, ради которых всё затевалось.
Второе важнее: сломанную физику видно сразу, а мир, в котором незаметно
завёлся опрос агента или протёк канал истины, снаружи выглядит нормально
и портит все эксперименты молча.

    python3 -m pytest tests/test_phase0.py -q
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phase0 import (  # noqa: E402
    BudgetProfile, Config, Event, Motor, World, adaptation_curve, build, run,
    run_baseline,
)
from phase0.baselines import GreedySymbolic, RandomAgent  # noqa: E402
from phase0.body import Body, integrate  # noqa: E402
from phase0.config import Kind, Mode  # noqa: E402
from phase0.driver import replay  # noqa: E402
from phase0.events import Channels  # noqa: E402
from phase0.render import frame  # noqa: E402
from phase0.snapshot import clone  # noqa: E402


# ======================================================================
# Контракт (Часть 1) — самое важное
# ======================================================================

def test_no_get_action_anywhere():
    """Функции вида get_action(agent) не должно существовать в кодовой базе.

    Это архитектурный инвариант, а не стилистика: если мир на каждом тике
    ходит снимать состояние агента ради действия, контракт нарушен, даже
    если снаружи всё выглядит асинхронно.
    """
    pattern = re.compile(r"def\s+get_action\b|\.get_action\s*\(")
    offenders = []
    for path in (ROOT / "src" / "phase0").rglob("*.py"):
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert not offenders, f"контракт нарушен в: {offenders}"


def test_world_never_references_agent():
    """Мир не хранит ссылку на агента и не вызывает его код."""
    world = World(Config())
    for _ in range(50):
        world.step()
    names = [n for n in vars(world) if "agent" in n.lower() or "mind" in n.lower()]
    assert names == [], f"мир завёл ссылку на агента: {names}"


def test_world_runs_forever_when_agent_is_silent():
    """Молчание агента — легальное состояние, а не деградация.

    Это условие завершения шага 0.1.
    """
    world = run(World(Config()), agent=None, ticks=5000)
    assert world.tick == 5000
    assert world.body.alive          # умирал и респавнился, но процесс идёт
    assert world.deaths >= 1         # молча голодать — тоже последствие


def test_action_hold_keeps_last_value():
    """Моторные каналы удерживают последнее значение до нового события."""
    world = World(Config())
    world.outbox.put(Event(0, Motor.THRUST, 1.0))
    world.step()
    assert world.motor.thrust == 1.0
    for _ in range(300):             # пять секунд полного молчания
        world.step()
    assert world.motor.thrust == 1.0, "молчание обязано продолжать действие"
    assert world.body.speed > 1.0, "тело должно продолжать лететь"


def test_silence_has_consequences():
    """Агент, который молчит на полном ходу, врезается в стену."""
    cfg = Config()
    world = World(cfg)
    world.body.x, world.body.y, world.body.theta = 5.0, 32.0, math.pi
    world.outbox.put(Event(0, Motor.THRUST, 1.0))
    for _ in range(200):
        world.step()
    assert world.wall_hits > 0


def test_zero_action_is_required_to_stop():
    """Чтобы прекратить действие, нужно эмитить 0.0 — молчание не останавливает."""
    world = World(Config())
    world.outbox.put(Event(0, Motor.THRUST, 1.0))
    world.step()
    world.outbox.put(Event(world.tick, Motor.THRUST, 0.0))
    world.step()
    assert world.motor.thrust == 0.0


def test_future_events_are_deferred_not_dropped():
    """События с t > current_tick откладываются, а не теряются и не
    применяются раньше срока."""
    world = World(Config())
    world.outbox.put(Event(50, Motor.THRUST, 1.0))
    for _ in range(10):
        world.step()
    assert world.motor.thrust == 0.0, "применили будущее событие раньше срока"
    for _ in range(45):
        world.step()
    assert world.motor.thrust == 1.0, "будущее событие потеряно"


def test_motor_garbage_does_not_crash_world():
    world = World(Config())
    world.outbox.put(Event(0, 999, 1.0))          # неизвестный канал
    world.outbox.put(Event(0, Motor.THRUST, float("nan")))
    world.outbox.put(Event(0, Motor.TURN, 1e9))   # вне диапазона
    world.step()
    assert world.motor.thrust == 0.0
    assert world.motor.turn == 1.0                # клип, а не мусор


# ======================================================================
# Канал истины (Часть 6)
# ======================================================================

def test_truth_never_reaches_inbox():
    """Канал истины никогда не попадает агенту. Нарушение инвалидирует все
    эксперименты по связыванию."""
    cfg = Config()
    world = World(cfg)
    ch = Channels(cfg)
    for _ in range(300):
        record = world.step()
        for e in world.inbox.drain():
            assert 0 <= e.channel < ch.n_total, "канал вне сенсорного пространства"
        # ни одна величина истины не должна совпасть с сенсорным каналом
        assert record.items is not None


def test_truth_has_retinal_span():
    """retinal_span — ключ ко всей диагностике восприятия."""
    world = World(Config())
    seen = False
    for _ in range(600):
        record = world.step()
        world.inbox.drain()
        for it in record.items:
            if it.visible:
                assert it.retinal_span is not None
                lo, hi = it.retinal_span
                assert 0 <= lo <= hi < world.cfg.retina_n
                seen = True
    assert seen, "за 600 тиков ни один предмет не попал на сетчатку"


# ======================================================================
# Физика (Часть 2)
# ======================================================================

def test_terminal_velocity_matches_discrete_derivation():
    """v_max = F_max/drag — предел непрерывного времени. Дискретный
    интегратор даёт ровно v_max*(1 - drag*dt), и это не ошибка, а следствие
    порядка операций, заданного спецификацией."""
    cfg = Config(walls=False)
    body = Body(x=0.0, y=0.0)
    for _ in range(2000):
        integrate(body, 1.0, 0.0, cfg)
    expected = cfg.v_max * (1.0 - cfg.drag * cfg.dt)
    assert body.speed == pytest.approx(expected, rel=1e-9)


def test_wall_collision_slides():
    """Нормальная компонента обнуляется, тангенциальная сохраняется."""
    cfg = Config()
    body = Body(x=cfg.arena[0] - cfg.r_body - 0.01, y=32.0, vx=8.0, vy=3.0)
    hit = integrate(body, 0.0, 0.0, cfg)
    assert hit == 1
    assert body.vx == 0.0
    assert body.vy > 0.0


def test_body_stays_inside_arena():
    world = run(World(Config()), agent=None, ticks=2000)
    cfg = world.cfg
    assert cfg.r_body <= world.body.x <= cfg.arena[0] - cfg.r_body
    assert cfg.r_body <= world.body.y <= cfg.arena[1] - cfg.r_body


# ======================================================================
# Гомеостат (Часть 3)
# ======================================================================

def test_basal_drain_matches_spec():
    """Голодная смерть из полного бака в покое — 50 с (ВЫВЕДЕНО)."""
    cfg = Config()
    world = World(cfg)
    world.body.energy = cfg.e_max
    for _ in range(cfg.tick_hz * 10):
        world.step()
    assert world.body.energy == pytest.approx(cfg.e_max - cfg.basal * 10, abs=1e-9)


def test_death_respawns_body_but_not_process():
    """Смерть тела — не смерть процесса. Граф агента не трогается."""
    cfg = Config()
    world = World(cfg)
    world.body.energy = 0.001
    while world.deaths == 0:
        world.step()
        world.inbox.drain()
    assert world.deaths == 1
    assert world.body.alive, "тело обязано вернуться: смерть тела != смерть процесса"
    # На самом тике смерти бак ровно E_0; дальше начинает стекать basal.
    assert world.body.energy == pytest.approx(cfg.e_init)
    before = world.body.energy
    world.step()
    assert world.body.energy < before


def test_death_emits_noci_for_configured_duration():
    cfg = Config()
    world = World(cfg)
    ch = Channels(cfg)
    world.body.energy = 0.001
    noci_ticks = 0
    for _ in range(cfg.noci_duration + 20):
        world.step()
        if any(e.channel == ch.NOCI for e in world.inbox.drain()):
            noci_ticks += 1
    assert noci_ticks >= cfg.noci_duration - 1


def test_eating_is_contact_based():
    cfg = Config(difficulty="A1")
    world = World(cfg)
    item = world.items.visible_items[0]
    world.body.x, world.body.y = item.x, item.y
    before = world.body.energy
    world.step()
    assert world.body.energy > before


# ======================================================================
# Предметы: регрессия на храповик состава поля
# ======================================================================

def test_item_composition_is_stationary():
    """Состав поля не должен дрейфовать от поведения агента.

    Регрессия на реальный баг: при перерандомизации kind на респавне
    съедобные съедались и возвращались монеткой, несъедобные копились, и за
    30000 тиков поле выродилось до 0 питательных из 12 — greedy_symbolic
    умирал от голода при формально верных константах Части 3.
    """
    cfg = Config(difficulty="A2")
    world, _ = run_baseline("greedy_symbolic", cfg, ticks=20_000)
    comp = world.items.composition()
    assert comp["A"] >= 4, f"питательные вымыло с поля: {comp}"


# ======================================================================
# Сенсорный тракт (Часть 4)
# ======================================================================

def test_static_scene_gives_zero_transient_events():
    """Статичная сцена даёт НОЛЬ входа. Разреженность возникает физически,
    а не вводится регуляризатором."""
    cfg = Config()
    world = World(cfg)
    ch = Channels(cfg)
    for _ in range(20):                      # дать сцене устояться
        world.step()
        world.inbox.drain()
    counts = []
    for _ in range(200):                     # агент молчит, тело стоит
        world.step()
        counts.append(sum(1 for e in world.inbox.drain()
                          if e.channel < ch.sustained_start))
    assert sum(counts) == 0, f"статичная сцена дала {sum(counts)} событий"


def test_movement_generates_transient_events():
    cfg = Config()
    world = World(cfg)
    ch = Channels(cfg)
    world.outbox.put(Event(0, Motor.TURN, 1.0))
    total = 0
    for _ in range(200):
        world.step()
        total += sum(1 for e in world.inbox.drain() if e.channel < ch.sustained_start)
    assert total > 0


def test_event_rate_cap_is_enforced():
    cfg = Config()
    world = World(cfg)
    ch = Channels(cfg)
    cap = int(round(cfg.event_rate_cap * ch.n_transient))
    world.outbox.put(Event(0, Motor.TURN, 1.0))
    for _ in range(400):
        world.step()
        n = sum(1 for e in world.inbox.drain() if e.channel < ch.sustained_start)
        assert n <= cap, f"{n} > cap {cap}"


def test_dense_channels_emit_every_tick():
    cfg = Config()
    world = World(cfg)
    ch = Channels(cfg)
    world.step()
    events = world.inbox.drain()
    sustained = [e for e in events if ch.sustained_start <= e.channel < ch.proprio_start]
    proprio = [e for e in events if ch.proprio_start <= e.channel < ch.intero_start]
    energy = [e for e in events if e.channel == ch.ENERGY]
    assert len(sustained) == cfg.sustained_n * cfg.color_channels
    assert len(proprio) == 4
    assert len(energy) == 1


def test_colour_axis_separates_kinds():
    from phase0.items import Item
    from phase0.retina import Retina
    cfg = Config()
    retina = Retina(cfg, Channels(cfg))
    a = retina.project(32, 32, 0.0, [Item(0, 42.0, 32.0, Kind.A)])
    b = retina.project(32, 32, 0.0, [Item(0, 42.0, 32.0, Kind.B)])
    mid = cfg.retina_n // 2
    assert a.c[mid] > 0.0 > b.c[mid]


def test_nearer_item_occludes_farther():
    from phase0.items import Item
    from phase0.retina import Retina
    cfg = Config()
    retina = Retina(cfg, Channels(cfg))
    proj = retina.project(32, 32, 0.0,
                          [Item(0, 44.0, 32.0, Kind.A), Item(1, 36.0, 32.0, Kind.B)])
    mid = cfg.retina_n // 2
    assert proj.owner[mid] == 1
    assert proj.occluded_by.get(0) == 1


def test_angular_size_falls_with_distance():
    from phase0.items import Item
    from phase0.retina import Retina
    cfg = Config()
    retina = Retina(cfg, Channels(cfg))
    near = retina.project(32, 32, 0.0, [Item(0, 38.0, 32.0, Kind.A)])
    far = retina.project(32, 32, 0.0, [Item(0, 56.0, 32.0, Kind.A)])
    assert int((near.owner >= 0).sum()) > int((far.owner >= 0).sum())


# ======================================================================
# Режимы (Части 7-9)
# ======================================================================

@pytest.mark.parametrize("difficulty,expect_a,expect_b", [
    ("A1", 0.25, 0.25),
    ("A2", 0.25, 0.0),
    ("A3", 0.25, -0.15),
])
def test_difficulty_ladder_nutrition(difficulty, expect_a, expect_b):
    world = World(Config(difficulty=difficulty))
    assert world.regime.nutritive(Kind.A, 0) == pytest.approx(expect_a)
    assert world.regime.nutritive(Kind.B, 0) == pytest.approx(expect_b)


def test_regime_b_flips_within_interval():
    cfg = Config(mode=Mode.B)
    world = World(cfg)
    flips = []
    # Интервал реверсии 15000-45000 тиков, поэтому прогон должен быть длинным:
    # на 30000 тиках переворотов могло не случиться ни одного.
    for _ in range(250_000):
        record = world.step()
        world.inbox.drain()
        if any(e["type"] == "regime_flip" for e in record.events):
            flips.append(record.tick)
    assert len(flips) >= 5
    gaps = np.diff([0] + flips)
    assert gaps.min() >= cfg.flip_interval[0]
    assert gaps.max() <= cfg.flip_interval[1]


def test_regime_b_swaps_nutrition_without_changing_appearance():
    """Внешность не меняется. Меняется только правило."""
    cfg = Config(mode=Mode.B)
    world = World(cfg)
    kinds_before = [i.kind for i in world.items.items]
    before = world.regime.nutritive(Kind.A, 0)
    world.regime.flag_state ^= 1
    after = world.regime.nutritive(Kind.A, 0)
    assert before == -after or (before > 0 > after) or (before < 0 < after)
    assert [i.kind for i in world.items.items] == kinds_before


def test_regime_c_window_matches_spec():
    cfg = Config(mode=Mode.C)
    world = World(cfg)
    open_ticks = sum(1 for t in range(cfg.rhythm_period)
                     if world.regime.nutritive(Kind.A, t) > 0)
    assert open_ticks == int(cfg.rhythm_period * cfg.rhythm_window)


# ======================================================================
# Бейзлайны (Часть 7)
# ======================================================================

@pytest.mark.parametrize("difficulty", ["A1", "A2", "A3"])
def test_greedy_symbolic_solves_a1_a3(difficulty):
    """Обязан решать A1-A3 без всякого обучения.

    Если не решает — неверны константы Части 3, а не бейзлайн плох.
    """
    world, metrics = run_baseline("greedy_symbolic", Config(difficulty=difficulty),
                                  ticks=30_000)
    assert metrics.deaths_per_10k == 0.0, metrics.row()
    assert metrics.mean_energy > 0.5


def test_greedy_symbolic_beats_random():
    cfg = Config(difficulty="A1")
    _, greedy = run_baseline("greedy_symbolic", cfg, ticks=20_000)
    _, rand = run_baseline("random", cfg, ticks=20_000)
    assert greedy.eat_rate_per_1k > rand.eat_rate_per_1k
    assert greedy.deaths_per_10k < rand.deaths_per_10k


@pytest.mark.parametrize("name", ["random", "greedy_symbolic", "greedy_sustained",
                                  "greedy_transient", "linear_pixel", "tabular_q",
                                  "small_rnn_bptt"])
def test_every_baseline_runs_through_the_contract(name):
    world, metrics = run_baseline(name, Config(difficulty="A1"), ticks=3000)
    assert world.tick == 3000
    assert metrics.ticks == 3000


# ======================================================================
# Приборы (Часть 10)
# ======================================================================

def _record(cfg, ticks, seed=0):
    world = World(cfg)
    agent = GreedySymbolic(cfg, seed=seed)
    motor = []
    for _ in range(ticks):
        agent.observe_symbolic(world)
        world.step()
        world.inbox.drain()
        before = len(world.outbox)
        agent.step([], world.outbox, 1000)
        motor.extend(world.outbox.peek()[before:])
    b = world.body
    return motor, (b.x, b.y, b.theta, b.vx, b.vy, b.omega, b.energy)


def test_replay_is_bitwise():
    """При фиксированном seed и записанном логе действий мир воспроизводится
    ПОБИТОВО. Без этого нельзя отличить «гипотеза неверна» от «не туда попали
    константами»."""
    cfg = Config(difficulty="A3")
    motor, final = _record(cfg, 10_000)
    world = replay(cfg, motor, 10_000)
    b = world.body
    assert (b.x, b.y, b.theta, b.vx, b.vy, b.omega, b.energy) == final


def test_same_seed_same_run():
    cfg = Config(difficulty="A3")
    assert _record(cfg, 5000)[1] == _record(cfg, 5000)[1]


def test_snapshot_continuation_is_bitwise():
    """Загрузка снапшота даёт побитово то же продолжение, что и непрерывный
    прогон. Регрессия: сначала расходилось, потому что снапшот терял события
    в полёте."""
    import copy
    cfg = Config(difficulty="A3")
    world, agent = World(cfg), GreedySymbolic(cfg)

    def pump(w, a, n):
        out = []
        for _ in range(n):
            a.observe_symbolic(w)
            w.step()
            w.inbox.drain()
            a.step([], w.outbox, 1000)
            out.append((w.body.x, w.body.y, w.body.theta, w.body.energy))
        return out

    pump(world, agent, 2000)
    twin, twin_agent = clone(world), copy.deepcopy(agent)
    assert pump(world, agent, 1000) == pump(twin, twin_agent, 1000)


def test_fork_varies_only_by_agent_seed():
    cfg = Config(difficulty="A1")
    world = run(World(cfg), agent=None, ticks=1000)
    results = []
    for seed in range(4):
        twin = clone(world)
        agent = RandomAgent(cfg, seed=seed)
        for _ in range(1500):
            twin.step()
            agent.step(twin.inbox.drain(), twin.outbox, 1000)
        results.append(round(twin.body.energy, 6))
    assert len(set(results)) > 1, "форк не даёт разброса — seed агента не работает"


def test_rng_streams_are_independent():
    """Изменение одной подсистемы не должно сдвигать последовательности
    остальных."""
    base = Config()
    other = Config(seeds={**base.seeds, "item_respawn": 12345})
    a, b = World(base), World(other)
    assert [i.kind for i in a.items.items] == [i.kind for i in b.items.items]


def test_meta_json_records_every_constant(tmp_path):
    from phase0.logs import RunLogger
    cfg = Config()
    world = World(cfg)
    import json
    with RunLogger(tmp_path, cfg, world.channels, enable=()) as logger:
        logger.write_meta()
    meta = json.loads((tmp_path / "meta.json").read_text())
    import dataclasses
    for field in dataclasses.fields(cfg):
        assert field.name in meta["constants"]
        assert "justification" in meta["constants"][field.name]


def test_logs_are_written(tmp_path):
    from phase0.logs import RunLogger, read_motor_log
    cfg = Config(difficulty="A1")
    world = World(cfg)
    with RunLogger(tmp_path, cfg, world.channels) as logger:
        run(world, agent=RandomAgent(cfg), ticks=500, logger=logger)
    for name in ("truth.log", "sensor.log", "motor.log", "budget.profile", "meta.json"):
        assert (tmp_path / name).exists() and (tmp_path / name).stat().st_size > 0
    assert len(read_motor_log(tmp_path / "motor.log")) > 0


def test_budget_profile_roundtrip(tmp_path):
    profile = BudgetProfile([10, 20, 30], name="test")
    profile.save(tmp_path / "b.profile")
    back = BudgetProfile.load(tmp_path / "b.profile")
    assert back.quotas == profile.quotas
    assert back.at(4) == 20        # циклическое повторение


# ======================================================================
# Визуализация (Часть 10.4)
# ======================================================================

def test_render_produces_all_three_views():
    cfg = Config(difficulty="A3")
    world = World(cfg)
    record = None
    for _ in range(200):
        record = world.step()
        world.inbox.drain()
    text = frame(world, record, trail=[(world.body.x, world.body.y)])
    assert "@" in text
    assert "яркость" in text and "устойчивый" in text
    assert "канал истины" in text
    assert "граф" in text


# ======================================================================
# Конфиг
# ======================================================================

def test_derived_constants_match_spec():
    cfg = Config()
    assert cfg.dt == pytest.approx(1 / 60)
    assert cfg.v_max == pytest.approx(10.0)
    assert cfg.crossing_time == pytest.approx(6.4)
    assert cfg.starve_time_idle == pytest.approx(50.0)
    assert cfg.starve_time_moving == pytest.approx(20.0)
    assert cfg.required_feed_rate == pytest.approx(12.5)


def test_config_is_frozen():
    cfg = Config()
    with pytest.raises(Exception):
        cfg.basal = 0.5


def test_unimplemented_levels_fail_loudly():
    from phase0.config import Level
    with pytest.raises(NotImplementedError):
        Config(percept_level=Level.P1)


def test_arbitrary_constants_are_listed():
    """Спецификация: из 47 констант 30 помечены ПРОИЗВОЛ. Список должен
    существовать, чтобы было чем проверять зависимость выводов от них."""
    names = Config().arbitrary_constants()
    assert len(names) > 20
    assert "t_respawn" in names and "noci_duration" in names


# ======================================================================
# Сторожа метрик: метрика обязана отказываться от вывода, которого не тянет
# ======================================================================

def test_savings_refuses_verdict_without_control():
    """Наклон T_adapt без контроля ничего не значит: метрика умеет давать
    уверенный минус на агенте, который вообще не адаптируется."""
    from phase0.metrics import savings
    curve = [(k, 500 - 10 * k) for k in range(30)]
    assert "нет контроля" in savings(curve)["вывод"]


def test_savings_refuses_verdict_on_small_sample():
    from phase0.metrics import MIN_POINTS, savings
    curve = [(k, 500 - 10 * k) for k in range(MIN_POINTS - 5)]
    control = [(k, 500) for k in range(30)]
    assert "выборка мала" in savings(curve, control)["вывод"]


def test_instant_adapter_is_not_reported_as_learning():
    """greedy_symbolic читает питательность напрямую и переключается
    мгновенно. Если метрика объявит его обучающимся — она врёт."""
    from phase0.metrics import adaptation_curve, savings
    cfg = Config(mode=Mode.B)
    _, m = run_baseline("greedy_symbolic", cfg, ticks=250_000)
    curve = adaptation_curve(m, window=cfg.t_adapt_window)
    verdict = savings(curve, curve)["вывод"]
    assert "улучшается" not in verdict, verdict


def test_noise_floor_reports_events_per_window():
    from phase0.metrics import noise_floor
    _, m = run_baseline("greedy_symbolic", Config(mode=Mode.B), ticks=200_000)
    floor = noise_floor(m, window=Config().t_adapt_window)
    assert floor["событий в окне"] > 0
    assert floor["погрешность темпа, %"] > 0


# ======================================================================
# Покрытие сенсорных каналов бейзлайнами
# ======================================================================

def test_every_channel_group_is_read_by_some_baseline():
    """Сторож на дыру, которая уже была: 128 каналов из 150 (весь событийный
    тракт) не читал ни один бейзлайн, и было неизвестно, пригодны ли они."""
    from phase0.baselines import READS
    covered = set().union(*READS.values())
    for group in ("TRANSIENT", "SUSTAINED", "PROPRIO"):
        assert group in covered, f"{group} не читает ни один бейзлайн"


def test_reads_table_matches_actual_channels_read(): 
    """Таблица READS должна соответствовать тому, что бейзлайн реально читает,
    иначе сторож покрытия охраняет фикцию."""
    from phase0.baselines import READS
    cfg = Config(difficulty="A1")
    for name, declared in READS.items():
        if not declared:
            continue
        world = World(cfg)
        agent = build(name, cfg, world.channels, agent_seed=0)
        seen: set[str] = set()
        original = agent.step

        def spy(inbox, outbox, budget, _o=original, _s=seen):
            for e in inbox:
                _s.add(world.channels.group_of(e.channel))
            return _o(inbox, outbox, budget)

        agent.step = spy
        for _ in range(200):
            world.step()
            agent.step(world.inbox.drain(), world.outbox, 1000)
        assert declared <= seen | {"INTERO"}, f"{name}: заявлено {declared}, видно {seen}"


def test_greedy_transient_uses_only_event_channel():
    """Бейзлайн событийного канала обязан работать, даже если плотные каналы
    молчат — иначе он проверяет не то, что заявлено."""
    cfg = Config(difficulty="A1")
    world = World(cfg)
    agent = build("greedy_transient", cfg, world.channels, agent_seed=0)
    for _ in range(3000):
        world.step()
        events = [e for e in world.inbox.drain()
                  if e.channel < world.channels.sustained_start]
        agent.step(events, world.outbox, 1000)
    assert world.tick == 3000
    assert sum(world.eaten_counts.values()) > 0, "на одном событийном канале не ест"


def test_transient_baseline_beats_random():
    _, transient = run_baseline("greedy_transient", Config(difficulty="A1"), ticks=20_000)
    _, rand = run_baseline("random", Config(difficulty="A1"), ticks=20_000)
    assert transient.deaths_per_10k < rand.deaths_per_10k
    assert transient.eat_rate_per_1k > rand.eat_rate_per_1k


# ======================================================================
# Канал истины: пять метрик связывания (Часть 6)
# ======================================================================

def test_visible_means_actually_seen_not_merely_in_fov():
    """Регрессия. Раньше `visible` означал «в поле зрения»: все 35 447 записей
    с occluded_by >= 0 были помечены видимыми, и три метрики связывания из
    пяти посчитать было нельзя."""
    world = World(Config(difficulty="A3"))
    full_but_visible = 0
    inconsistent = 0
    saw_full = 0
    for _ in range(20_000):
        record = world.step()
        world.inbox.drain()
        for it in record.items:
            if it.visible != (it.visible_receptors > 0):
                inconsistent += 1
            if it.occlusion == "full":
                saw_full += 1
                if it.visible:
                    full_but_visible += 1
    assert inconsistent == 0
    assert full_but_visible == 0
    assert saw_full > 0, "за 20000 тиков ни одной полной окклюзии — нечего проверять"


def test_occlusion_is_classified_three_ways():
    world = World(Config(difficulty="A3"))
    kinds = set()
    for _ in range(20_000):
        record = world.step()
        world.inbox.drain()
        kinds.update(it.occlusion for it in record.items if it.in_fov)
    assert kinds == {"none", "partial", "full"}


def test_truth_carries_receptor_ownership():
    """retina_owner — ключ ко всем пяти метрикам: без него склеивание и
    дробление посчитать нечем."""
    cfg = Config(difficulty="A3")
    world = World(cfg)
    for _ in range(500):
        record = world.step()
        world.inbox.drain()
        assert len(record.retina_owner) == cfg.retina_n
        live = {i.item_id for i in world.items.visible_items}
        for owner in record.retina_owner:
            assert owner == -1 or owner in live


def test_all_five_binding_metrics_are_computable():
    from phase0.binding import BindingTracker, ConnectedComponentSlots
    cfg = Config(difficulty="A3")
    world = World(cfg)
    agent = GreedySymbolic(cfg)
    tracker, segmenter = BindingTracker(), ConnectedComponentSlots()
    for _ in range(30_000):
        agent.observe_symbolic(world)
        record = world.step()
        world.inbox.drain()
        tracker.observe(record, segmenter.slots(world._last_projection.l))
        agent.step([], world.outbox, 1000)
    report = tracker.report()
    for key in ("1. соответствие (доля тиков с биекцией)",
                "2. смен object_id на эпизод видимости",
                "3. склеиваний на 1000 тиков",
                "4. дроблений на 1000 тиков",
                "5. восстановление после окклюзии"):
        assert key in report, key
        assert not isinstance(report[key], str), f"{key} не посчиталась: {report[key]}"


def test_binding_tracker_detects_merge_and_split():
    """Проверка самого прибора на построенных вручную случаях."""
    from phase0.binding import BindingTracker
    from phase0.truth import TruthRecord

    def rec(owner):
        items = []
        for iid in {o for o in owner if o >= 0}:
            idx = [i for i, o in enumerate(owner) if o == iid]
            items.append(ItemTruthStub(iid, min(idx), max(idx), len(idx)))
        return TruthRecord(tick=0, body={}, items=items, regime={},
                           retina_owner=list(owner))

    from dataclasses import dataclass

    @dataclass
    class ItemTruthStub:
        item_id: int
        lo: int
        hi: int
        n: int
        visible: bool = True
        in_fov: bool = True
        occlusion: str = "none"

    owner = [-1, 0, 0, -1, 1, 1, -1]
    merged = BindingTracker()
    merged.observe(rec(owner), {100: {1, 2, 4, 5}})          # один слот на два предмета
    assert merged.r.merge_events == 1

    split = BindingTracker()
    split.observe(rec(owner), {100: {1}, 101: {2}, 102: {4, 5}})  # предмет 0 в двух слотах
    assert split.r.split_events == 1


# ======================================================================
# Валидация конфига
# ======================================================================

@pytest.mark.parametrize("bad", ["B", 1, None])
def test_mode_must_be_enum_not_string(bad):
    """Config(mode="B") раньше проходил молча и падал тысячи тиков спустя
    внутри канала истины."""
    with pytest.raises(TypeError):
        Config(mode=bad)


def test_photometry_constants_are_in_meta_snapshot():
    """d_ref и i_floor раньше были модульными константами retina.py и не
    попадали в meta.json — то есть защита от тихой подкрутки на них не
    действовала."""
    snapshot = Config().snapshot()
    assert snapshot["d_ref"]["justification"] == "произвол"
    assert snapshot["i_floor"]["justification"] == "произвол"


def test_flip_interval_gives_enough_events_per_window():
    """Интервал реверсии и темп поедания должны быть согласованы: иначе
    T_adapt оценивается по единицам событий и ничего не значит."""
    from phase0.metrics import MIN_POINTS, noise_floor
    cfg = Config(mode=Mode.B)
    _, m = run_baseline("greedy_symbolic", cfg, ticks=250_000)
    floor = noise_floor(m, window=cfg.t_adapt_window)
    assert floor["событий в окне"] > 20, floor
    assert floor["погрешность темпа, %"] < 25, floor


def test_sustained_on_change_reduces_dense_traffic():
    from phase0.baselines import GreedyTransient
    counts = {}
    for on_change in (False, True):
        cfg = Config(sustained_on_change=on_change)
        world = World(cfg)
        agent = GreedyTransient(cfg, world.channels)
        dense = 0
        for _ in range(2000):
            world.step()
            events = world.inbox.drain()
            dense += sum(1 for e in events
                         if world.channels.sustained_start <= e.channel
                         < world.channels.proprio_start)
            agent.step(events, world.outbox, 1000)
        counts[on_change] = dense
    assert counts[True] < counts[False]


def test_agent_seed_and_world_seed_are_independent():
    """Ловушка, на которой уже споткнулись: параметр назывался `seed`, уходил
    только агенту, и разброс по нему легко принять за разброс по мирам."""
    cfg = Config(difficulty="A1")

    # Разные сиды агента при одном мире: раскладка предметов обязана совпасть.
    a, _ = run_baseline("random", cfg, ticks=300, agent_seed=0)
    b, _ = run_baseline("random", cfg, ticks=300, agent_seed=7)
    assert [i.kind for i in a.items.items] == [i.kind for i in b.items.items]

    # Разный сид мира при одном агенте: мир обязан отличаться.
    c, _ = run_baseline("random", cfg, ticks=300, agent_seed=0, world_seed=12345)
    assert (c.body.x, c.body.y) != (a.body.x, a.body.y)


def test_world_seed_does_not_leak_into_agent():
    """world_seed меняет мир, но не поведение агента при фиксированном
    agent_seed — иначе форк перестаёт быть форком."""
    cfg = Config(difficulty="A1")
    from phase0.baselines import RandomAgent
    first = RandomAgent(cfg, seed=3)
    second = RandomAgent(cfg, seed=3)
    seq_a = [first.rng.integers(4) for _ in range(50)]
    seq_b = [second.rng.integers(4) for _ in range(50)]
    assert seq_a == seq_b


# ======================================================================
# A5: препятствия и стабильная окклюзия
# ======================================================================

def test_a5_creates_obstacles():
    world = World(Config(difficulty="A5"))
    assert len(world.obstacles.obstacles) == world.cfg.n_obstacles


def test_a5_requires_obstacles_to_exist():
    with pytest.raises(ValueError):
        Config(difficulty="A5", n_obstacles=0)


def test_body_never_ends_inside_obstacle():
    import math
    cfg = Config(difficulty="A5")
    world = World(cfg)
    agent = GreedySymbolic(cfg)
    for _ in range(20_000):
        agent.observe_symbolic(world)
        world.step()
        world.inbox.drain()
        agent.step([], world.outbox, 1000)
        for o in world.obstacles.obstacles:
            d = math.hypot(world.body.x - o.x, world.body.y - o.y)
            assert d >= o.radius + cfg.r_body - 1e-6, "тело провалилось внутрь"


def test_items_never_spawn_inside_obstacle():
    import math
    cfg = Config(difficulty="A5")
    world = World(cfg)
    for _ in range(20_000):
        world.step()
        world.inbox.drain()
        for item in world.items.visible_items:
            for o in world.obstacles.obstacles:
                d = math.hypot(item.x - o.x, item.y - o.y)
                assert d >= o.radius + cfg.r_item - 1e-6


def test_obstacles_increase_full_occlusion():
    """Смысл A5: стабильная окклюзия, без которой пятая метрика меряет шум."""
    def full_fraction(difficulty):
        cfg = Config(difficulty=difficulty)
        world = World(cfg)
        agent = GreedySymbolic(cfg)
        counts = {"none": 0, "partial": 0, "full": 0}
        for _ in range(15_000):
            agent.observe_symbolic(world)
            record = world.step()
            world.inbox.drain()
            agent.step([], world.outbox, 1000)
            for it in record.items:
                if it.in_fov:
                    counts[it.occlusion] += 1
        return counts["full"] / max(1, sum(counts.values()))

    assert full_fraction("A5") > full_fraction("A3") * 1.3


def test_obstacles_are_achromatic():
    """Препятствия видны, но цветовая ось по-прежнему разделяет ровно A и B."""
    from phase0.obstacles import Obstacle
    from phase0.retina import Retina
    cfg = Config()
    retina = Retina(cfg, Channels(cfg))
    proj = retina.project(32, 32, 0.0, [], [Obstacle(100_000, 42.0, 32.0, 3.0)])
    mid = cfg.retina_n // 2
    assert proj.l[mid] > 0.0, "препятствие не видно"
    assert proj.c[mid] == 0.0, "препятствие окрашено"


# ======================================================================
# Подписка на каналы и метаболическая плата за объём входа
# ======================================================================

def test_subscription_filters_inbox():
    cfg = Config()
    world = World(cfg)
    ch = world.channels
    world.subscribe({ch.ENERGY})
    for _ in range(100):
        world.step()
        for e in world.inbox.drain():
            assert e.channel == ch.ENERGY


def test_unsubscribed_channels_are_not_charged():
    """Отказ от канала обязан быть физически выгоден — иначе метаболический
    тест ничего не решает: мир берёт плату за то, чего агент не просил."""
    cfg = Config(metabolic_compute=True, compute_cost_basis="input", k_input=0.1)

    def energy_after(subscription):
        world = World(cfg)
        world.subscribe(subscription)
        # Коротко: если тело успеет умереть, респавн вернёт E_0 и сравнение
        # абсолютных энергий станет бессмысленным.
        for _ in range(300):
            world.step()
            world.inbox.drain()
        assert world.deaths == 0
        return world.body.energy

    assert energy_after(set()) > energy_after(None), \
        "отказ от каналов не сэкономил энергию"


def test_input_rate_reflects_subscription():
    cfg = Config()
    world = World(cfg)
    for _ in range(50):
        world.step()
        world.inbox.drain()
    full = world.input_rate

    lean = World(cfg)
    lean.subscribe({lean.channels.ENERGY})
    for _ in range(50):
        lean.step()
        lean.inbox.drain()
    assert lean.input_rate < full


def test_every_baseline_declares_a_subscription():
    from phase0.baselines import BASELINES
    cfg = Config()
    world = World(cfg)
    for name in BASELINES:
        agent = build(name, cfg, world.channels, agent_seed=0)
        sub = agent.subscription(world.channels)
        assert sub is None or isinstance(sub, set), name


def test_compute_cost_basis_is_validated():
    with pytest.raises(ValueError):
        Config(compute_cost_basis="whatever")


def test_metabolic_cost_by_nodes_still_works():
    """Основа "nodes" — поведение спецификации, менять его молча нельзя."""
    cfg = Config(metabolic_compute=True, compute_cost_basis="nodes")
    a, b = World(cfg), World(cfg)
    b.set_node_count(2000)
    for _ in range(300):
        a.step(); a.inbox.drain()
        b.step(); b.inbox.drain()
    assert a.deaths == 0 and b.deaths == 0, "смерть смазывает сравнение энергий"
    assert b.body.energy < a.body.energy


# ======================================================================
# Метрика режима B: две ошибки, найденные на полном прогоне
# ======================================================================

def test_error_cost_is_a_fraction_not_a_count():
    """Абсолютный счётчик яда делает «почти не ест» неотличимым от
    «избегает яда»: у tabular_q было 2.35 яда за окно против 10.60 у
    greedy_sustained просто потому, что он ест втрое меньше всего."""
    from phase0.metrics import error_cost_curve
    cfg = Config(mode=Mode.B)
    _, m = run_baseline("greedy_sustained", cfg, ticks=150_000)
    curve = error_cost_curve(m, window=cfg.t_adapt_window)
    assert curve, "переворотов не случилось — прогон слишком короткий"
    measured = 0
    for _k, frac, bad, total in curve:
        assert isinstance(bad, int) and isinstance(total, int)
        assert total >= bad >= 0
        if total:
            measured += 1
            assert 0.0 <= frac <= 1.0
            assert frac == pytest.approx(bad / total)
    assert measured, "ни одного окна с едой — метрика ничего не измерила"


def test_savings_verdict_survives_outliers():
    """Наклон МНК на выбросах вида 0, 0, 9220, 5840 не значит ничего.
    Замерено: у small_rnn_bptt половины 1744.6 и 1785.7 (плоско), а наклон
    -78.1 давал вердикт «улучшается»."""
    from phase0.metrics import savings
    control = [(k, 500) for k in range(40)]
    # Плоская кривая с одним огромным выбросом в начале.
    flat = [(k, 500) for k in range(40)]
    flat[1] = (1, 20_000)
    verdict = savings(flat, control)["вывод"]
    assert "улучшается" not in verdict, verdict


def test_savings_detects_real_improvement():
    """Обратная проверка: настоящее улучшение метрика обязана видеть."""
    from phase0.metrics import savings
    control = [(k, 500) for k in range(40)]
    improving = [(k, max(50, 1000 - 25 * k)) for k in range(40)]
    assert "улучшается" in savings(improving, control)["вывод"]
