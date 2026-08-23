"""Приёмка Фазы 0: таблица бейзлайнов и кривая адаптации.

    python3 experiments/phase0_report.py baselines   # шаг 0.7, режим A
    python3 experiments/phase0_report.py reversal    # шаг 0.8, режим B
    python3 experiments/phase0_report.py economy     # проверка констант Части 3
    python3 experiments/phase0_report.py determinism # шаг 0.5

Числа отсюда — единственное основание считать шаги 0.3-0.8 закрытыми.
Ни одно из них не возвращается в мир: это приборы, а не обратная связь.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase0 import (Config, World, adaptation_curve, build, error_cost_curve,  # noqa: E402
                    error_cost_summary,
                    noise_floor, run_baseline, savings)
from phase0.config import Kind, Mode  # noqa: E402
from phase0.driver import replay  # noqa: E402
from phase0.baselines import GreedySymbolic  # noqa: E402
from phase0.binding import BindingTracker, ConnectedComponentSlots  # noqa: E402
from phase0.events import Event  # noqa: E402

RULE = "=" * 92
ALL = ("random", "greedy_symbolic", "greedy_sustained", "greedy_transient",
       "linear_pixel", "tabular_q", "small_rnn_bptt")


def baselines(ticks: int = 30_000) -> None:
    print(RULE)
    print(f"ШАГ 0.7. Бейзлайны на A1-A3, по {ticks} тиков ({ticks/60/60:.1f} мин "
          f"модельного времени)")
    print(RULE)
    print()
    print("  Три бейзлайна разделяют три разных вопроса, смешивать нельзя:")
    print("    greedy_symbolic  — решаема ли задача ВООБЩЕ (вход: позиции)")
    print("    greedy_sustained — устойчивый канал, 8 рецепторов по 15 град.")
    print("    greedy_transient — СОБЫТИЙНЫЙ канал, 32 рецептора по 3.75 град.")
    print()
    header = (f"  {'бейзлайн':17s} {'ур.':4s} {'смертей/10k':>11s} {'E сред':>7s} "
              f"{'E дисп':>7s} {'съед/1k':>8s} {'яда':>5s} {'E<0.2':>6s} {'стен':>6s}")
    for difficulty in ("A1", "A2", "A3"):
        print(header if difficulty == "A1" else "")
        for name in ALL:
            _, m = run_baseline(name, Config(difficulty=difficulty), ticks=ticks)
            r = m.row()
            print(f"  {name:17s} {difficulty:4s} {r['смертей/10k']:11.2f} "
                  f"{r['E среднее']:7.3f} {r['E дисперсия']:7.4f} "
                  f"{r['съедено/1k']:8.2f} {r['из них яд']:5d} "
                  f"{r['доля E<0.2']:6.3f} {r['ударов о стену']:6d}")
    print()
    print("  Как читать. greedy_symbolic обязан держать A1-A3 с нулём смертей —")
    print("  это проверка КОНСТАНТ Части 3, а не бейзлайна. Разрыв между ним и")
    print("  greedy_sustained/greedy_transient — цена того, что мир виден через")
    print("  сетчатку, а не дан списком координат. Разница МЕЖДУ ними — цена")
    print("  событийного кодирования против плотного, при равном намерении.")
    print("  random — нижняя отметка, без неё остальные числа ни о чём не говорят.")


def economy() -> None:
    """Константы Части 3 — проверка, что задача решаема, но требует усилия."""
    cfg = Config()
    print(RULE)
    print("ЭКОНОМИКА (Часть 3). Выведенные величины против замера")
    print(RULE)
    print()
    print(f"  голодная смерть в покое      {cfg.starve_time_idle:6.1f} с  [выведено]")
    print(f"  голодная смерть на полной тяге {cfg.starve_time_moving:4.1f} с  [выведено]")
    print(f"  требуемый темп в покое       {cfg.required_feed_rate:6.1f} с/предмет")
    print(f"  плотность еды                {cfg.item_density:.5f} предметов/у.е.^2")
    print(f"  пересечение арены            {cfg.crossing_time:6.1f} с  [выведено]")
    print()

    # Расстояние до ближайшего предмета: спецификация обещает 8-10 у.е.
    rng = np.random.default_rng(0)
    dists = []
    for _ in range(4000):
        pts = rng.uniform(1, 63, (cfg.n_items, 2))
        me = rng.uniform(1, 63, 2)
        dists.append(float(np.min(np.hypot(pts[:, 0] - me[0], pts[:, 1] - me[1]))))
    print(f"  расстояние до ближайшего предмета: медиана {np.median(dists):.1f} у.е., "
          f"среднее {np.mean(dists):.1f} у.е.")
    print(f"  спецификация обещает 8-10 у.е. -> {'СХОДИТСЯ' if 7 <= np.mean(dists) <= 11 else 'РАСХОДИТСЯ'}")
    print()
    print("  Замер темпа поедания у потолка задачи:")
    for difficulty in ("A1", "A2", "A3"):
        world, m = run_baseline("greedy_symbolic", Config(difficulty=difficulty),
                                ticks=30_000)
        interval = 30_000 / max(1, m.eaten_a + m.eaten_b) / 60
        need = cfg.required_feed_rate
        print(f"    {difficulty}: один предмет за {interval:5.1f} с "
              f"(нужно не реже {need:.1f} с в покое) -> "
              f"{'запас есть' if interval < need else 'НЕ ТЯНЕТ'}"
              f",  состав поля {world.items.composition()}")


def sensory() -> None:
    """Числа для решений по спецификации, которые принимает автор, а не код.

    Здесь ничего не чинится. Здесь измеряется то, от чего зависят три
    открытых вопроса: чем занят вход, что несёт транзиентный канал и
    сколько времени агент вообще ничего не видит.
    """
    from phase0.baselines import GreedyTransient, RandomAgent
    from phase0.items import Item
    from phase0.retina import Retina

    print(RULE)
    print("СЕНСОРНЫЙ ТРАКТ: замеры для решений по спецификации")
    print(RULE)

    # --- 1. чем занят вход ---
    print()
    print("A. Объём входа по группам каналов (активное движение, 20000 тиков)")
    print()
    for on_change in (False, True):
        cfg = Config(sustained_on_change=on_change)
        world = World(cfg)
        agent = GreedyTransient(cfg, world.channels)
        counts = {"TRANSIENT": 0, "SUSTAINED": 0, "PROPRIO": 0, "INTERO": 0}
        ticks = 20_000
        for _ in range(ticks):
            world.step()
            events = world.inbox.drain()
            for e in events:
                counts[world.channels.group_of(e.channel)] += 1
            agent.step(events, world.outbox, 1000)
        total = sum(counts.values()) / ticks
        trans = counts["TRANSIENT"] / ticks
        label = "sustained ПО ИЗМЕНЕНИЮ" if on_change else "sustained плотный (как в спец.)"
        print(f"  {label}")
        print("    " + "  ".join(f"{k} {v/ticks:5.2f}/тик" for k, v in counts.items()))
        print(f"    всего {total:5.2f}/тик, доля событийного канала "
              f"{100*trans/max(total,1e-9):4.1f}%")
    print()
    print("  Утверждение 4.4 «статичная сцена даёт НОЛЬ входа» верно только для")
    print("  транзиентного канала. Полный вход статичной сцены — 21 событие в тик")
    print("  (16 sustained + 4 proprio + 1 energy). Узел, которому нужно «что")
    print("  вокруг», будет учиться на плотном канале и игнорировать событийный.")
    print("  Решение автора: либо признать sustained ведущим и переписать 4.4,")
    print("  либо включить sustained_on_change и сделать событийным весь тракт.")

    # --- 2. что несёт транзиентный канал по глубине ---
    print()
    print("B. Несёт ли транзиентный канал ГЛУБИНУ")
    print()
    cfg = Config()
    print(f"  при d_ref={cfg.d_ref}, theta={cfg.theta_event}:")
    retina = Retina(cfg, __import__("phase0").Channels(cfg))
    empty = retina.project(32, 32, 0.0, [])
    for dist in (4.0, 8.0, 16.0, 32.0):
        proj = retina.project(32, 32, 0.0, [Item(0, 32.0 + dist, 32.0, Kind.A)])
        mid = int(np.argmax(proj.l))
        contrast = float(proj.l[mid]) - float(empty.l[mid])
        print(f"    предмет на {dist:5.1f} у.е.: контраст к пустоте {contrast:5.2f} "
              f"= {contrast/cfg.theta_event:5.1f} порогов, "
              f"рецепторов {int((proj.owner >= 0).sum()):2d}")

    # Сколько событий даёт всё сближение с 32 у.е. до контакта.
    retina = Retina(cfg, __import__("phase0").Channels(cfg))
    total_events = 0
    for step in range(300):
        d = 32.0 - step * 0.1
        if d < 2.0:
            break
        proj = retina.project(32, 32, 0.0, [Item(0, 32.0 + d, 32.0, Kind.A)])
        ev, _ = retina.transient_events(step * 3, proj)
        total_events += len(ev)
    print(f"    ВСЁ сближение с 32 у.е. до контакта: {total_events} событий")
    print()
    print("  Контраст «объект против пустоты» огромен, а градиент по расстоянию")
    print("  пологий: канал работает детектором ГРАНИЦ И ДВИЖЕНИЯ, дальномер из")
    print("  него слабый. Ручка — d_ref (меньше = чувствительнее к глубине),")
    print("  обе константы помечены ПРОИЗВОЛ.")

    # --- 3. слепые тики против FOV ---
    print()
    print("C. Доля тиков, в которых не видно НИ ОДНОГО предмета")
    print()
    print("  Число зависит от ПОВЕДЕНИЯ, а не только от мира: тот, кто крутится")
    print("  к еде, слеп реже. Поэтому замер идёт по двум агентам сразу —")
    print("  нижняя и верхняя оценка.")
    print()
    print(f"    {'FOV':>8s} {'random':>10s} {'greedy_transient':>18s}")
    for fov in (90, 120, 150, 180, 240):
        row = []
        for maker in ("random", "transient"):
            cfg = Config(fov_deg=float(fov))
            world = World(cfg)
            agent = (RandomAgent(cfg) if maker == "random"
                     else GreedyTransient(cfg, world.channels))
            blind = 0
            ticks = 20_000
            for _ in range(ticks):
                record = world.step()
                events = world.inbox.drain()
                if not any(it.visible for it in record.items):
                    blind += 1
                agent.step(events, world.outbox, 1000)
            row.append(100 * blind / ticks)
        mark = "  <- значение спецификации" if fov == 120 else ""
        print(f"    {fov:6d}   {row[0]:9.1f}% {row[1]:17.1f}%{mark}")
    print()
    print("  Слепые тики бьют по любому предсказателю: предсказывать нечего, а")
    print("  Уровень 1 онтогенеза живёт именно на предсказании последствий")
    print("  собственного действия. fov_deg помечен ПРОИЗВОЛ, решение за автором.")


def binding(ticks: int = 60_000) -> None:
    """Пять метрик связывания (Часть 6) на A3 и A5.

    Опорный сегментатор — ПРИБОР, а не кандидат в агенты: он режет сетчатку на
    непрерывные куски и держит идентичность по перекрытию с прошлым тиком.
    Его числа — нижняя отметка: столько выбивает тривиальная сегментация без
    всякой памяти.

    Зачем оба уровня. На A3 окклюзия мимолётна (предметы движутся и
    респавнятся), и пятая метрика меряет почти шум. На A5 препятствия стоят,
    поэтому предмет скрыт столько, сколько агент туда не заходит.
    """
    from phase0.baselines import GreedySymbolic

    print(RULE)
    print(f"МЕТРИКИ СВЯЗЫВАНИЯ (Часть 6), {ticks} тиков")
    print(RULE)

    for difficulty in ("A3", "A5"):
        cfg = Config(difficulty=difficulty)
        world = World(cfg)
        agent = GreedySymbolic(cfg)
        tracker = BindingTracker()
        segmenter = ConnectedComponentSlots()
        occl = {"none": 0, "partial": 0, "full": 0}

        for _ in range(ticks):
            agent.observe_symbolic(world)
            record = world.step()
            world.inbox.drain()
            tracker.observe(record, segmenter.slots(world._last_projection.l))
            for it in record.items:
                if it.in_fov:
                    occl[it.occlusion] += 1
            agent.step([], world.outbox, 1000)

        total = sum(occl.values()) or 1
        note = " (препятствий нет)" if difficulty == "A3" else \
               f" ({len(world.obstacles.obstacles)} препятствия)"
        print()
        print(f"  --- {difficulty}{note} ---")
        print("  Окклюзия: " + "  ".join(
            f"{k} {100*v/total:4.1f}%" for k, v in occl.items()))
        print("  Опорный сегментатор:")
        for k, v in tracker.report().items():
            print(f"    {k}: {v}")

    print()
    print("  Читать так. Все пять метрик считаются. Раньше три из пяти посчитать")
    print("  было нельзя: visible означал «в поле зрения», а не «видно».")
    print("  Разница A3 против A5 показывает, чего стоила мимолётность окклюзии:")
    print("  пятая метрика на A3 меряла в основном перекрытия длиной в доли")
    print("  секунды, на A5 — настоящее скрытие за препятствием.")


def metabolic(ticks: int = 40_000) -> None:
    """Оправданы ли 128 транзиентных каналов. Часть 3.2 плюс замер.

    Вопрос ребром. При выровненных условиях (одинаковое разрешение, без
    запаздывания) событийный канал НЕ даёт выигрыша в поведении: 1.67 против
    1.33 у плотного. Единственное, что у него остаётся, — объём: 1.92 события
    в тик против 64. Если метаболическая плата берётся за объём входа, эта
    разница становится физической. Если нет — держать событийный тракт незачем.

    Ответ зависит от k_input, помеченного ПРОИЗВОЛ, поэтому здесь меряется
    ПОРОГ, на котором сравнение переворачивается, а не один вердикт.
    """
    print(RULE)
    print("МЕТАБОЛИЧЕСКИЙ ТЕСТ СОБЫТИЙНОГО КАНАЛА")
    print(RULE)
    print()

    matched = dict(difficulty="A1", sustained_n=32, tau_sustained=0.0)

    print("  Объём входа, событий/тик (выписанные каналы):")
    for name in ("greedy_transient", "greedy_sustained"):
        cfg = Config(**matched)
        world, _ = run_baseline(name, cfg, ticks=3000)
        print(f"    {name:18s} {world.input_rate:6.1f}")
    print()
    print("  Смертей/10k при разной плате за объём входа")
    print("  (условия выровнены: sustained_n=32, tau_sustained=0)")
    print()
    print("  Усреднение по 4 сидам МИРА. Именно мира, а не агента: обе жадины")
    print("  детерминированы, rng в их политике не участвует, и разброс по")
    print("  agent_seed равен ровно нулю. Усреднять по сиду, который ни на что")
    print("  не влияет, — способ получить красивую погрешность вместо честной.")
    print()
    seeds = (0, 1, 2, 3)
    print(f"    {'k_input':>9s} | {'greedy_transient':>21s} | {'greedy_sustained':>21s}")
    flip = None
    for k in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
        row = []
        for name in ("greedy_transient", "greedy_sustained"):
            vals = []
            for s in seeds:
                cfg = Config(metabolic_compute=(k > 0), compute_cost_basis="input",
                             k_input=k, **matched)
                _, m = run_baseline(name, cfg, ticks=ticks, world_seed=s)
                vals.append(m.deaths_per_10k)
            row.append((float(np.mean(vals)), float(np.std(vals))))
        if flip is None and row[0][0] < row[1][0] - row[1][1]:
            flip = k
        print(f"    {k:9.3f} | {row[0][0]:14.2f} ±{row[0][1]:4.2f} | "
              f"{row[1][0]:14.2f} ±{row[1][1]:4.2f}")

    print()
    print(f"  basal = {Config().basal}/с — с ним и надо сравнивать k_input.")
    if flip is not None:
        print(f"  Событийный канал начинает выигрывать при k_input >= {flip:.3f},")
        print(f"  то есть когда плата за плотный вход составляет "
              f"{100*flip/Config().basal:.0f}% от basal.")
    else:
        print("  Событийный канал не выиграл ни при какой из проверенных плат.")
    print()
    print("  Как это читать. Число k_input — ПРОИЗВОЛ, и решение «удалять ли")
    print("  128 каналов» им и определяется. Замер говорит только одно: при")
    print("  нулевой плате за объём входа событийный тракт не нужен, а при")
    print("  плате порядка десятков процентов от basal — нужен. Выбрать")
    print("  величину платы должен автор, и это решение по спецификации.")


def prediction(ticks: int = 20_000) -> None:
    """Линейка для «Фазы 0.5»: с чем сравнивать предсказывающего агента.

    Здесь нет агента. Здесь эталон и проверка, что задача вообще непуста.
    """
    from phase0.predict import MeanPredictor, TrivialPredictor, score_predictor
    from phase0.events import Motor

    print(RULE)
    print("ПРЕДСКАЗАНИЕ СЕНСОРНОГО ПОТОКА: эталоны и непустота задачи")
    print(RULE)
    print()

    def spin(world, tick):
        if tick == 0:
            world.outbox.put(Event(world.tick, Motor.TURN, 0.6))
            world.outbox.put(Event(world.tick, Motor.THRUST, 0.4))

    setups = [
        ("A3, тело НЕПОДВИЖНО", Config(difficulty="A3"), None),
        ("A4, тело неподвижно, предметы движутся", Config(difficulty="A4"), None),
        ("A3, тело движется", Config(difficulty="A3"), spin),
        ("A4, тело движется, предметы движутся", Config(difficulty="A4"), spin),
    ]

    print(f"  {'постановка':42s} {'эталон':14s} {'норм.ошибка':>12s} {'движение входа':>15s}")
    for label, cfg, driver in setups:
        for maker in (TrivialPredictor, MeanPredictor):
            pred = maker()
            s = score_predictor(pred, cfg, ticks=ticks, driver=driver)
            print(f"  {label:42s} {pred.name:14s} {s.normalised:12.5f} "
                  f"{s.change_rate:15.4f}")
        print()

    print("  Читать так. «Движение входа» близко к нулю означает, что")
    print("  предсказывать НЕЧЕГО: устойчивый канал почти постоянен, и")
    print("  персистентность идеальна по построению. Сравнивать агента с ней")
    print("  на такой постановке бессмысленно — он не сможет выиграть, и это")
    print("  не будет говорить о нём ничего.")
    print()
    print("  Отсюда практический вывод для «Фазы 0.5»: агент, который только")
    print("  смотрит и не двигается, должен смотреть на A4, где движутся")
    print("  предметы. В статичном мире неподвижный наблюдатель не имеет")
    print("  задачи вовсе.")
    print()
    print("  Второй эталон, running_mean, нужен чтобы отличить «выучил")
    print("  динамику» от «выучил, что вход почти постоянен».")


def reversal(ticks: int = 1_200_000) -> None:
    print(RULE)
    print(f"ШАГ 0.8. Режим B — реверсия. {ticks} тиков ({ticks/60/60:.1f} мин)")
    print(RULE)
    print()
    print("  Ключевой вопрос НЕ «адаптируется ли», а УБЫВАЕТ ЛИ T_adapt(k).")
    print("  Убывающая кривая = выучена структура переключения. Плоская =")
    print("  переучивание с нуля каждый раз.")
    print()
    print("  ВНИМАНИЕ, ЧИТАТЬ ДО ТАБЛИЦЫ. greedy_symbolic читает питательность")
    print("  напрямую, то есть переключается МГНОВЕННО и не учится ничему.")
    print("  Поэтому его кривая — это КОНТРОЛЬ, то есть шум самой метрики.")
    print("  Наклон любого другого агента имеет смысл только рядом с ним.")
    print()

    results = {}
    for name in ("greedy_symbolic", "greedy_sustained", "greedy_transient",
                 "small_rnn_bptt", "tabular_q"):
        cfg = Config(mode=Mode.B)
        _, m = run_baseline(name, cfg, ticks=ticks)
        results[name] = (m, adaptation_curve(m, window=cfg.t_adapt_window),
                         error_cost_curve(m, window=cfg.t_adapt_window))

    control_curve = results["greedy_symbolic"][1]
    floor = noise_floor(results["greedy_symbolic"][0],
                        window=Config().t_adapt_window)
    print(f"  Шумовой пол метрики: {floor}")
    print()

    for name, (m, curve, errors) in results.items():
        tag = "  <- КОНТРОЛЬ (адаптируется мгновенно)" if name == "greedy_symbolic" else ""
        s = savings(curve, None if name == "greedy_symbolic" else control_curve)
        shown = " ".join(f"{v}" if v is not None else "—" for _, v in curve[:10])
        cost = error_cost_summary(errors)
        print(f"  {name}{tag}")
        print(f"    переворотов {len(m.flips):3d}  яда всего {m.poison:4d}  "
              f"смертей/10k {m.deaths_per_10k:5.2f}")
        print(f"    T_adapt(k) первые 10: {shown}")
        print(f"    {s}")
        print(f"    стоимость ошибок (ДОЛЯ яда среди съеденного за окно): {cost}")
        print()

    print("  Стоимость ошибок надёжнее T_adapt на здешних константах: это целые")
    print("  счётчики событий, а не оценка ТЕМПА по горстке событий.")
    print()
    print("  ОЖИДАНИЕ спецификации: у small_rnn_bptt кривая ПЛОСКАЯ. Это не")
    print("  неудача бейзлайна, а подтверждение, что режим B меряет нужное.")


def determinism(ticks: int = 200_000) -> None:
    print(RULE)
    print(f"ШАГ 0.5. Детерминизм: побитовый реплей {ticks} тиков")
    print(RULE)
    cfg = Config(difficulty="A3")

    def record(n):
        world, agent = World(cfg), GreedySymbolic(cfg)
        motor = []
        for _ in range(n):
            agent.observe_symbolic(world)
            world.step()
            world.inbox.drain()
            before = len(world.outbox)
            agent.step([], world.outbox, 1000)
            motor.extend(world.outbox.peek()[before:])
        b = world.body
        return motor, (b.x, b.y, b.theta, b.vx, b.vy, b.omega, b.energy), world

    t0 = time.time()
    motor, final, world = record(ticks)
    t1 = time.time()
    twin = replay(cfg, motor, ticks)
    t2 = time.time()
    b = twin.body
    got = (b.x, b.y, b.theta, b.vx, b.vy, b.omega, b.energy)

    print()
    print(f"  живой прогон : {t1 - t0:6.1f} с  ({ticks / (t1 - t0):,.0f} тиков/с)")
    print(f"  реплей       : {t2 - t1:6.1f} с")
    print(f"  моторных событий {len(motor)}, смертей {world.deaths}, "
          f"съедено {sum(world.eaten_counts.values())}")
    print(f"  потеряно сенсорных событий (EVENT_RATE_CAP): {world.dropped_events}")
    print()
    print(f"  ПОБИТОВОЕ СОВПАДЕНИЕ: {got == final}")
    if got != final:
        print(f"    живой : {final}")
        print(f"    реплей: {got}")


SECTIONS = {"baselines": baselines, "economy": economy, "sensory": sensory,
            "prediction": prediction,
            "metabolic": metabolic,
            "reversal": reversal, "determinism": determinism,
            "binding": binding}


def main() -> None:
    names = sys.argv[1:] or ["economy", "baselines"]
    for name in names:
        if name not in SECTIONS:
            raise SystemExit(f"нет раздела {name}. есть: {', '.join(SECTIONS)}")
    for i, name in enumerate(names):
        if i:
            print()
        SECTIONS[name]()


if __name__ == "__main__":
    main()
