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

from phase0 import (Config, World, adaptation_curve, error_cost_curve,  # noqa: E402
                    noise_floor, run_baseline, savings)
from phase0.config import Kind, Mode  # noqa: E402
from phase0.driver import replay  # noqa: E402
from phase0.baselines import GreedySymbolic  # noqa: E402

RULE = "=" * 92
ALL = ("random", "greedy_symbolic", "greedy_pixel", "linear_pixel",
       "tabular_q", "small_rnn_bptt")


def baselines(ticks: int = 30_000) -> None:
    print(RULE)
    print(f"ШАГ 0.7. Бейзлайны на A1-A3, по {ticks} тиков ({ticks/60/60:.1f} мин "
          f"модельного времени)")
    print(RULE)
    print()
    print("  Два бейзлайна разделяют два разных вопроса, и смешивать их нельзя:")
    print("    greedy_symbolic — решаема ли задача ВООБЩЕ (вход: позиции)")
    print("    greedy_pixel    — решаема ли она ИЗ ПИКСЕЛЕЙ (вход: сетчатка)")
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
    print("  greedy_pixel — цена того, что мир виден через сетчатку, а не дан")
    print("  списком координат. random — нижняя отметка, без неё остальные числа")
    print("  ни о чём не говорят.")


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


def reversal(ticks: int = 200_000) -> None:
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
    for name in ("greedy_symbolic", "greedy_pixel", "small_rnn_bptt", "tabular_q"):
        _, m = run_baseline(name, Config(mode=Mode.B), ticks=ticks)
        results[name] = (m, adaptation_curve(m), error_cost_curve(m))

    control_curve = results["greedy_symbolic"][1]
    floor = noise_floor(results["greedy_symbolic"][0])
    print(f"  Шумовой пол метрики: {floor}")
    print()

    for name, (m, curve, errors) in results.items():
        tag = "  <- КОНТРОЛЬ (адаптируется мгновенно)" if name == "greedy_symbolic" else ""
        s = savings(curve, None if name == "greedy_symbolic" else control_curve)
        shown = " ".join(f"{v}" if v is not None else "—" for _, v in curve[:10])
        first_half = [c for _, c in errors[:len(errors) // 2]]
        second_half = [c for _, c in errors[len(errors) // 2:]]
        print(f"  {name}{tag}")
        print(f"    переворотов {len(m.flips):3d}  яда всего {m.poison:4d}  "
              f"смертей/10k {m.deaths_per_10k:5.2f}")
        print(f"    T_adapt(k) первые 10: {shown}")
        print(f"    {s}")
        print(f"    стоимость ошибок (яда за 900 тиков после переворота): "
              f"первая половина {np.mean(first_half):.2f}, "
              f"вторая {np.mean(second_half):.2f}")
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


SECTIONS = {"baselines": baselines, "economy": economy,
            "reversal": reversal, "determinism": determinism}


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
