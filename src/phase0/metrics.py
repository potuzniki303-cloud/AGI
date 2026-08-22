"""Метрики режимов. Части 7-9.

Всё, что здесь считается, считается ДЛЯ ЧЕЛОВЕКА и никогда не возвращается
в мир. Как только любая из этих величин начнёт влиять на физику, она станет
функцией приспособленности, а мир — оптимизатором.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Kind

# Минимум измеренных переворотов, при котором наклон T_adapt(k) вообще
# можно интерпретировать. Взят из шумового пола метрики, не из головы.
MIN_POINTS = 20


@dataclass
class RunMetrics:
    """Сводка прогона (Часть 7, раздел «Метрики»)."""

    ticks: int = 0
    deaths: int = 0
    energy: list[float] = field(default_factory=list)
    eaten_a: int = 0
    eaten_b: int = 0
    poison: int = 0
    wall_hits: int = 0
    dropped_events: int = 0
    danger_ticks: int = 0          # тиков с E < 0.2
    flips: list[int] = field(default_factory=list)
    eat_ticks: list[int] = field(default_factory=list)
    eat_values: list[float] = field(default_factory=list)

    # ------------------------------------------------------------- сводка
    @property
    def deaths_per_10k(self) -> float:
        return self.deaths / max(1, self.ticks) * 10_000

    @property
    def mean_energy(self) -> float:
        return float(np.mean(self.energy)) if self.energy else 0.0

    @property
    def var_energy(self) -> float:
        return float(np.var(self.energy)) if self.energy else 0.0

    @property
    def eat_rate_per_1k(self) -> float:
        return (self.eaten_a + self.eaten_b) / max(1, self.ticks) * 1000

    @property
    def danger_fraction(self) -> float:
        return self.danger_ticks / max(1, self.ticks)

    def row(self) -> dict[str, float]:
        return {
            "смертей/10k": round(self.deaths_per_10k, 2),
            "E среднее": round(self.mean_energy, 3),
            "E дисперсия": round(self.var_energy, 4),
            "съедено/1k": round(self.eat_rate_per_1k, 2),
            "из них яд": self.poison,
            "доля E<0.2": round(self.danger_fraction, 3),
            "ударов о стену": self.wall_hits,
        }

    # -------------------------------------------------------------- сбор
    def observe(self, world, record) -> None:
        self.ticks += 1
        e = world.body.energy
        self.energy.append(e)
        if e < 0.2:
            self.danger_ticks += 1
        for ev in record.events:
            kind = ev.get("type")
            if kind == "eat":
                self.eat_ticks.append(record.tick)
                self.eat_values.append(ev["dE"])
            elif kind == "regime_flip":
                self.flips.append(record.tick)
        self.deaths = world.deaths
        self.eaten_a = world.eaten_counts[Kind.A]
        self.eaten_b = world.eaten_counts[Kind.B]
        self.poison = world.poison_events
        self.wall_hits = world.wall_hits
        self.dropped_events = world.dropped_events


def adaptation_curve(metrics: RunMetrics, window: int = 6000) -> list[tuple[int, int | None]]:
    """T_adapt(k) для режима B (Часть 8).

    T_adapt(k) — число тиков от k-го переворота до восстановления
    дореверсионного темпа поедания ПИТАТЕЛЬНОГО.

    Главный вопрос не «адаптируется ли», а УБЫВАЕТ ЛИ T_adapt с ростом k.
    Убывающая кривая означает, что система выучила не два правила, а структуру
    переключения. Плоская — что она переучивается с нуля каждый раз.

    Возвращает [(k, T_adapt или None, если не восстановился)].
    """
    if not metrics.flips:
        return []

    good = np.array([t for t, v in zip(metrics.eat_ticks, metrics.eat_values) if v > 0])
    out: list[tuple[int, int | None]] = []

    for k, flip in enumerate(metrics.flips):
        before = int(((good >= flip - window) & (good < flip)).sum())
        baseline = before / window if window else 0.0
        if baseline <= 0.0:
            out.append((k, None))
            continue
        end = metrics.flips[k + 1] if k + 1 < len(metrics.flips) else metrics.ticks
        recovered: int | None = None
        # Скользящее окно после переворота: когда темп вернулся к дореверсионному.
        for t in range(flip, end - window, 20):
            rate = int(((good >= t) & (good < t + window)).sum()) / window
            if rate >= baseline:
                recovered = t - flip
                break
        out.append((k, recovered))
    return out


def error_cost_curve(metrics: RunMetrics, window: int = 9000) -> list[tuple[int, int]]:
    """Стоимость ошибок: сколько ЯДА съедено за окно после k-го переворота.

    Спецификация называет это сопутствующей метрикой, но на здешних константах
    она надёжнее главной: это целые счётчики событий, а не оценка ТЕМПА по
    горстке событий. См. предупреждение в `adaptation_curve`.
    """
    if not metrics.flips:
        return []
    bad = np.array([t for t, v in zip(metrics.eat_ticks, metrics.eat_values) if v < 0])
    out = []
    for k, flip in enumerate(metrics.flips):
        end = min(flip + window,
                  metrics.flips[k + 1] if k + 1 < len(metrics.flips) else metrics.ticks)
        out.append((k, int(((bad >= flip) & (bad < end)).sum())))
    return out


def noise_floor(metrics: RunMetrics, window: int = 6000) -> dict[str, float]:
    """Сколько событий поедания приходится на окно оценки темпа.

    Если это единицы, `T_adapt` шумодоминирована и её наклон ничего не
    значит без контроля. Замерено: при интервале реверсии 1500-4500 тиков и
    темпе `greedy_symbolic` в окно попадает ~3.6 события, то есть
    пуассоновская погрешность оценки темпа около +-53%.
    """
    good = [t for t, v in zip(metrics.eat_ticks, metrics.eat_values) if v > 0]
    per_tick = len(good) / max(1, metrics.ticks)
    per_window = per_tick * window
    return {
        "событий в окне": round(per_window, 2),
        "погрешность темпа, %": round(100 / max(per_window, 1e-9) ** 0.5, 0),
        "событий на период режима": round(
            len(good) / max(1, len(metrics.flips)), 1) if metrics.flips else 0.0,
    }


def savings(curve: list[tuple[int, int | None]],
            control: list[tuple[int, int | None]] | None = None) -> dict[str, float]:
    """Сбережение: улучшается ли адаптация от переворота к перевороту."""
    vals = [(k, v) for k, v in curve if v is not None]
    if len(vals) < 4:
        # Ключ «вывод» присутствует ВСЕГДА: вызывающий не должен гадать,
        # есть он или нет, и падать по KeyError на коротком прогоне.
        return {"n": len(vals),
                "вывод": f"измеримых переворотов {len(vals)} — считать нечего"}
    half = len(vals) // 2
    first = float(np.mean([v for _, v in vals[:half]]))
    second = float(np.mean([v for _, v in vals[half:]]))
    ks = np.array([k for k, _ in vals], dtype=float)
    ts = np.array([v for _, v in vals], dtype=float)
    slope = float(np.polyfit(ks, ts, 1)[0])
    out = {
        "n": len(vals),
        "T_adapt первая половина": round(first, 1),
        "T_adapt вторая половина": round(second, 1),
        "наклон": round(slope, 2),
    }
    if control is None:
        # Без контроля наклон НЕ является выводом: метрика умеет давать
        # уверенный отрицательный наклон на агенте, который вообще не
        # адаптируется (замерено: -2.31 у greedy_symbolic, который читает
        # питательность напрямую и переключается мгновенно).
        out["вывод"] = "нет контроля — наклон ничего не значит"
        return out

    cvals = [(k, v) for k, v in control if v is not None]
    if len(cvals) < 4:
        out["вывод"] = "контроль слишком короткий"
        return out
    cslope = float(np.polyfit(np.array([k for k, _ in cvals], dtype=float),
                              np.array([v for _, v in cvals], dtype=float), 1)[0])
    out["наклон контроля"] = round(cslope, 2)

    # Порог по размеру выборки. При погрешности оценки темпа около +-53%
    # (см. noise_floor) наклон по десятку точек — это не вывод, а гадание.
    # Без этого порога метрика уверенно объявляла «адаптация улучшается»
    # на n=6.
    if len(vals) < MIN_POINTS:
        out["вывод"] = (f"выборка мала (n={len(vals)} < {MIN_POINTS}) — "
                        "наклон не интерпретируется")
        return out

    out["вывод"] = ("адаптация улучшается" if slope < cslope - abs(cslope) * 0.5
                    else "неотличимо от контроля")
    return out
