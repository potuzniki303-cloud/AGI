"""Прямая конкуренция двух разумов в одном мире + калибровка плотности.

    python3 experiments/compete.py window     # какая плотность что отбирает
    python3 experiments/compete.py duel       # конкуренция линий, с контролем

Зачем это отдельно от diagnose.py. Сводные числа (популяция, медиана жизни)
при выходе на ёмкость среды перестают различать компетентность: приток еды
фиксирован (regrow x food_energy), поэтому средний агент по определению
выходит в ноль, каким бы умным он ни был. Разницу видно только в том, чья
ЛИНИЯ вытесняет чью.

ГЛАВНОЕ ПРЕДУПРЕЖДЕНИЕ. Дуэль без контроля не значит ничего. На популяции ~30
линия фиксируется чистым дрейфом: замер «одинаковые против одинаковых» давал
15-42% вместо 50%, и на нём легко принять случайность за отбор. Поэтому:

  * контроль (A против такого же A) гоняется всегда и обязан дать ~50%;
  * мера — доля ПРОГОНОВ, где B победил, по многим сидам, а не одна траектория;
  * популяция берётся большой, чтобы дрейф был слабее отбора.

Это общее правило, а не частность: любой вывод об отборе в конечной популяции
надо сравнивать с нулевой гипотезой «работал только дрейф».
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world import make, simulate  # noqa: E402

RULE = "=" * 78

# Поле вчетверо больше дефолтного forage при той же плотности и той же
# экономике на душу: пропорционально подняты и потолок еды, и отрастание.
# Смысл только один — больше тел, слабее дрейф.
BIG = dict(size=48, max_food=64, regrow_per_step=4.8,
           max_population=600, initial_population=160)


class MixedMind:
    """Ползунок компетентности: с вероятностью p идёт к еде, иначе случайно.

    p=0 — RandomMind, p=1 — GreedyMind без шума. Нужен, чтобы измерять не
    «работает/не работает», а форму зависимости выживания от умения: именно
    через промежуточные состояния эволюция обязана пройти.
    """

    def __init__(self, n_actions: int, rng: np.random.Generator, p: float,
                 tag: str = "") -> None:
        self.n_actions = n_actions
        self.rng = rng
        self.p = p
        self.tag = tag

    def act(self, observation: np.ndarray) -> int:
        if self.rng.random() >= self.p:
            return int(self.rng.integers(self.n_actions))
        antenna = observation[:4]
        if not np.any(antenna > 0):
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(antenna))

    def spawn(self, rng: np.random.Generator) -> "MixedMind":
        return MixedMind(self.n_actions, rng, self.p, self.tag)


# --------------------------------------------------------------------------
def window() -> None:
    """Какая плотность еды какую компетентность пропускает."""
    print(RULE)
    print("ОКНО ОТБОРА: роды за 4000 тактов (forage 24x24) при разной плотности")
    print(RULE)
    ps = (0.0, 0.1, 0.2, 0.3, 0.5, 0.9)
    print()
    print("  max_food плотн.|" + "".join(f"  p={p:<4.1f}" for p in ps) + " | порог")
    for max_food in (12, 16, 20, 24, 32, 40, 60):
        births = []
        for p in ps:
            world = make("forage", max_food=max_food)
            hist = simulate(world,
                            lambda rng: MixedMind(world.action_size, rng, p),
                            steps=4000)
            births.append(sum(hist.births))
        peak = max(births)
        if births[0] > 0.5 * peak:
            note = "давления НЕТ"
        else:
            thr = next((ps[i] for i, v in enumerate(births) if v > 0.3 * peak), None)
            note = f"p≈{thr}"
        print(f"  {max_food:8d} {max_food / 24 ** 2:.3f}|"
              + "".join(f" {v:6d}" for v in births) + f" | {note}")
    print()
    print("  Два вывода, оба важны для настройки мира:")
    print()
    print("  1. Плотность — это РУЧКА ПОРОГА. Она задаёт, насколько компетентным")
    print("     надо быть, чтобы вообще начать размножаться: 0.021 требует p≈0.3,")
    print("     0.056 требует p≈0.1, а на 0.104 не требует ничего.")
    print()
    print("  2. Порог — это ОБРЫВ, а не склон. Выше порога прибавка компетентности")
    print("     почти не окупается (роды 500 -> 650 при p от 0.2 до 0.9), потому что")
    print("     популяция упирается в ёмкость и приток еды делится на всех.")
    print()
    print("  Отсюда рецепт холодного старта: ставь плотность так, чтобы порог был")
    print("  ЧУТЬ НИЖЕ того, что случайный геном уже умеет, и опускай её потом.")
    print("  Ровно это делает bounty — но его параметры надо брать из замера,")
    print("  а не из формулы: настоящая безубыточность 0.103, а не 0.040.")


# --------------------------------------------------------------------------
def _one_duel(p_a: float, p_b: float, cfg: dict, steps: int, seed: int) -> float | None:
    world = make("forage", seed=seed, **cfg)
    counter = {"i": 0}

    def factory(rng: np.random.Generator) -> MixedMind:
        i = counter["i"]
        counter["i"] += 1
        return (MixedMind(world.action_size, rng, p_a, "A") if i % 2 == 0
                else MixedMind(world.action_size, rng, p_b, "B"))

    world.reset(factory)
    for _ in range(steps):
        world.step()
        if not world.bodies:
            return None
    return sum(1 for b in world.bodies if b.mind.tag == "B") / len(world.bodies)


def _report(label: str, p_a: float, p_b: float, cfg: dict,
            n_seeds: int = 40, steps: int = 3000) -> None:
    shares = [_one_duel(p_a, p_b, cfg, steps, s) for s in range(n_seeds)]
    shares = [s for s in shares if s is not None]
    if not shares:
        print(f"    {label:22s}: все прогоны вымерли")
        return
    wins = 100.0 * float(np.mean([s > 0.5 for s in shares]))
    err = 2 * 100.0 * np.sqrt(0.25 / len(shares))
    verdict = "ОТБОР" if abs(wins - 50) > err else "неотличимо от дрейфа"
    print(f"    {label:22s}: B победил в {wins:5.1f}% прогонов (±{err:.0f}), "
          f"средняя доля {100 * np.mean(shares):5.1f}%  -> {verdict}")


def duel() -> None:
    print(RULE)
    print(f"ДУЭЛЬ ЛИНИЙ: поле 48x48, плотность 0.028, популяция ~{BIG['initial_population']}")
    print(RULE)
    print()
    print("  --- КОНТРОЛЬ: типы одинаковые, обязано выйти ~50% ---")
    _report("A=0.5 vs B=0.5", 0.5, 0.5, BIG)
    _report("A=0.2 vs B=0.2", 0.2, 0.2, BIG)
    print()
    print("  --- ОПЫТ: B компетентнее ---")
    _report("A=0.2 vs B=0.3", 0.2, 0.3, BIG)
    _report("A=0.3 vs B=0.5", 0.3, 0.5, BIG)
    _report("A=0.5 vs B=0.7", 0.5, 0.7, BIG)
    print()
    print("  --- ОБРАТНЫЙ ОПЫТ: B слабее, обязано выйти НИЖЕ 50% ---")
    _report("A=0.9 vs B=0.7", 0.9, 0.7, BIG)
    print()
    print("  Если контроль ушёл от 50% — замеру нельзя верить: популяция мала,")
    print("  и дрейф пересиливает отбор. Лечится размером популяции и числом сидов,")
    print("  а не пересказом результата.")


SECTIONS = {"window": window, "duel": duel}


def main() -> None:
    names = sys.argv[1:] or list(SECTIONS)
    for name in names:
        if name not in SECTIONS:
            raise SystemExit(f"неизвестный раздел: {name}. есть: {', '.join(SECTIONS)}")
    for i, name in enumerate(names):
        if i:
            print()
        SECTIONS[name]()


if __name__ == "__main__":
    main()
