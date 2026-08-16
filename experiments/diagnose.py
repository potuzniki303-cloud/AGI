"""Замеры к разделу 5 CLAUDE.md. Каждое число здесь можно перепроверить.

    python3 experiments/diagnose.py            # всё
    python3 experiments/diagnose.py economy    # экономика мира
    python3 experiments/diagnose.py weights    # динамика весов
    python3 experiments/diagnose.py actions    # выбор действия

Зачем отдельным файлом. В этом проекте легко построить убедительную теорию,
которая не выдерживает первого же прогона. Правило: сначала замер, потом вывод.

Про numpy-двойник агента ниже. torch в контейнере ассистента не ставится
(в pyproject.toml прибиты ROCm-колёса под Windows), поэтому динамика весов
проверяется точной numpy-копией той же арифметики. Копия — НЕ агент и не
предложение, как писать агента; это измерительный прибор. Если ты меняешь
src/agent/, поменяй и её, иначе замеры начнут врать.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world import make  # noqa: E402
from world.stubs import GreedyMind, RandomMind  # noqa: E402

RULE = "=" * 76


# --------------------------------------------------------------------------
# Общее: набрать реальные наблюдения из мира, чтобы мерить не на выдуманном входе
# --------------------------------------------------------------------------
def collect_observations(world_name: str = "forage", ticks: int = 400) -> np.ndarray:
    world = make(world_name)
    world.reset(lambda rng: GreedyMind(world.action_size, rng))
    rows: list[np.ndarray] = []
    for _ in range(ticks):
        world.step()
        _ids, obs = world.observe_all()
        rows.extend(obs)
        if not world.bodies:
            break
    return np.array(rows, dtype=np.float64)


# --------------------------------------------------------------------------
# 1. Экономика
# --------------------------------------------------------------------------
def _random_walk_income(density: float, steps: int, size: int, food_energy: float,
                        seed: int = 0) -> float:
    """Доход случайного ходока при ПОДДЕРЖИВАЕМОЙ плотности еды.

    Съеденное сразу возвращается в случайную пустую клетку. Это заведомо
    щедрее настоящего мира (там отрастание ограничено regrow_per_step), так
    что полученное число — верхняя оценка.
    """
    rng = np.random.default_rng(seed)
    grid = rng.random((size, size)) < density
    x = y = size // 2
    eaten = 0
    for _ in range(steps):
        d = rng.integers(4)
        if d == 0:
            y = (y - 1) % size
        elif d == 1:
            y = (y + 1) % size
        elif d == 2:
            x = (x - 1) % size
        else:
            x = (x + 1) % size
        if grid[y, x]:
            grid[y, x] = False
            eaten += 1
            while True:
                ry, rx = rng.integers(size), rng.integers(size)
                if not grid[ry, rx]:
                    grid[ry, rx] = True
                    break
    return eaten / steps * food_energy


def _distinct_cells(steps: int, trials: int, size: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(trials):
        x = y = size // 2
        seen = {(x, y)}
        for _ in range(steps):
            d = rng.integers(4)
            if d == 0:
                y = (y - 1) % size
            elif d == 1:
                y = (y + 1) % size
            elif d == 2:
                x = (x - 1) % size
            else:
                x = (x + 1) % size
            seen.add((x, y))
        totals.append(len(seen))
    return float(np.mean(totals))


def economy() -> None:
    print(RULE)
    print("1. ЭКОНОМИКА: почему формула 'плотность x food_energy' завышает доход")
    print(RULE)
    size = 40
    print()
    print("  Формула молча считает, что каждый шаг ходок пробует НОВУЮ клетку.")
    print(f"  Случайное блуждание по 2D так не делает (тор {size}x{size}):")
    print()
    print("    шагов | разных клеток | новых клеток за шаг")
    for steps in (100, 400, 1600):
        d = _distinct_cells(steps, trials=200, size=size)
        print(f"    {steps:5d} | {d:13.1f} | {d / steps:.2f}")

    print()
    print("  Прямой замер дохода (еда мгновенно восстанавливается — верхняя оценка):")
    print()
    print("    плотность | формула | замер | замер/формула")
    for dens in (0.028, 0.040, 0.057, 0.100):
        formula = dens * 25.0
        real = _random_walk_income(dens, steps=20000, size=size, food_energy=25.0)
        print(f"    {dens:9.3f} | {formula:7.3f} | {real:5.3f} | {real / formula:.2f}")

    lo, hi = 0.0, 0.5
    for _ in range(30):
        mid = (lo + hi) / 2
        if _random_walk_income(mid, steps=40000, size=size, food_energy=25.0) < 1.0:
            lo = mid
        else:
            hi = mid
    real_break_even = (lo + hi) / 2
    print()
    print(f"    формула обещает безубыточность на плотности {1.0 / 25.0:.3f}"
          f" ({1.0 / 25.0 * size ** 2:.0f} еды)")
    print(f"    ЗАМЕР даёт безубыточность на плотности      {real_break_even:.3f}"
          f" ({real_break_even * size ** 2:.0f} еды)")
    print(f"    -> формула ошибается в {real_break_even / (1.0 / 25.0):.1f} раза")

    print()
    print("  Что это значит для настроенных миров (линейка Random / Greedy):")
    print()
    print("    мир                              | Random         | Greedy")
    configs = [
        ("forage (дефолт)", "forage", {}),
        ("bounty из main.py", "bounty",
         dict(size=40, seed=42, initial_population=30, max_food=10, energy_at_birth=80)),
    ]
    for label, world_name, cfg in configs:
        cells = []
        for mind_cls in (RandomMind, GreedyMind):
            world = make(world_name, **cfg)
            hist = _simulate(world, mind_cls, steps=3000)
            spans = hist.lifespans
            med = float(np.median(spans)) if spans else 0.0
            ext = hist.extinct_at
            cells.append(f"жизнь~{med:5.0f} "
                         + (f"ВЫМЕРЛИ@{ext}" if ext else f"жив,pop={hist.population[-1]}"))
        print(f"    {label:32s} | {cells[0]:14s} | {cells[1]}")
    print()
    print("  Вывод: в конфиге из main.py плотность 10/1600 = 0.006 — это в 16 раз")
    print("  ниже настоящей точки безубыточности. Разгон bounty какое-то время")
    print("  держит популяцию, но как только он спадает, выжить нельзя никак.")


def _simulate(world, mind_cls, steps: int):
    from world import simulate
    return simulate(world, lambda rng: mind_cls(world.action_size, rng), steps=steps)


# --------------------------------------------------------------------------
# 2. Динамика весов: numpy-двойник src/agent/
# --------------------------------------------------------------------------
class _Layer:
    """Копия src/agent/layer.py + Genome.forward(). Измерительный прибор."""

    def __init__(self, n_in: int, n_out: int, genome: np.ndarray,
                 rng: np.random.Generator) -> None:
        k = 1.0 / np.sqrt(n_in)  # инициализация nn.Linear
        self.W = rng.uniform(-k, k, (n_out, n_in))
        self.bias = rng.uniform(-k, k, n_out)
        self.genome = genome

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = self.W @ x + self.bias  # y — ДО нелинейности, как в оригинале
        a, b, c, d, lr = self.genome
        self.W += lr * (a * np.outer(y, x) + b * x[None, :] + c * y[:, None] + d)
        return y


def _random_genome(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.normal(0, 0.5), rng.normal(0, 0.5), rng.normal(0, 0.5),
                     rng.normal(0, 0.5), abs(rng.normal(1e-3, 5e-4))])


class _Agent:
    """Копия src/agent/agent.py: fc1 -> fc2 БЕЗ нелинейности между слоями."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 hidden: int = 150, inner_tanh: bool = False,
                 g1: np.ndarray | None = None, g2: np.ndarray | None = None) -> None:
        self.fc1 = _Layer(n_in, hidden, _random_genome(rng) if g1 is None else g1, rng)
        self.fc2 = _Layer(hidden, n_out, _random_genome(rng) if g2 is None else g2, rng)
        self.inner_tanh = inner_tanh

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.fc1.forward(x)
        if self.inner_tanh:
            h = np.tanh(h)
        return self.fc2.forward(h)

    def peak(self) -> tuple[float, float]:
        return float(np.abs(self.fc1.W).max()), float(np.abs(self.fc2.W).max())


def weights() -> None:
    print(RULE)
    print("2. ВЕСА: правило ABCD аффинно по W, третьего режима у него нет")
    print(RULE)
    obs = collect_observations()
    x2_first = float(np.mean((obs ** 2).sum(1)))
    print()
    print(f"  Реальные наблюдения forage: {len(obs)} шт, "
          f"диапазон [{obs.min():.3f}, {obs.max():.3f}]")

    print()
    print("  Δw = η(a·yxᵀ + b·x + c·y + d),  y = Wx + β")
    print("  Член a даёт  Δw ⊃ η·a·W·(xxᵀ)  ->  W(t) ~ W(0)·exp(η·a·‖x‖²·t)")
    print("  Показатель роста пропорционален ‖x‖² ВХОДА ЭТОГО СЛОЯ:")
    print()

    rng = np.random.default_rng(3)
    probe = _Agent(obs.shape[1], 4, rng)
    hs = np.array([probe.fc1.forward(obs[t]) for t in range(200)])
    x2_second = float(np.mean((hs ** 2).sum(1)))
    print(f"    fc1: вход — {obs.shape[1]} сенсоров из [0,1]      ‖x‖² = {x2_first:8.3f}")
    print(f"    fc2: вход — {hs.shape[1]} выходов fc1, без границ  ‖x‖² = {x2_second:8.3f}")
    print(f"    -> fc2 разгоняется в {x2_second / x2_first:.0f} раз быстрее при том же геноме")
    print()
    print("    при a=0.5, η=1e-3 время удвоения весов:")
    for name, x2 in (("fc1", x2_first), ("fc2", x2_second)):
        rate = 1e-3 * 0.5 * x2
        print(f"      {name}: {np.log(2) / rate:8.0f} шагов")
    print()
    print("    Причина такого разрыва: между fc1 и fc2 сейчас НЕТ нелинейности,")
    print("    поэтому вход второго слоя ничем не ограничен и растёт вместе с")
    print("    весами первого. Заодно два линейных слоя подряд математически")
    print("    равны одному — 150 скрытых нейронов не добавляют выразительности.")

    print()
    print("  Сколько случайных геномов разносит веса за 2000 шагов в реальном forage:")
    print()
    for inner_tanh, label in ((False, "как сейчас (без tanh между слоями)"),
                              (True, "с tanh между слоями")):
        blown, finals = 0, []
        for i in range(60):
            r = np.random.default_rng(500 + i)
            agent = _Agent(obs.shape[1], 4, r, inner_tanh=inner_tanh)
            with np.errstate(over="ignore", invalid="ignore"):
                for t in range(2000):
                    agent.forward(obs[t % len(obs)])
            peak = agent.peak()[1]
            if not np.isfinite(peak) or peak > 1e6:
                blown += 1
            else:
                finals.append(peak)
        med = float(np.median(finals)) if finals else float("nan")
        print(f"    {label:36s}: разошлось {blown:2d}/60, медиана max|W2| = {med:.4g}")
    print()
    print("    tanh между слоями помогает, но не лечит: источник роста в структуре")
    print("    правила, а не в масштабе. Все члены не выше первой степени по W,")
    print("    а аффинная динамика умеет только расти экспоненциально или сходиться")
    print("    в точку. Режима «ограничен, но продолжает учиться» здесь нет ни при")
    print("    каком наборе знаков. Лечится членом степени 3 — например Ойей")
    print("    −e·y²·w, где e должен быть ПЯТЫМ ГЕНОМ, а не константой дизайнера.")


# --------------------------------------------------------------------------
# 3. Выбор действия
# --------------------------------------------------------------------------
def _softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = z / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def actions() -> None:
    print(RULE)
    print("3. ДЕЙСТВИЕ: tanh на логитах ставит потолок на осмысленность поведения")
    print(RULE)
    obs = collect_observations()
    antenna = obs[:, :4]
    live = antenna[antenna.max(1) > 0]
    best = live.argmax(1)

    print()
    print(f"  Сигнал антенн: максимум медиана {np.median(live.max(1)):.3f}, "
          f"разрыв лучшая-минус-вторая медиана "
          f"{np.median(np.sort(live, 1)[:, -1] - np.sort(live, 1)[:, -2]):.3f}")
    print("  Направление кодируется именно этим разрывом — он крошечный,")
    print("  поэтому логиты обязаны быть усилены, иначе softmax почти равномерен.")
    print()
    print("  Берём ИДЕАЛЬНУЮ сеть-таксис (логиты = антенны x усиление) и смотрим,")
    print("  как часто она выберет лучшее направление. 25% = случайно, 100% = Greedy.")
    print()
    print("    усиление | softmax(tanh(z)) | softmax(z)")
    for gain in (1, 5, 20, 100):
        z = live * gain
        with_tanh = _softmax(np.tanh(z))[np.arange(len(best)), best].mean()
        without = _softmax(z)[np.arange(len(best)), best].mean()
        print(f"    {gain:8d} | {100 * with_tanh:15.1f}% | {100 * without:9.1f}%")

    print()
    print("  Кривая с tanh НЕ МОНОТОННА: после усиления ~5 рост весов делает")
    print("  поведение хуже и возвращает его к случайному. Механизм:")
    print()
    row = live[np.argsort(live.max(1))[len(live) // 2]]
    for gain in (10, 50):
        z = row * gain
        t = np.tanh(z)
        srt = np.sort(t)
        print(f"    усиление {gain:3d}: {np.array2string(z, precision=2)}"
              f" -> tanh -> {np.array2string(t, precision=3)}"
              f"  разрыв {srt[-1] - srt[-2]:.4f}")
    print()
    print("  tanh сжимает всё в (-1,1). Чем больше веса, тем сильнее ВСЕ ненулевые")
    print("  логиты прилипают к 1, и разрыв между ними — единственное, что несёт")
    print("  информацию о направлении — схлопывается. Информация уничтожается ДО")
    print("  softmax, поэтому температура её не возвращает:")
    print()
    print("      T   |", "".join(f"{g:>8d}" for g in (1, 5, 20, 100)), " <- усиление")
    for temp in (1.0, 0.25, 0.05):
        vals = [100 * _softmax(np.tanh(live * g), temp)[np.arange(len(best)), best].mean()
                for g in (1, 5, 20, 100)]
        print(f"    {temp:5.2f} |", "".join(f"{v:7.1f}%" for v in vals), " с tanh")
    for temp in (1.0, 0.25, 0.05):
        vals = [100 * _softmax(live * g, temp)[np.arange(len(best)), best].mean()
                for g in (1, 5, 20, 100)]
        print(f"    {temp:5.2f} |", "".join(f"{v:7.1f}%" for v in vals), " без tanh")

    print()
    print("  Почему это критично: эволюция может забраться только на ту гору,")
    print("  у которой есть склон. Если «выучить больше» перестаёт быть «выжить")
    print("  лучше» после какого-то момента, отбор упирается в потолок.")

    print()
    print("  ТИСКИ. Осмысленное поведение требует усиления ~20-50x от старта,")
    print("  а растит его то же экспоненциальное правило. Шаг, когда |логиты|")
    print("  впервые > 2, и шаг, когда они > 1e6:")
    print()
    print("    геном a  | стал неслучайным | разнесло")
    for i in range(6):
        r = np.random.default_rng(200 + i)
        g1, g2 = _random_genome(r), _random_genome(r)
        agent = _Agent(obs.shape[1], 4, r, g1=g1, g2=g2)
        useful = blown = None
        with np.errstate(over="ignore", invalid="ignore"):
            for t in range(20000):
                peak = np.abs(agent.forward(obs[t % len(obs)])[:4]).max()
                if useful is None and peak > 2:
                    useful = t
                if not np.isfinite(peak) or peak > 1e6:
                    blown = t
                    break
        print(f"    a={g1[0]:6.2f} | {str(useful):>16s} | {str(blown)}")
    print()
    print("  Окно между «ещё случайный» и «уже сломан» определяется удачей генома,")
    print("  а не обучением. Это и есть «математика ломается».")


SECTIONS = {"economy": economy, "weights": weights, "actions": actions}


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
