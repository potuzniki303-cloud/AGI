"""Прогон всех миров на заглушках.

Смысл не в результатах агентов (агентов тут нет), а в калибровке физики.
Смотреть надо на две вещи:

  1. RandomMind не должен процветать. Если случайные ходоки держат
     стабильную популяцию, в мире нет давления, и эволюции не за что
     зацепиться.
  2. RandomMind не должен вымирать за десяток тактов. Если вымирает,
     ни одно правило не успеет ничего найти до вымирания популяции.

GreedyMind — верхняя отметка: так выглядит мир, если таксис уже найден.
Разрыв между Random и Greedy это и есть место, в котором живёт эволюция.

    python experiments/demo_worlds.py
    python experiments/demo_worlds.py --watch forage
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from world import list_worlds, make, print_frame, simulate  # noqa: E402
from world.stubs import GreedyMind, RandomMind  # noqa: E402

STEPS = 3000


def bench(name: str, steps: int = STEPS, seed: int = 1) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    world = make(name, seed=seed)
    print(f"вход: {world.observation_size}   действий: {world.action_size}")
    print(f"сенсор: {world.sensor!r}")

    for label, factory in (
        ("random", lambda rng, w=world: RandomMind(w.action_size, rng)),
        ("greedy", lambda rng, w=world: GreedyMind(w.action_size, rng)),
    ):
        w = make(name, seed=seed)
        t0 = time.perf_counter()
        h = simulate(w, factory, steps=steps)
        dt = time.perf_counter() - t0
        print(f"\n  [{label}] {h.summary()}")
        print(f"  {steps} тактов за {dt:.1f}с")


def watch(name: str, steps: int = 400, fps: float = 12.0, seed: int = 1) -> None:
    world = make(name, seed=seed)
    factory = lambda rng: GreedyMind(world.action_size, rng)  # noqa: E731
    world.reset(factory)
    for _ in range(steps):
        world.step()
        print_frame(world)
        if world.extinct:
            break
        time.sleep(1.0 / fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", metavar="WORLD", help="смотреть мир вживую")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.watch:
        watch(args.watch, steps=min(args.steps, 600), seed=args.seed)
        return

    for name in list_worlds():
        bench(name, steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
