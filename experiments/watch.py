"""Живой просмотр мира в окне arcade.

    python3 experiments/watch.py --world forage
    python3 experiments/watch.py --world shift --size 16 --fps 8

На машине без дисплея (сервер, контейнер) — под виртуальным:

    xvfb-run -a python3 experiments/watch.py --world forage --frames 200 --save out.png

По умолчанию показывает заглушку GreedyMind: так выглядит мир, в котором
таксис уже найден. Это верхняя отметка для сравнения, а не то, что должен
выучить агент.

Что видно на экране:
  яркость тела  — энергия, тусклое вот-вот умрёт
  обводка       — поколение, чем светлее тем позже родился
  красная рамка — телу сломали сенсомоторику (мир shift)
  цвет кружка   — тип еды (в two_foods их два)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gymnasium as gym  # noqa: E402

from world import list_worlds, register_all  # noqa: E402
from world.gym_env import gym_id  # noqa: E402
from world.stubs import GreedyMind, RandomMind  # noqa: E402

STUBS = {"greedy": GreedyMind, "random": RandomMind}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="forage", choices=list_worlds())
    ap.add_argument("--mind", default="greedy", choices=sorted(STUBS))
    ap.add_argument("--frames", type=int, default=100000)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--save", help="сохранить последний кадр в PNG и выйти")
    args = ap.parse_args()

    register_all()
    mode = "rgb_array" if args.save else "human"
    env = gym.make(gym_id(args.world), size=args.size, seed=args.seed, render_mode=mode)
    env.unwrapped.metadata["render_fps"] = args.fps

    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    # Разумы держит вызывающий — мир их не хранит и не создаёт.
    stub = STUBS[args.mind]
    minds = {i: stub(env.unwrapped.world.action_size, rng) for i in info["ids"]}

    try:
        for _ in range(args.frames):
            actions = [minds[i].act(o) for i, o in zip(info["ids"], obs)]
            obs, _, terminated, truncated, info = env.step(actions)

            for parent, child in info["born"]:
                minds[child] = minds[parent].spawn(rng)
            for i in info["died"]:
                minds.pop(i, None)

            if terminated:
                print(f"вымерли на такте {info['tick']}")
                break
            if truncated:
                break
    except KeyboardInterrupt:
        print("\nостановлено")

    if args.save:
        from PIL import Image

        Image.fromarray(env.render()).save(args.save)
        print(f"кадр сохранён: {args.save}")

    print(
        f"t={info['tick']} популяция={info['population']} "
        f"еда={info['food']} "
        f"поколение={int(info['generation'].max()) if info['population'] else 0}"
    )
    env.close()


if __name__ == "__main__":
    main()
