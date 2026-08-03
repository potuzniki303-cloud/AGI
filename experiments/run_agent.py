"""Запуск твоего Agent в мире.

ВНИМАНИЕ: в контейнере, где это писалось, torch не ставится (в pyproject
прибиты ROCm-колёса под Windows), поэтому скрипт проверен только на заглушке,
имитирующей интерфейс forward()/mutate(). Сам переходник покрыт тестами,
а вот эти несколько строк с torch — нет. Первый запуск у тебя.

Чтобы это заработало, в src/agent/agent.py нужно поменять одну вещь:

    class Agent(nn.Module):
        def __init__(self, grid_size: int, device) -> None:
            self.fc1 = Layer(int(15 + grid_size ** 2 + 1), 15, device=device)

Размер входа больше не выводится из grid_size. Мир сам говорит, сколько у
него сенсоров — world.observation_size, и это 5, а не 41. Конструктор должен
принимать размер входа напрямую:

    def __init__(self, n_inputs: int, n_actions: int, hidden: int, device):
        self.fc1 = Layer(n_inputs + hidden, hidden, device=device)
        self.fc2 = Layer(hidden, n_actions + hidden, device=device)

Второе, не обязательное: mutate() у тебя дёргает глобальный random, поэтому
прогон не воспроизводится по seed мира. Мир передаёт rng в spawn() как раз
для этого — если протянешь его в Genome.mutate(), эксперименты станут
повторяемыми.

    python3 experiments/run_agent.py --world forage --steps 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from world import list_worlds, make, simulate  # noqa: E402
from world.adapters import MindAdapter  # noqa: E402

HIDDEN = 15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="forage", choices=list_worlds())
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch  # локальный импорт: мир сам по себе в torch не нуждается

    from agent import Agent  # noqa: F401  (твой код)

    world = make(args.world, seed=args.seed)
    print(world.describe())

    device = torch.device(args.device)

    def to_input(obs):
        return torch.from_numpy(obs).to(device)

    def factory(rng):
        agent = Agent(
            n_inputs=world.observation_size,
            n_actions=world.action_size,
            hidden=HIDDEN,
            device=device,
        )
        return MindAdapter(
            agent,
            world.action_size,
            rng,
            to_input=to_input,
            temperature=args.temperature,
        )

    history = simulate(world, factory, steps=args.steps)
    print(history.summary())

    # Сравнивать надо не с нулём, а с отметками из demo_worlds.py:
    # RandomMind снизу, GreedyMind сверху. Разрыв между ними — это и есть
    # место, в котором должно оказаться правило пластичности.


if __name__ == "__main__":
    main()
