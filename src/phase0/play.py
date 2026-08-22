"""Играбельность человеком. Часть 10.4.

Даёт человеческий бейзлайн бесплатно и — важнее — обнаруживает бедность или
неиграбельность сенсорного интерфейса за пять минут игры, а не за месяц
отладки агента. Если человек не может играть по этой сетчатке, агент тем
более не сможет, и это не его вина.

Человек подключён через ТОТ ЖЕ событийный контракт: клавиши превращаются в
события в outbox с той же семантикой удержания. Никакого особого пути в мир
у человека нет.

Два режима:
  terminal — работает везде, включая ssh без дисплея. Стрелки/WASD.
  arcade   — графический, если библиотека установлена.

Флаг --blind рисует ТОЛЬКО сетчатку. Это и есть настоящая проверка шага 0.6:
пройти на виде сверху может кто угодно, а вот выжить, глядя только на полосу
рецепторов, — уже вопрос к качеству сенсорного тракта.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty

from .config import Config, Mode
from .events import Event, Motor
from .render import frame, retina_view
from .world import World


class KeyReader:
    """Неблокирующее чтение клавиш из терминала."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old: list | None = None

    def __enter__(self) -> "KeyReader":
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self) -> list[str]:
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = os.read(self.fd, 1).decode("utf-8", "ignore")
            if ch == "\x1b":  # escape-последовательность стрелок
                rest = os.read(self.fd, 2).decode("utf-8", "ignore")
                keys.append({"[A": "up", "[B": "down", "[C": "right", "[D": "left"}
                            .get(rest, "esc"))
            else:
                keys.append(ch.lower())
        return keys


def play_terminal(cfg: Config, fps: float = 20.0, blind: bool = False) -> None:
    world = World(cfg)
    trail: list[tuple[float, float]] = []
    thrust = turn = 0.0
    record = None
    period = 1.0 / fps
    # Мир тикает на TICK_HZ, экран обновляется реже: рисовать 60 раз в секунду
    # в терминал бессмысленно, а физику замедлять нельзя.
    ticks_per_frame = max(1, int(round(cfg.tick_hz / fps)))

    print("\x1b[2J", end="")
    with KeyReader() as keys:
        try:
            while True:
                t0 = time.perf_counter()
                for k in keys.poll():
                    if k in ("q", "esc"):
                        return
                    if k in ("w", "up"):
                        thrust = 1.0
                    elif k in ("s", "down"):
                        thrust = -1.0
                    elif k in ("a", "left"):
                        turn = -1.0
                    elif k in ("d", "right"):
                        turn = 1.0
                    elif k == " ":
                        thrust = turn = 0.0
                    world.outbox.put(Event(world.tick, Motor.THRUST, thrust))
                    world.outbox.put(Event(world.tick, Motor.TURN, turn))

                for _ in range(ticks_per_frame):
                    record = world.step()
                    world.inbox.drain()
                    trail.append((world.body.x, world.body.y))

                view = (retina_view(world, record) if blind
                        else frame(world, record, trail))
                status = (f"E={world.body.energy:.3f}  смертей={world.deaths}  "
                          f"тик={world.tick}")
                print("\x1b[H" + view + "\n\n" + status +
                      "\nWASD/стрелки — движение, пробел — нейтраль, q — выход",
                      flush=True)

                sleep = period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            return


def play_arcade(cfg: Config) -> None:
    """Графический вариант — НЕ РЕАЛИЗОВАН.

    Честнее сказать это прямо, чем оставить непроверенный код: arcade в
    контейнере, где писался мир, не устанавливается, и отладить графический
    путь было негде. Терминальный вид даёт все три обязательных вида Части
    10.4 и работает без дисплея, включая ssh.
    """
    raise SystemExit(
        "Графический вид пока не реализован (arcade недоступен там, где мир "
        "писался, и непроверенный код тут хуже его отсутствия).\n"
        "Терминальный вид умеет всё то же:\n"
        "  python3 -m phase0.play --view terminal\n"
        "  python3 -m phase0.play --view terminal --blind   # только сетчатка"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Играть в мир Фазы 0 руками")
    ap.add_argument("--view", choices=("terminal", "arcade"), default="terminal")
    ap.add_argument("--mode", choices=("A", "B", "C"), default="A")
    ap.add_argument("--difficulty", default="A1")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--blind", action="store_true",
                    help="показывать ТОЛЬКО сетчатку — настоящая проверка 0.6")
    args = ap.parse_args(argv)

    cfg = Config(mode=Mode[args.mode], difficulty=args.difficulty)
    if args.view == "arcade":
        play_arcade(cfg)
    else:
        play_terminal(cfg, fps=args.fps, blind=args.blind)


if __name__ == "__main__":
    main()
