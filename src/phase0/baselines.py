"""Шесть бейзлайнов. Часть 7.

Все эмитят действия через тот же событийный контракт: пишут в outbox, когда
считают нужным, и пользуются action-hold. Различаются они ВХОДОМ, и это
различие принципиально:

  greedy_symbolic — привилегированный вход (позиции предметов). Отвечает на
                    вопрос «решаема ли задача вообще».
  greedy_pixel    — только сетчатка. Отвечает на вопрос «решаема ли она из
                    пикселей».

Смешивать эти два вопроса нельзя, поэтому оба бейзлайна обязаны быть.

Привилегированный вход — свойство ПРИБОРА, не агента. Ни один из этих классов
не является кандидатом в агенты, и канал истины по-прежнему не попадает
в inbox: символические бейзлайны получают его отдельным путём, через явный
вызов `observe_symbolic`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .events import Channels, Event, Motor, Queue
from .world import World


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class BaseAgent:
    """Общая часть: эмиссия только при изменении значения.

    Молчание — не пропуск хода, а продолжение прежнего действия, поэтому
    посылать одно и то же значение каждый тик бессмысленно и засоряет motor.log.
    """

    name = "base"

    def __init__(self, cfg: Any, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)   # RNG агента, независим от мира
        self._held = {Motor.THRUST: 0.0, Motor.TURN: 0.0}
        self._tick = 0

    def emit(self, outbox: Queue, channel: int, value: float, eps: float = 0.02) -> None:
        value = max(-1.0, min(1.0, float(value)))
        if abs(value - self._held.get(channel, 0.0)) >= eps:
            self._held[channel] = value
            outbox.put(Event(self._tick, channel, value))

    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------
class RandomAgent(BaseAgent):
    """Нижняя граница. Меняет курс изредка, иначе просто летит."""

    name = "random"

    def __init__(self, cfg: Any, seed: int = 0, change_every: int = 30) -> None:
        super().__init__(cfg, seed)
        self.change_every = change_every

    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None:
        if self._tick % self.change_every == 0:
            self.emit(outbox, Motor.THRUST, float(self.rng.uniform(-1, 1)))
            self.emit(outbox, Motor.TURN, float(self.rng.uniform(-1, 1)))
        self._tick += 1


# --------------------------------------------------------------------------
class GreedySymbolic(BaseAgent):
    """Потолок задачи. Идёт к ближайшему питательному предмету.

    Реализуется в три десятка строк и ОБЯЗАН решать A1-A3 без всякого
    обучения. Если не решает — неверны константы Части 3, а не бейзлайн.
    """

    name = "greedy_symbolic"
    symbolic = True

    def __init__(self, cfg: Any, seed: int = 0, k_p: float = 2.5,
                 k_d: float = 0.35) -> None:
        super().__init__(cfg, seed)
        self.k_p, self.k_d = k_p, k_d
        self._view: dict[str, Any] | None = None

    def observe_symbolic(self, world: World) -> None:
        """Привилегированный вход. Вызывается драйвером, не миром."""
        b = world.body
        targets = []
        for item in world.items.visible_items:
            value = world.regime.nutritive(item.kind, world.tick)
            if value > 0.0:
                targets.append((item.x, item.y))
        self._view = {
            "x": b.x, "y": b.y, "theta": b.theta, "omega": b.omega,
            "targets": targets,
        }

    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None:
        v = self._view
        self._tick += 1
        if v is None or not v["targets"]:
            self.emit(outbox, Motor.THRUST, 0.0)
            self.emit(outbox, Motor.TURN, 0.3)   # оглядеться
            return

        bx, by = v["x"], v["y"]
        tx, ty = min(v["targets"], key=lambda p: (p[0] - bx) ** 2 + (p[1] - by) ** 2)
        bearing = _wrap(math.atan2(ty - by, tx - bx) - v["theta"])

        turn = self.k_p * bearing - self.k_d * v["omega"]
        # Пока цель сильно вбок, полная тяга только уносит мимо.
        thrust = 1.0 if abs(bearing) < 0.6 else 0.15
        self.emit(outbox, Motor.TURN, turn)
        self.emit(outbox, Motor.THRUST, thrust)


# --------------------------------------------------------------------------
class GreedyPixel(BaseAgent):
    """То же намерение, но вход — только сетчатка (устойчивый канал).

    Честное сравнение с greedy_symbolic: разрыв между ними и есть цена
    того, что мир виден через пиксели, а не дан списком координат.
    """

    name = "greedy_pixel"

    def __init__(self, cfg: Any, channels: Channels, seed: int = 0,
                 k_p: float = 2.0) -> None:
        super().__init__(cfg, seed)
        self.ch = channels
        self.k_p = k_p
        self.n = cfg.sustained_n
        self.l = np.zeros(self.n)
        self.c = np.zeros(self.n)
        self.omega = 0.0

    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None:
        ch = self.ch
        for e in inbox:
            k = e.channel
            if ch.sustained_start <= k < ch.proprio_start:
                idx = k - ch.sustained_start
                recep, colour = divmod(idx, self.cfg.color_channels)
                (self.l if colour == 0 else self.c)[recep] = e.value
            elif k == ch.OMEGA:
                self.omega = e.value
        self._tick += 1

        # Питательным считаем kind=A (положительный полюс цветовой оси).
        # Это ЗАШИТО и в режиме B будет неверно ровно половину времени —
        # именно поэтому greedy_pixel не решает режим B, и это ожидаемо.
        weight = np.clip(self.c, 0.0, None) * np.clip(self.l, 0.0, None)
        if weight.sum() <= 1e-9:
            self.emit(outbox, Motor.THRUST, 0.1)
            self.emit(outbox, Motor.TURN, 0.35)
            return

        centre = (self.n - 1) / 2.0
        offset = float((weight * (np.arange(self.n) - centre)).sum() / weight.sum())
        bearing = offset / centre * math.radians(self.cfg.fov_deg) / 2.0

        self.emit(outbox, Motor.TURN, self.k_p * bearing - 0.3 * self.omega)
        self.emit(outbox, Motor.THRUST, 1.0 if abs(bearing) < 0.6 else 0.15)


# --------------------------------------------------------------------------
class LinearPixel(BaseAgent):
    """Линейная политика поверх устойчивого канала.

    Веса не обучаются градиентом: это бейзлайн «что даёт линейная функция от
    пикселей», а не «что даёт обучение». Обновление — (1+1)-случайный поиск
    по накопленной энергии за окно, чтобы не тащить сюда ни backprop, ни RL.
    """

    name = "linear_pixel"

    def __init__(self, cfg: Any, channels: Channels, seed: int = 0,
                 window: int = 1800, sigma: float = 0.25) -> None:
        super().__init__(cfg, seed)
        self.ch = channels
        self.n_in = cfg.sustained_n * cfg.color_channels + 1
        self.w = self.rng.normal(0, 0.5, (2, self.n_in))
        self.best = self.w.copy()
        self.best_score = -np.inf
        self.window, self.sigma = window, sigma
        self.x = np.zeros(self.n_in)
        self.x[-1] = 1.0
        self._score = 0.0

    def note_energy(self, energy: float) -> None:
        self._score += energy

    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None:
        ch = self.ch
        for e in inbox:
            if ch.sustained_start <= e.channel < ch.proprio_start:
                self.x[e.channel - ch.sustained_start] = e.value
        out = np.tanh(self.w @ self.x)
        self.emit(outbox, Motor.THRUST, float(out[0]))
        self.emit(outbox, Motor.TURN, float(out[1]))

        self._tick += 1
        if self._tick % self.window == 0:
            if self._score > self.best_score:
                self.best_score, self.best = self._score, self.w.copy()
            self.w = self.best + self.rng.normal(0, self.sigma, self.w.shape)
            self._score = 0.0


# --------------------------------------------------------------------------
class TabularQ(BaseAgent):
    """Классический RL на дискретизованном символическом состоянии.

    Состояние: (сектор пеленга на ближайшую еду, корзина дистанции, корзина
    энергии). Действие: одна из пяти пар (thrust, turn). Награда — прирост
    энергии за тик. Здесь награда легальна: это БЕЙЗЛАЙН, его назначение —
    показать, что даёт обычный RL, а не соблюдать ограничения агента.
    """

    name = "tabular_q"
    symbolic = True

    ACTIONS = ((1.0, 0.0), (0.6, 0.7), (0.6, -0.7), (0.0, 1.0), (0.0, -1.0))

    def __init__(self, cfg: Any, seed: int = 0, alpha: float = 0.2,
                 gamma: float = 0.95, eps: float = 0.1,
                 n_bearing: int = 8, n_dist: int = 4, n_energy: int = 4) -> None:
        super().__init__(cfg, seed)
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.dims = (n_bearing, n_dist, n_energy)
        self.q = np.zeros(self.dims + (len(self.ACTIONS),))
        self._view: dict[str, Any] | None = None
        self._prev: tuple[tuple[int, int, int], int] | None = None
        self._prev_energy = cfg.e_init

    def observe_symbolic(self, world: World) -> None:
        b = world.body
        targets = [(i.x, i.y) for i in world.items.visible_items
                   if world.regime.nutritive(i.kind, world.tick) > 0.0]
        self._view = {"x": b.x, "y": b.y, "theta": b.theta,
                      "energy": b.energy, "targets": targets}

    def _discretise(self) -> tuple[int, int, int]:
        v = self._view
        nb, nd, ne = self.dims
        if not v or not v["targets"]:
            return (0, nd - 1, min(ne - 1, int(v["energy"] * ne) if v else 0))
        bx, by = v["x"], v["y"]
        tx, ty = min(v["targets"], key=lambda p: (p[0] - bx) ** 2 + (p[1] - by) ** 2)
        bearing = _wrap(math.atan2(ty - by, tx - bx) - v["theta"])
        dist = math.hypot(tx - bx, ty - by)
        b_idx = int((bearing + math.pi) / (2 * math.pi) * nb) % nb
        d_idx = min(nd - 1, int(dist / 16.0 * nd))
        e_idx = min(ne - 1, max(0, int(v["energy"] * ne)))
        return (b_idx, d_idx, e_idx)

    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None:
        s = self._discretise()
        energy = self._view["energy"] if self._view else self._prev_energy

        if self._prev is not None:
            ps, pa = self._prev
            reward = energy - self._prev_energy
            target = reward + self.gamma * float(self.q[s].max())
            self.q[ps][pa] += self.alpha * (target - self.q[ps][pa])

        if self.rng.random() < self.eps:
            a = int(self.rng.integers(len(self.ACTIONS)))
        else:
            a = int(np.argmax(self.q[s]))

        thrust, turn = self.ACTIONS[a]
        self.emit(outbox, Motor.THRUST, thrust)
        self.emit(outbox, Motor.TURN, turn)

        self._prev, self._prev_energy = (s, a), energy
        self._tick += 1


# --------------------------------------------------------------------------
class SmallRnnBptt(BaseAgent):
    """Маленькая RNN с backprop through time. Архитектура-конкурент.

    Учится предсказывать СЛЕДУЮЩИЙ устойчивый кадр (self-supervised), а
    политика берётся линейной поверх скрытого состояния и подстраивается тем
    же (1+1)-поиском, что и linear_pixel. Так бейзлайн остаётся честным
    конкурентом по представлению, не превращаясь в полноценный RL-стек.

    Numpy без torch: torch в контейнере не ставится, а тащить зависимость
    ради 60 строк BPTT незачем.
    """

    name = "small_rnn_bptt"

    def __init__(self, cfg: Any, channels: Channels, seed: int = 0,
                 hidden: int = 24, bptt: int = 16, lr: float = 1e-2,
                 window: int = 1800, sigma: float = 0.25) -> None:
        super().__init__(cfg, seed)
        self.ch = channels
        self.n_in = cfg.sustained_n * cfg.color_channels
        self.h_size = hidden
        self.bptt, self.lr = bptt, lr
        r = self.rng
        self.wx = r.normal(0, 1 / np.sqrt(self.n_in), (hidden, self.n_in))
        self.wh = np.eye(hidden) * 0.9 + r.normal(0, 0.01, (hidden, hidden))
        self.wo = r.normal(0, 1 / np.sqrt(hidden), (self.n_in, hidden))
        self.policy = r.normal(0, 0.5, (2, hidden + 1))
        self.best, self.best_score = self.policy.copy(), -np.inf
        self.window, self.sigma = window, sigma
        self.h = np.zeros(hidden)
        self._hist: list[tuple[np.ndarray, np.ndarray]] = []
        self.x = np.zeros(self.n_in)
        self._score = 0.0
        self.pred_loss = 0.0

    def note_energy(self, energy: float) -> None:
        self._score += energy

    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None:
        ch = self.ch
        prev_x = self.x.copy()
        for e in inbox:
            if ch.sustained_start <= e.channel < ch.proprio_start:
                self.x[e.channel - ch.sustained_start] = e.value

        self.h = np.tanh(self.wx @ prev_x + self.wh @ self.h)
        self._hist.append((prev_x.copy(), self.h.copy()))
        pred = self.wo @ self.h
        err = pred - self.x
        self.pred_loss = 0.9 * self.pred_loss + 0.1 * float(err @ err)

        self.wo -= self.lr * np.outer(err, self.h)
        if len(self._hist) >= self.bptt:
            self._bptt_update(err)
            self._hist = self._hist[-1:]

        feat = np.concatenate([self.h, [1.0]])
        out = np.tanh(self.policy @ feat)
        self.emit(outbox, Motor.THRUST, float(out[0]))
        self.emit(outbox, Motor.TURN, float(out[1]))

        self._tick += 1
        if self._tick % self.window == 0:
            if self._score > self.best_score:
                self.best_score, self.best = self._score, self.policy.copy()
            self.policy = self.best + self.rng.normal(0, self.sigma, self.policy.shape)
            self._score = 0.0

    def _bptt_update(self, err: np.ndarray) -> None:
        """Развернуть по времени и прокатить градиент назад."""
        dh = self.wo.T @ err
        gwx = np.zeros_like(self.wx)
        gwh = np.zeros_like(self.wh)
        for k in range(len(self._hist) - 1, 0, -1):
            x_k, h_k = self._hist[k]
            h_prev = self._hist[k - 1][1]
            raw = dh * (1.0 - h_k ** 2)
            gwx += np.outer(raw, x_k)
            gwh += np.outer(raw, h_prev)
            dh = self.wh.T @ raw
            n = np.linalg.norm(dh)
            if n > 5.0:                      # обрезка, иначе BPTT разносит
                dh *= 5.0 / n
        for g in (gwx, gwh):
            n = np.linalg.norm(g)
            if n > 5.0:
                g *= 5.0 / n
        self.wx -= self.lr * gwx
        self.wh -= self.lr * gwh


BASELINES = {
    c.name: c for c in
    (RandomAgent, GreedySymbolic, GreedyPixel, LinearPixel, TabularQ, SmallRnnBptt)
}
