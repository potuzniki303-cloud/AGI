"""Шесть бейзлайнов. Часть 7.

Все эмитят действия через тот же событийный контракт: пишут в outbox, когда
считают нужным, и пользуются action-hold. Различаются они ВХОДОМ, и это
различие принципиально:

  greedy_symbolic  — привилегированный вход (позиции). «Решаема ли задача
                     вообще?»
  greedy_sustained — устойчивый канал, 8 рецепторов по 15 градусов.
  greedy_transient — событийный канал, 32 рецептора по 3.75 градуса.

Смешивать эти вопросы нельзя, поэтому все три бейзлайна обязаны быть.

Пара sustained/transient появилась не для симметрии. До неё 128 каналов из
150 не читал ни один бейзлайн: событийная сетчатка, центральная часть Части 4,
не проверялась ничем, и было неизвестно, пригодна ли она в принципе. Заодно
это чинит вывод про «цену пикселей»: один только устойчивый канал видит
вчетверо грубее самой сетчатки, и разрыв по нему завышен.

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

    def subscription(self, channels: Channels) -> set[int] | None:
        """Какие сенсорные каналы бейзлайн себе выписывает.

        None означает «все». Важно при METABOLIC_COMPUTE с основой "input":
        невыписанные каналы не эмитятся и не оплачиваются, то есть отказ от
        канала — физически возможное решение, а не риторика.
        """
        return None

    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None:
        raise NotImplementedError


def _range(start: int, count: int) -> set[int]:
    return set(range(start, start + count))


# --------------------------------------------------------------------------
class RandomAgent(BaseAgent):
    """Нижняя граница. Меняет курс изредка, иначе просто летит."""

    name = "random"

    def subscription(self, channels: Channels) -> set[int]:
        return set()          # не читает ничего

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

    def subscription(self, channels: Channels) -> set[int]:
        return set()          # привилегированный вход, сенсорика не нужна

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
class GreedySustained(BaseAgent):
    """Вход — только УСТОЙЧИВЫЙ канал: 8 рецепторов по 15 градусов каждый.

    Имя важно. Раньше он назывался greedy_pixel, и это вводило в заблуждение:
    разрыв с greedy_symbolic списывался на «цену того, что мир виден через
    сетчатку», хотя сетчатка у мира 32 рецептора по 3.75 градуса, а этот
    бейзлайн видит вчетверо грубее. Цену пикселей показывает пара
    greedy_sustained / greedy_transient, а не один из них.
    """

    name = "greedy_sustained"

    def subscription(self, channels: Channels) -> set[int]:
        return _range(channels.sustained_start, channels.n_sustained) | {channels.OMEGA}

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
        # именно поэтому пиксельные жадины не решают режим B, и это ожидаемо.
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

    def subscription(self, channels: Channels) -> set[int]:
        return _range(channels.sustained_start, channels.n_sustained)

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

    def subscription(self, channels: Channels) -> set[int]:
        return set()          # привилегированный вход

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

    def subscription(self, channels: Channels) -> set[int]:
        return _range(channels.sustained_start, channels.n_sustained)

    # НА KILL CRITERION ШАГА 0.8 ЭТОТ БЕЙЗЛАЙН НЕ ОТВЕЧАЕТ.
    # BPTT здесь настоящий, но обучает только предсказатель следующего кадра.
    # ПОЛИТИКА подбирается (1+1)-случайным поиском с окном 1800 тиков, то есть
    # меньше одного решения «принять/отвергнуть» на переворот режима. Плоская
    # кривая T_adapt(k) у него докажет, что случайный поиск медленнее среды, а
    # НЕ что backprop страдает катастрофическим забыванием. Для проверки
    # заявления Части VI нужен конкурент с настоящим policy gradient.
    answers_reversal_kill_criterion = False

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


class GreedyTransient(BaseAgent):
    """Вход — ТОЛЬКО транзиентный канал: 128 событийных каналов, 32 рецептора.

    Зачем он нужен. До него событийная сетчатка — центральная часть Части 4 —
    не проверялась ни одним бейзлайном: 128 каналов из 150 не были покрыты
    ни одним замером, и было неизвестно, пригодны ли они в принципе.

    Как работает. Транзиентное событие сообщает ЗНАК и величину изменения
    сигнала в рецепторе. Складывая их, сигнал восстанавливается — это ровно
    то, что делают с потоком DVS-камеры:

        est[i, c] += sign * value

    Утечка к нулю нужна потому, что интегратор без неё копит ошибку
    отброшенных по EVENT_RATE_CAP событий. Она же делает оценку забывчивой:
    то, что давно не менялось, растворяется. Это честная цена событийного
    кодирования — статичная сцена не даёт входа, значит и помнить её нечем.
    """

    name = "greedy_transient"

    def subscription(self, channels: Channels) -> set[int]:
        return _range(channels.transient_start, channels.n_transient) | {channels.OMEGA}

    def __init__(self, cfg: Any, channels: Channels, seed: int = 0,
                 k_p: float = 2.0, leak: float = 0.002) -> None:
        super().__init__(cfg, seed)
        self.ch = channels
        self.k_p = k_p
        self.leak = leak
        self.n = cfg.retina_n
        self.est = np.zeros((self.n, cfg.color_channels), dtype=np.float64)
        self.omega = 0.0

    def step(self, inbox: list[Event], outbox: Queue, budget: int) -> None:
        ch = self.ch
        self.est *= 1.0 - self.leak
        for e in inbox:
            if e.channel < ch.sustained_start:
                i, c, _pol = ch.decode_transient(e.channel)
                self.est[i, c] += e.sign * e.value
            elif e.channel == ch.OMEGA:
                self.omega = e.value
        self._tick += 1

        lum = np.clip(self.est[:, 0], 0.0, None)
        col = np.clip(self.est[:, 1], 0.0, None)   # положительный полюс = kind A
        weight = lum * col
        if weight.sum() <= 1e-9:
            # Ничего не видно — крутиться выгодно вдвойне: и осмотреться,
            # и породить события там, где сцена статична.
            self.emit(outbox, Motor.THRUST, 0.1)
            self.emit(outbox, Motor.TURN, 0.35)
            return

        centre = (self.n - 1) / 2.0
        offset = float((weight * (np.arange(self.n) - centre)).sum() / weight.sum())
        bearing = offset / centre * math.radians(self.cfg.fov_deg) / 2.0

        self.emit(outbox, Motor.TURN, self.k_p * bearing - 0.3 * self.omega)
        self.emit(outbox, Motor.THRUST, 1.0 if abs(bearing) < 0.6 else 0.15)


BASELINES = {
    c.name: c for c in
    (RandomAgent, GreedySymbolic, GreedySustained, GreedyTransient,
     LinearPixel, TabularQ, SmallRnnBptt)
}

# Что каждый бейзлайн реально читает. Нужно для проверки покрытия каналов:
# без неё легко получить набор бейзлайнов, ни один из которых не трогает
# главный сенсорный тракт (так и было — 128 каналов из 150 не проверялись).
READS: dict[str, frozenset[str]] = {
    "random": frozenset(),
    "greedy_symbolic": frozenset(),          # привилегированный вход
    "tabular_q": frozenset(),                # привилегированный вход
    "greedy_sustained": frozenset({"SUSTAINED", "PROPRIO"}),
    "greedy_transient": frozenset({"TRANSIENT", "PROPRIO"}),
    "linear_pixel": frozenset({"SUSTAINED"}),
    "small_rnn_bptt": frozenset({"SUSTAINED"}),
}
