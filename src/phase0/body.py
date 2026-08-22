"""Тело и стены. Часть 2.1-2.2.

Интегратор выписан ровно в том порядке, что в спецификации. Порядок операций
здесь — часть определения физики, а не деталь: перестановка «сначала drag,
потом ускорение» даёт другую траекторию, и побитовый реплей это заметит.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Config


@dataclass(slots=True)
class Body:
    """Круг радиуса R_body с инерцией.

    Почему инерция обязательна: без неё действие мгновенно даёт результат,
    предсказывать нечего, и сенсомоторный бэббинг вырождается. Инерция создаёт
    короткую предсказуемую динамику — материал, на котором можно учиться.
    """

    x: float
    y: float
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    energy: float = 0.6
    alive: bool = True

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def v_forward(self) -> float:
        """Проекция скорости на направление взгляда."""
        return self.vx * math.cos(self.theta) + self.vy * math.sin(self.theta)

    @property
    def v_lateral(self) -> float:
        """Поперечная составляющая — снос вбок."""
        return -self.vx * math.sin(self.theta) + self.vy * math.cos(self.theta)


def integrate(body: Body, thrust: float, turn: float, cfg: Config) -> int:
    """Один шаг физики. Возвращает число столкновений со стеной (0 или 1).

    thrust/turn — удерживаемые значения моторных каналов, уже приведённые
    к [-1, 1] моторным трактом. Мир не проверяет, «хотел» ли агент их подать:
    для физики нет разницы между свежим событием и удержанным с прошлого раза.
    """
    dt = cfg.dt

    # --- линейное движение ---
    ax = thrust * cfg.f_max / cfg.mass * math.cos(body.theta)
    ay = thrust * cfg.f_max / cfg.mass * math.sin(body.theta)
    body.vx += ax * dt
    body.vy += ay * dt
    damp = 1.0 - cfg.drag * dt
    body.vx *= damp
    body.vy *= damp
    body.x += body.vx * dt
    body.y += body.vy * dt

    # --- вращение ---
    alpha = turn * cfg.t_max_over_i
    body.omega += alpha * dt
    body.omega *= 1.0 - cfg.drag_ang * dt
    body.theta += body.omega * dt
    body.theta = _wrap_angle(body.theta)

    if not cfg.walls:
        return 0
    return _collide_walls(body, cfg)


def _collide_walls(body: Body, cfg: Config) -> int:
    """Столкновение со стеной: нормальная компонента обнуляется,
    тангенциальная сохраняется (скольжение)."""
    w, h = cfg.arena
    r = cfg.r_body
    hit = 0

    if body.x < r:
        body.x = r
        if body.vx < 0.0:
            body.vx = 0.0
            hit = 1
    elif body.x > w - r:
        body.x = w - r
        if body.vx > 0.0:
            body.vx = 0.0
            hit = 1

    if body.y < r:
        body.y = r
        if body.vy < 0.0:
            body.vy = 0.0
            hit = 1
    elif body.y > h - r:
        body.y = h - r
        if body.vy > 0.0:
            body.vy = 0.0
            hit = 1

    return hit


def _wrap_angle(theta: float) -> float:
    """Держать угол в (-pi, pi]. Иначе за миллион тиков накопится величина,
    у которой float потеряет разрешение, и реплей перестанет быть побитовым."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi
