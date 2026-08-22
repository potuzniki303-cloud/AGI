"""Мир Фазы 0. Тик, гомеостат, сборка сенсорных событий.

ГЛАВНЫЙ ИНВАРИАНТ. В этом классе нет и не должно появиться метода, который
спрашивает у агента действие. Мир не хранит ссылку на агента, не знает его
типа и не вызывает его код. Он осушает `outbox` и наполняет `inbox`. Всё.

Дисциплина тика (1.2), порядок фиксирован:
  1. Осушить outbox. События с t <= tick применить, с t > tick — отложить.
  2. Проинтегрировать физику на dt.
  3. Сгенерировать сенсорные события, записать в inbox.
  4. Записать строку канала истины.
  5. Тик завершён. Мир не ждёт.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .body import Body, integrate
from .config import Config, Kind, Mode
from .events import Channels, Event, Motor, Queue
from .items import ItemField
from .motor import MotorTract
from .regimes import Regime
from .retina import Retina
from .rng import RngBundle
from .truth import ItemTruth, TruthRecord


class World:
    """Арена, тело, предметы, гомеостат и сенсорный тракт."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.channels = Channels(self.cfg)
        self.rng = RngBundle(self.cfg.seeds)

        self.inbox = Queue()   # мир пишет, агент читает
        self.outbox = Queue()  # агент пишет, мир читает

        self.tick = 0
        self.motor = MotorTract(self.cfg)
        self.regime = Regime(self.cfg, self.rng.regime_flips)
        self.items = ItemField(self.cfg, self.rng.world_layout)
        self.retina = Retina(self.cfg, self.channels)

        self.body = Body(
            x=float(self.rng.world_layout.uniform(self.cfg.r_body,
                                                  self.cfg.arena[0] - self.cfg.r_body)),
            y=float(self.rng.world_layout.uniform(self.cfg.r_body,
                                                  self.cfg.arena[1] - self.cfg.r_body)),
            theta=float(self.rng.world_layout.uniform(-math.pi, math.pi)),
            energy=self.cfg.e_init,
        )

        # Фильтры плотных каналов.
        self._proprio = np.zeros(4, dtype=np.float64)
        self._alpha_proprio = self.cfg.dt / (self.cfg.tau_proprio + self.cfg.dt)
        self._energy_filt = self.cfg.e_init
        self._alpha_energy = self.cfg.dt / (self.cfg.tau_energy + self.cfg.dt)

        self._noci_until = -1      # тик, до которого держится боль смерти
        self._noci_impulse = 0.0   # мгновенная боль этого тика

        self._deferred: list[Event] = []
        self._node_count = 0
        self._budget = 0

        # Счётчики для метрик и канала истины.
        self.deaths = 0
        self.eaten_counts = {Kind.A: 0, Kind.B: 0}
        self.poison_events = 0
        self.wall_hits = 0
        self.dropped_events = 0
        self.last_truth: TruthRecord | None = None

    # ------------------------------------------------------------- бюджет
    def set_node_count(self, n: int) -> None:
        """Агент СООБЩАЕТ размер своего графа (push, не pull).

        Это не нарушение контракта: мир не идёт к агенту за данными, агент сам
        кладёт число, когда оно меняется. Нужно только при METABOLIC_COMPUTE,
        где растущий граф буквально голодает.
        """
        self._node_count = max(0, int(n))

    def set_budget(self, node_updates_allowed: int) -> None:
        self._budget = int(node_updates_allowed)

    # --------------------------------------------------------------- такт
    def step(self) -> TruthRecord:
        """Один тик. Возвращает строку канала истины (для лога, НЕ для агента)."""
        cfg = self.cfg
        tick = self.tick
        events_log: list[dict[str, Any]] = []
        self._noci_impulse = 0.0

        # --- 1. осушить outbox ---
        self._drain_outbox(tick)

        # --- 2. физика ---
        if self.regime.step(tick):
            events_log.append({"type": "regime_flip",
                               "flag_state": self.regime.flag_state})

        self.motor.decay()
        wall_hit = 0
        if self.body.alive:
            wall_hit = integrate(self.body, self.motor.thrust, self.motor.turn, cfg)
            if wall_hit:
                self.wall_hits += 1
                self.body.energy -= cfg.e_wall_hit
                self._noci_impulse = max(self._noci_impulse, 0.5)
                events_log.append({"type": "wall_hit"})

        self.items.step(tick, (self.body.x, self.body.y), self.rng.item_respawn)

        # --- 2b. поедание ---
        if self.body.alive:
            for item in self.items.touching(self.body.x, self.body.y):
                delta = self.regime.nutritive(item.kind, tick)
                self.body.energy += delta
                self.eaten_counts[item.kind] += 1
                if delta < 0.0:
                    self.poison_events += 1
                    self._noci_impulse = max(self._noci_impulse, 1.0)
                self.items.consume(item, tick)
                events_log.append({"type": "eat", "item_id": item.item_id,
                                   "kind": item.kind.name, "dE": delta})

        # --- 2c. гомеостат ---
        if self.body.alive:
            drain = cfg.basal + cfg.move_cost * (
                abs(self.motor.thrust) + cfg.turn_cost_ratio * abs(self.motor.turn)
            )
            if cfg.metabolic_compute:
                drain += cfg.k_compute * (self._node_count / cfg.node_count_ref)
            self.body.energy -= drain * cfg.dt
            self.body.energy = min(self.body.energy, cfg.e_max)

            if self.body.energy <= 0.0:
                self._die(tick, events_log)

        # --- 3. сенсорные события ---
        losses = self._emit_sensory(tick)

        # --- 4. канал истины ---
        record = self._truth(tick, events_log, losses)
        self.last_truth = record

        # --- 5. тик завершён, мир не ждёт ---
        self.tick += 1
        return record

    # ------------------------------------------------------------ outbox
    def _drain_outbox(self, tick: int) -> None:
        """События с t <= tick применить, с t > tick отложить.

        Опережение возможно в fast-режиме, где агент может уехать вперёд по
        логическому времени. Отложенные события не теряются и не применяются
        раньше срока — иначе fast и realtime расходились бы.
        """
        pending = self._deferred + self.outbox.drain()
        self._deferred = []
        for ev in pending:
            if ev.t > tick:
                self._deferred.append(ev)
            else:
                self.motor.apply(ev)

    # ------------------------------------------------------------ сенсоры
    def _emit_sensory(self, tick: int) -> int:
        cfg, ch = self.cfg, self.channels
        proj = self.retina.project(self.body.x, self.body.y, self.body.theta,
                                   self.items.visible_items)
        self._last_projection = proj

        transient, dropped = self.retina.transient_events(tick, proj)
        self.dropped_events += dropped
        self.inbox.extend(transient)
        self.inbox.extend(self.retina.sustained_events(tick, proj))

        # Проприоцепция: без неё сенсомоторные контингенции неполны —
        # агент не может отличить «мир движется» от «я движусь».
        target = np.array([self.body.v_forward, self.body.v_lateral,
                           self.body.omega, self.body.speed], dtype=np.float64)
        self._proprio += self._alpha_proprio * (target - self._proprio)
        if cfg.sensor_noise_sigma > 0.0:
            self._proprio = self._proprio + self.rng.sensor_noise.normal(
                0.0, cfg.sensor_noise_sigma, 4)
        for i, value in enumerate(self._proprio):
            self.inbox.put(Event(tick, ch.proprio_start + i, float(value)))

        # Интероцепция. ENERGY даётся напрямую и это законно: это ощущение
        # себя, а не подсказка про среду.
        self._energy_filt += self._alpha_energy * (self.body.energy - self._energy_filt)
        self.inbox.put(Event(tick, ch.ENERGY, float(self._energy_filt)))

        noci = self._noci_impulse
        if tick < self._noci_until:
            noci = max(noci, 1.0)
        if noci > 0.0:
            self.inbox.put(Event(tick, ch.NOCI, float(noci)))

        return dropped

    # -------------------------------------------------------------- смерть
    def _die(self, tick: int, events_log: list[dict[str, Any]]) -> None:
        """Смерть тела, но НЕ процесса.

        Граф агента не трогается: ни сброса, ни обнуления, ни перезапуска.
        Это сознательное ослабление Принципа 1 ради достижимой статистики,
        и оно должно оставаться в списке допущений, а не растворяться в коде.
        """
        cfg = self.cfg
        self.deaths += 1
        events_log.append({"type": "death", "cause": "starvation"})

        self.body.x = float(self.rng.world_layout.uniform(
            cfg.r_body, cfg.arena[0] - cfg.r_body))
        self.body.y = float(self.rng.world_layout.uniform(
            cfg.r_body, cfg.arena[1] - cfg.r_body))
        self.body.theta = float(self.rng.world_layout.uniform(-math.pi, math.pi))
        self.body.vx = self.body.vy = self.body.omega = 0.0
        self.body.energy = cfg.e_init
        self.body.alive = True

        self._noci_until = tick + cfg.noci_duration
        events_log.append({"type": "respawn"})

    # -------------------------------------------------------- канал истины
    def _truth(self, tick: int, events_log: list[dict[str, Any]],
               losses: int) -> TruthRecord:
        proj = self._last_projection
        items: list[ItemTruth] = []
        for item in self.items.items:
            span = proj.spans.get(item.item_id)
            seen = proj.visible_receptors(item.item_id)
            in_fov = bool(item.alive and span is not None)
            items.append(
                ItemTruth(
                    item_id=item.item_id,
                    x=item.x,
                    y=item.y,
                    kind=item.kind.name,
                    nutritive=self.regime.nutritive(item.kind, tick),
                    # ВИДНО — значит закрашен хотя бы один рецептор.
                    # Попадание в поле зрения это отдельный факт (in_fov).
                    visible=bool(in_fov and seen > 0),
                    in_fov=in_fov,
                    retinal_span=span,
                    visible_receptors=seen,
                    occlusion=proj.occlusion.get(item.item_id, "none"),
                    occluded_by=proj.occluded_by.get(item.item_id, -1),
                )
            )
        rs = self.regime.state(tick)
        return TruthRecord(
            tick=tick,
            body={
                "x": self.body.x, "y": self.body.y, "theta": self.body.theta,
                "vx": self.body.vx, "vy": self.body.vy, "omega": self.body.omega,
                "E": self.body.energy, "alive": self.body.alive,
            },
            items=items,
            regime={"mode": rs.mode, "flag_state": rs.flag_state,
                    "ticks_since_flip": rs.ticks_since_flip,
                    "phase": rs.phase, "window_open": rs.window_open},
            retina_owner=proj.owner.tolist(),
            events=events_log,
            budget=self._budget,
            losses=losses,
        )

    # ------------------------------------------------------------- снапшот
    def state(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "rng": self.rng.state(),
            "body": (self.body.x, self.body.y, self.body.theta, self.body.vx,
                     self.body.vy, self.body.omega, self.body.energy, self.body.alive),
            "motor": self.motor.state(),
            "items": self.items.state(),
            "regime": self.regime.snapshot(),
            "retina": self.retina.state(),
            "proprio": self._proprio.copy(),
            "energy_filt": self._energy_filt,
            "noci_until": self._noci_until,
            "deferred": [e.as_row() for e in self._deferred],
            # События в полёте — часть состояния мира. Без них снапшот теряет
            # то, что агент уже эмитил, но мир ещё не применил, и продолжение
            # расходится с непрерывным прогоном (проверено: расходилось).
            "inbox": [e.as_row() for e in self.inbox.peek()],
            "outbox": [e.as_row() for e in self.outbox.peek()],
            "counters": {
                "deaths": self.deaths,
                "eaten_A": self.eaten_counts[Kind.A],
                "eaten_B": self.eaten_counts[Kind.B],
                "poison": self.poison_events,
                "wall_hits": self.wall_hits,
                "dropped": self.dropped_events,
            },
            "node_count": self._node_count,
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.tick = state["tick"]
        self.rng.restore(state["rng"])
        (self.body.x, self.body.y, self.body.theta, self.body.vx, self.body.vy,
         self.body.omega, self.body.energy, self.body.alive) = state["body"]
        self.motor.restore(state["motor"])
        self.items.restore(state["items"])
        self.regime.restore(state["regime"])
        self.retina.restore(state["retina"])
        self._proprio = state["proprio"].copy()
        self._energy_filt = state["energy_filt"]
        self._noci_until = state["noci_until"]
        self._deferred = [Event(*row) for row in state["deferred"]]
        c = state["counters"]
        self.deaths = c["deaths"]
        self.eaten_counts = {Kind.A: c["eaten_A"], Kind.B: c["eaten_B"]}
        self.poison_events = c["poison"]
        self.wall_hits = c["wall_hits"]
        self.dropped_events = c["dropped"]
        self._node_count = state["node_count"]
        self.inbox.drain()
        self.outbox.drain()
        self.inbox.extend(Event(*row) for row in state.get("inbox", ()))
        self.outbox.extend(Event(*row) for row in state.get("outbox", ()))

    def __repr__(self) -> str:
        return (f"World(tick={self.tick}, mode={self.cfg.mode.name}"
                f"/{self.cfg.difficulty}, E={self.body.energy:.3f}, "
                f"deaths={self.deaths})")
