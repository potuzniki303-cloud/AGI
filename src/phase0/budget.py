"""Вычислительный бюджет. Часть 1.4.

Противоречие, которое здесь разрешается: ограничение должно навязываться
железом, а не назначаться константой, но реальное время делает эксперименты
невоспроизводимыми.

Разрешение: бюджет измеряется в КВОТЕ ОБНОВЛЕНИЙ УЗЛОВ на тик. В realtime она
измеряется на живом железе, в fast — воспроизводится из записанного профиля.
Код агента в обоих режимах одинаков и о режиме не знает; физическое
происхождение ограничения сохраняется, но однажды измеренный профиль
становится воспроизводимым артефактом.

Профиль обязан быть указан в метаданных каждого эксперимента: результаты,
полученные на разных профилях, НЕСРАВНИМЫ.
"""

from __future__ import annotations

import json
from pathlib import Path


class BudgetProfile:
    """Последовательность (tick, node_updates_allowed)."""

    def __init__(self, quotas: list[int], name: str = "unnamed") -> None:
        if not quotas:
            raise ValueError("профиль бюджета не может быть пустым")
        self.quotas = list(quotas)
        self.name = name

    def at(self, tick: int) -> int:
        """Квота на тик. За концом профиля — циклическое повторение: профиль
        описывает установившуюся производительность машины, а не расписание."""
        return self.quotas[tick % len(self.quotas)]

    # ------------------------------------------------------------- запись
    @classmethod
    def constant(cls, quota: int, length: int = 1, name: str = "constant") -> "BudgetProfile":
        """Синтетический профиль. Годится для отладки, но не для сравнения
        результатов: он не измерен, а назначен."""
        return cls([int(quota)] * length, name=name)

    @classmethod
    def measure(cls, updates_per_tick_probe, ticks: int = 600,
                name: str = "measured") -> "BudgetProfile":
        """Измерить на живом железе: сколько обновлений успевает машина за
        один тик реального времени (16.67 мс при 60 Гц).

        `updates_per_tick_probe` — вызываемое, делающее одно обновление узла.
        """
        import time

        budget_window = 1.0 / 60.0
        quotas = []
        for _ in range(ticks):
            n = 0
            deadline = time.perf_counter() + budget_window
            while time.perf_counter() < deadline:
                updates_per_tick_probe()
                n += 1
            quotas.append(n)
        return cls(quotas, name=name)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps({"name": self.name, "quotas": self.quotas}))

    @classmethod
    def load(cls, path: str | Path) -> "BudgetProfile":
        data = json.loads(Path(path).read_text())
        return cls(data["quotas"], name=data.get("name", str(path)))

    def summary(self) -> dict:
        q = self.quotas
        return {
            "name": self.name,
            "ticks": len(q),
            "min": min(q),
            "max": max(q),
            "mean": sum(q) / len(q),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return f"BudgetProfile({s['name']}, {s['ticks']} тиков, среднее {s['mean']:.0f})"
