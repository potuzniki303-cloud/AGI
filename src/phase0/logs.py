"""Логи и meta.json. Часть 10.2.

meta.json содержит ПОЛНЫЙ снимок констант вместе с пометками обоснования.
Единственная защита от тихой подкрутки — то, что константы записаны в
артефакт вместе с результатом.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, TextIO

from .config import Config
from .events import Channels, Event, Motor
from .truth import TruthRecord


def _git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class RunLogger:
    """Пишет truth.log, sensor.log, motor.log, budget.profile и meta.json.

    Все потоки — JSONL, чтобы прогон можно было читать построчно и обрывать
    в любом месте без порчи файла.
    """

    def __init__(self, directory: str | Path, cfg: Config,
                 channels: Channels, enable: tuple[str, ...] = ("truth", "sensor", "motor", "budget")) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.channels = channels
        self.enable = set(enable)
        self._files: dict[str, TextIO] = {}
        for name in ("truth", "sensor", "motor", "budget"):
            if name in self.enable:
                suffix = "profile" if name == "budget" else "log"
                self._files[name] = open(self.dir / f"{name}.{suffix}", "w",
                                         encoding="utf-8")
        self.write_meta()

    # ----------------------------------------------------------- meta.json
    def write_meta(self, extra: dict[str, Any] | None = None) -> None:
        meta = {
            "world_version": "phase0.1",
            "git_hash": _git_hash(),
            "mode": self.cfg.mode.name,
            "difficulty": self.cfg.difficulty,
            "clock": self.cfg.clock,
            "budget_profile": self.cfg.budget_profile,
            "seeds": self.cfg.seeds,
            "channels": {k: list(v) for k, v in self.channels.describe().items()},
            "motor_channels": {str(k): v for k, v in Motor.NAMES.items()},
            "constants": self.cfg.snapshot(),
        }
        if extra:
            meta["extra"] = extra
        (self.dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --------------------------------------------------------------- строки
    def truth(self, record: TruthRecord) -> None:
        f = self._files.get("truth")
        if f is not None:
            f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    def sensor(self, events: list[Event]) -> None:
        f = self._files.get("sensor")
        if f is None:
            return
        for e in events:
            f.write(f"{e.t} {e.channel} {e.value!r} {e.sign}\n")

    def motor(self, events: list[Event]) -> None:
        f = self._files.get("motor")
        if f is None:
            return
        for e in events:
            f.write(f"{e.t} {e.channel} {e.value!r}\n")

    def budget(self, tick: int, quota: int) -> None:
        f = self._files.get("budget")
        if f is not None:
            f.write(f"{tick} {quota}\n")

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_motor_log(path: str | Path) -> list[Event]:
    """Прочитать motor.log обратно — основа побитового реплея."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t, ch, val = line.split()
        out.append(Event(int(t), int(ch), float(val)))
    return out
