import math
import random
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field


class Genome(BaseModel):
    """Геном для одного слоя модели.
    Один набор (A, B, C, D, η)"""
    a: float = Field(0.0)
    b: float = Field(0.0)
    c: float = Field(0.0)
    d: float = Field(0.0)
    lr: float = Field(1e-3)

    device: Any = Field(None)

    @staticmethod
    def random(device: str = 'cpu') -> 'Genome':
        def r(mu=0.0, sigma=0.5): return random.gauss(mu, sigma)

        return Genome(
            a=r(),
            b=r(),
            c=r(),
            d=r(),
            lr=abs(r(1e-3, 5e-4)),
            device=device,
        )

    def mutate(self) -> 'Genome':
        def r(mu=0.0, sigma=0.1): return random.gauss(mu, sigma)
        return Genome(
            a=self.a + r(),
            b=self.b + r(),
            c=self.c + r(),
            d=self.d + r(),
            lr=self.lr * math.exp(random.gauss(0, 5e-4)),
            device=self.device,
        )

    def forward(self, layer_input: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        dw = self.lr * (
                self.a * torch.outer(layer_output, layer_input) +
                self.b * layer_input[None, :] +
                self.c * layer_output[:, None] +
                self.d
        )
        return dw