import torch.nn as nn

from .genome import Genome


class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, genome: Genome=None, device: str = "cpu"):
        super().__init__(in_features, out_features, bias, device=device)
        self.genome = genome or Genome.random(device=device)

        self.device = device

    def forward(self, layer_input):
        layer_output = super().forward(layer_input)

        delta_weight = self.genome.forward(layer_input, layer_output)
        self.weight.data += delta_weight
        return layer_output

    def mutate(self):
        return Layer(self.in_features, self.out_features, bias=True, genome=self.genome.mutate(), device=self.device)