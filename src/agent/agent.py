import random

import torch
from torch import nn, Tensor

from .layer import Layer

class Agent(nn.Module):
    def __init__(self, input_size: int, output_size: int, device) -> None:
        super().__init__()
        self.fc1 = Layer(input_size, 150,device=device)
        self.fc2 = Layer(150, output_size, device=device)
        self.hidden = torch.zeros(15, device=device)

        self.input_size = input_size
        self.output_size = output_size

        self.device = device

    @torch.no_grad()
    def forward(self, observation: Tensor) -> int:
        model_input = observation

        inb_hidden_state = self.fc1(model_input)

        out = self.fc2(inb_hidden_state)

        logits = out[:4]
        self.hidden = out[4:]
        try:
            return int(torch.multinomial(torch.softmax(torch.tanh(logits), dim=0).to(self.device), 1).to(self.device))
        except RuntimeError:
            return random.randint(0, 3)

    def mutate(self) -> 'Agent':
        new_agent = Agent(self.input_size, self.output_size, self.device)
        new_agent.fc1 = self.fc1.mutate()
        new_agent.fc2 = self.fc2.mutate()
        return new_agent
