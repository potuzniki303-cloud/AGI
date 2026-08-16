import gymnasium as gym
import torch

from agent import Agent
from world import register_all

register_all()

env = gym.make("Life/Bounty-v0",
               size=40,
               seed=42,
               initial_population=30,
               max_food=10,
               energy_at_birth=80,
               render_mode="fullscreen",)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

n_in  = env.unwrapped.single_observation_space.shape[0]
n_act = env.unwrapped.single_action_space.n

obs, info = env.reset()


agents = {}
for agent_index in info['ids']:
    agents[agent_index] = Agent(n_in, n_act, device)

truncated = terminated = False

while not truncated and not terminated:
    for parent, child in info['born']:
        agents[child] = agents[parent].mutate()
        print('я родился')
    for died in info['died']:
        del agents[died]
    actions = []
    for observation_index, agent_index in enumerate(info['ids']):
        actions.append(agents[agent_index](torch.from_numpy(obs[observation_index]).to(device)))
    obs, _reward_unusable, truncated, terminated, info = env.step(actions)

env.close()