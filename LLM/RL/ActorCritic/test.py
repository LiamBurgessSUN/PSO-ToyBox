import torch

from LLM.RL.ActorCritic.Actor import Actor
from LLM.RL.ActorCritic.Critic import QNetwork

state_dim = 3
action_dim = 1
actor = Actor(state_dim, action_dim)
q1 = QNetwork(state_dim, action_dim)
q2 = QNetwork(state_dim, action_dim)

state = torch.randn(5, state_dim)
action, logp = actor(state)
q_value1 = q1(state, action)
q_value2 = q2(state, action)

print("Sample action:", action)
print("Q1 value:", q_value1)
