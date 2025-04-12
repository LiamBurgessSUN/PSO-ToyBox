import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from LLM.RL.ActorCritic.Actor import Actor
from LLM.RL.ActorCritic.Critic import QNetwork


class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        device="cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic networks (Q1 & Q2)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target = deepcopy(self.q1).eval().to(device)
        self.q2_target = deepcopy(self.q2).eval().to(device)

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        self.loss_fn = nn.MSELoss()

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        print(f"Checkpoint loaded from {path}")

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            action = self.actor.get_deterministic_action(state)
        else:
            action, _ = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size):
        # Sample from buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_logp = self.actor(next_state)
            target_q1 = self.q1_target(next_state, next_action)
            target_q2 = self.q2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_logp
            target = reward + (1 - done) * self.gamma * target_q

        # Update Q networks
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)

        q1_loss = self.loss_fn(current_q1, target)
        q2_loss = self.loss_fn(current_q2, target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy network (actor)
        new_action, log_prob = self.actor(state)
        q1_val = self.q1(state, new_action)
        q2_val = self.q2(state, new_action)
        q_val = torch.min(q1_val, q2_val)

        actor_loss = (self.alpha * log_prob - q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target Q networks
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
