# --- Imports remain the same ---
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from LLM.RL.ActorCritic.Actor import Actor
from LLM.RL.ActorCritic.Critic import QNetwork


class SACAgent:
    # --- ADDED adaptive_nt flag ---
    def __init__(
        self,
        state_dim,
        action_dim, # This should be 3 if fixed nt, 4 if adaptive
        hidden_dim=256,
        gamma=0.99, # Note: Paper uses gamma=1.0
        tau=0.005,
        alpha=0.2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        device="cpu",
        adaptive_nt=False # Pass the flag
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.adaptive_nt = adaptive_nt # Store flag
        self.action_dim = action_dim # Store correct action dim (3 or 4)

        # --- MODIFIED: Pass adaptive_nt and correct action_dim to Actor ---
        self.actor = Actor(state_dim, self.action_dim, hidden_dim, adaptive_nt=self.adaptive_nt).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic networks need correct action_dim
        self.q1 = QNetwork(state_dim, self.action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, self.action_dim, hidden_dim).to(device)
        self.q1_target = deepcopy(self.q1).eval().to(device)
        self.q2_target = deepcopy(self.q2).eval().to(device)

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        self.loss_fn = nn.MSELoss()

    # --- select_action method remains the same ---
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): # Make sure this is wrapped
            if deterministic:
                action = self.actor.get_deterministic_action(state)
            else:
                action, _ = self.actor(state)
        return action.detach().cpu().numpy()[0]


    # --- update method remains the same (handles variable action_dim implicitly) ---
    def update(self, replay_buffer, batch_size):
        # Sample from buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        # Ensure action tensor has correct shape [batch_size, action_dim]
        # The replay buffer should store actions with the correct dimension (3 or 4)

        with torch.no_grad():
            next_action, next_logp = self.actor(next_state)
            # Ensure next_action tensor has correct shape before passing to target critics

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
        # Freeze Q-networks when optimizing policy
        for param in self.q1.parameters(): param.requires_grad = False
        for param in self.q2.parameters(): param.requires_grad = False

        new_action, log_prob = self.actor(state)
        # Ensure new_action has correct shape
        q1_val = self.q1(state, new_action)
        q2_val = self.q2(state, new_action)
        q_val = torch.min(q1_val, q2_val)

        actor_loss = (self.alpha * log_prob - q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze Q-networks
        for param in self.q1.parameters(): param.requires_grad = True
        for param in self.q2.parameters(): param.requires_grad = True


        # Soft update target Q networks
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

    # --- _soft_update, save, load methods remain the same ---
    def _soft_update(self, source, target):
        # ... (Keep existing implementation) ...
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        # ... (Keep existing implementation) ...
        torch.save({
            'actor': self.actor.state_dict(), 'q1': self.q1.state_dict(), 'q2': self.q2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(), 'q1_optimizer': self.q1_optimizer.state_dict(), 'q2_optimizer': self.q2_optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load(self, path):
        # ... (Keep existing implementation) ...
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1.load_state_dict(checkpoint['q1']); self.q2.load_state_dict(checkpoint['q2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer']); self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        # Re-initialize target networks after loading main networks
        self.q1_target = deepcopy(self.q1).eval().to(self.device)
        self.q2_target = deepcopy(self.q2).eval().to(self.device)
        print(f"Checkpoint loaded from {path}")