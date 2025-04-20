import gym
import numpy as np

from LLM.PSO.PSO import PSO
from LLM.PSO.ObjectiveFunctions.Training.Rastrgin import RastriginFunction
from LLM.PSO.Cognitive.LBest import LocalBestStrategy
from LLM.SwarmMetrics import compute_swarm_metrics


class PSOEnv(gym.Env):
    def __init__(self, dim=30, num_particles=30, max_steps=100):
        super().__init__()
        self.history = []
        self.max_steps = max_steps
        self.current_step = 0
        self.last_gbest = None

        # Init PSO problem
        self.obj_fn = RastriginFunction(dim=dim, num_particles=num_particles)
        self.strategy = LocalBestStrategy(neighborhood_size=2)
        self.pso = PSO(self.obj_fn, self.strategy)
        self.pso.kb_sharing_strat.swarm = self.pso  # backref

        # === Observation Space ===
        # avg_velocity, feasible_ratio, stable_ratio, normalized gbest
        self.observation_space = gym.spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(4,),
            dtype=np.float32
        )

        # === Action Space ===
        # SAC outputs: omega ∈ [0.3, 1.0], c1, c2 ∈ [0.0, 3.0]
        self.action_space = gym.spaces.Box(
            low=np.array([0.3, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 3.0, 3.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.history = []
        self.current_step = 0
        self.obj_fn = RastriginFunction(dim=self.obj_fn.dim, num_particles=self.obj_fn.num_particles)
        self.pso = PSO(self.obj_fn, self.strategy)
        self.pso.kb_sharing_strat.swarm = self.pso

        metrics, gbest = self.pso.optimize_step(omega=0.5, c1=1.0, c2=1.0)
        self.last_gbest = gbest
        return self._get_obs(metrics), {}

    def step(self, action):
        self.current_step += 1
        omega, c1, c2 = np.clip(action, self.action_space.low, self.action_space.high)

        metrics, gbest = self.pso.optimize_step(omega, c1, c2)

        reward = self.last_gbest - gbest  # improvement reward
        self.last_gbest = gbest

        obs = self._get_obs(metrics, gbest)
        swarm_stats = compute_swarm_metrics(self.pso.particles)
        self.history.append({
            "step": self.current_step,
            "omega": float(omega),
            "c1": float(c1),
            "c2": float(c2),
            "gbest": gbest,
            "metrics": metrics,
            "positions": [p.position.copy() for p in self.pso.particles],
            "diversity": swarm_stats["diversity"],
            "velocity": swarm_stats["velocity"]
        })

        done = self.current_step >= self.max_steps

        return obs, reward, done, False, {}

    def _get_obs(self, metrics, gbest=None):
        norm_gbest = (self.obj_fn.dim * 10 - self.pso.gbest_value) / (self.obj_fn.dim * 10)
        return np.array([
            metrics['avg_velocity'],
            metrics['feasible_ratio'],
            metrics['stable_ratio'],
            np.clip(norm_gbest, 0.0, 1.0)
        ], dtype=np.float32)
