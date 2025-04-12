device = "cuda" if torch.cuda.is_available() else "cpu"
buffer = ReplayBuffer(10000, state_dim=3, action_dim=1, device=device)

for _ in range(100):
    buffer.push([1.0, 2.0, 3.0], [0.5], 1.0, [1.1, 2.1, 3.1], False)

s, a, r, ns, d = buffer.sample(10)
print(f"State batch shape: {s.shape}, device: {s.device}")
