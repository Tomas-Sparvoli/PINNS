import torch

from pinns.physics.mass_spring import simulate_oscillator


t = torch.linspace(0, 1, 200)

omega = torch.tensor(
    15.0,
    requires_grad=True,
)

x = simulate_oscillator(t, omega)

print("omega:", omega)
print("x shape:", x.shape)
print("Primeros valores:", x[:5])

loss = (x**2).mean()

loss.backward()

print("Gradiente respecto de omega:", omega.grad)