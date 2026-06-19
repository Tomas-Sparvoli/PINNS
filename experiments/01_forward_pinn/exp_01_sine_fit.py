from pathlib import Path
import sys

import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pinns.models import MLP
from pinns.utils.device import get_device
from pinns.utils.reproducibility import set_seed


def main() -> None:
    set_seed(42)
    device = get_device()

    print("Usando:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    t = torch.linspace(0, 1, 1000).view(-1, 1).to(device)
    x_real = torch.sin(15 * t)

    model = MLP(input_dim=1, output_dim=1, hidden_dim=64, num_hidden_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        optimizer.zero_grad()
        pred = model(t)
        loss = torch.mean((pred - x_real) ** 2)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss={loss.item():.6f}")

    print("\nENTRENAMIENTO FINALIZADO")


if __name__ == "__main__":
    main()
