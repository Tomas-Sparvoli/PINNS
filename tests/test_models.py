import torch

from pinns.models import MLP


def test_mlp_output_shape() -> None:
    model = MLP(input_dim=1, output_dim=2, hidden_dim=8, num_hidden_layers=2)
    x = torch.zeros(4, 1)

    y = model(x)

    assert y.shape == (4, 2)
