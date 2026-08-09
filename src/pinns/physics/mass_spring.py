import torch


def simulate_oscillator(
    t: torch.Tensor,
    omega: torch.Tensor,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """
    Simula un oscilador armónico simple.

    Parámetros
    ----------
    t : torch.Tensor
        Vector temporal.
    omega : torch.Tensor
        Frecuencia angular del sistema.
    amplitude : float
        Amplitud inicial.

    Retorna
    -------
    torch.Tensor
        Respuesta x(t) del oscilador.
    """

    x = amplitude * torch.sin(omega * t)

    return x