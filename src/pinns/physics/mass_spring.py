import torch


def simulate_oscillator(
    t: torch.Tensor,
    omega: torch.Tensor,
    x0: torch.Tensor,
    v0: torch.Tensor,
) -> torch.Tensor:
    """
    Simula un oscilador armónico simple a partir
    de sus condiciones iniciales.

    Parámetros
    ----------
    t : torch.Tensor
        Vector temporal.

    omega : torch.Tensor
        Frecuencia angular del sistema.

    x0 : torch.Tensor
        Posición inicial x(0).

    v0 : torch.Tensor
        Velocidad inicial dx/dt(0).

    Retorna
    -------
    torch.Tensor
        Respuesta x(t) del oscilador.
    """

    x = (
        x0 * torch.cos(omega * t)
        +
        (v0 / omega) * torch.sin(omega * t)
    )

    return x