import torch

from pinns.physics.mass_spring import simulate_oscillator


# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------

N_POINTS = 200

OMEGAS = [
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
]


# ----------------------------------
# TIEMPO
# ----------------------------------

t = torch.linspace(0, 1, N_POINTS)


# ----------------------------------
# GENERACIÓN DEL DATASET
# ----------------------------------

signals_time = []
signals_frequency = []
parameters = []


for omega_value in OMEGAS:

    omega = torch.tensor(omega_value)

    # Simulación física
    x = simulate_oscillator(t, omega)

    # Transformada de Fourier
    X = torch.fft.fft(x)

    # Guardamos los datos
    signals_time.append(x)
    signals_frequency.append(X)
    parameters.append(omega)


# ----------------------------------
# CONVERTIR LISTAS EN TENSORES
# ----------------------------------

signals_time = torch.stack(signals_time)
signals_frequency = torch.stack(signals_frequency)
parameters = torch.stack(parameters)


# ----------------------------------
# MOSTRAR INFORMACIÓN
# ----------------------------------

print("Cantidad de señales:", len(signals_time))
print("Forma de señales temporales:", signals_time.shape)
print("Forma de señales frecuenciales:", signals_frequency.shape)
print("Parámetros:", parameters)