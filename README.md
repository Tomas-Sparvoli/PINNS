# Proyecto PINNs

Proyecto de investigacion orientado al desarrollo de:

- Physics-Informed Neural Networks (PINNs)
- Physics-Informed Autoencoders (PIAE)
- Modelos hibridos con restricciones fisicas
- Parametric PINNs
- Problemas inversos en sistemas dinamicos

## Estado actual

- [x] VS Code configurado
- [x] Entorno virtual (`.venv`)
- [x] PyTorch instalado
- [x] CUDA funcionando
- [x] RTX 3080 Ti detectada
- [x] Estructura base de investigacion academica

## Estructura del proyecto

```text
PINNS/
|-- src/pinns/          # Codigo reutilizable: modelos, fisica, losses y training
|-- experiments/        # Experimentos concretos separados por linea de investigacion
|-- configs/            # Configuraciones globales y compartidas
|-- data/               # Datos raw, procesados, sinteticos y externos
|-- results/            # Corridas, logs, checkpoints, figuras y tablas
|-- notebooks/          # Exploracion, analisis y figuras para papers
|-- scripts/            # Utilidades ejecutables
|-- tests/              # Tests unitarios y de integracion
|-- docs/               # Notas teoricas, referencias y documentacion tecnica
`-- papers/             # Notas, bibliografia y material asociado a publicaciones
```

## Separacion conceptual

- `src/pinns/models/`: arquitecturas reutilizables sin hiperparametros de experimentos.
- `src/pinns/physics/`: ecuaciones, residuos fisicos y condiciones de borde.
- `src/pinns/losses/`: funciones de perdida para PINNs, PIAE y problemas inversos.
- `src/pinns/training/`: ciclos de entrenamiento genericos.
- `experiments/`: scripts reproducibles que combinan modelos, datos, fisica y configs.

## Primeros experimentos

- `experiments/00_system_checks/exp_00_gpu_test.py`: verificacion de GPU y CUDA.
- `experiments/01_forward_pinn/exp_01_sine_fit.py`: ajuste simple de una senal seno con una MLP.

## Uso recomendado

Desde la raiz del proyecto:

```powershell
pip install -e .
python scripts/check_gpu.py
python experiments/00_system_checks/exp_00_gpu_test.py
python experiments/01_forward_pinn/exp_01_sine_fit.py
```
