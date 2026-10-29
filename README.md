# Index Rebalance: Reproduction Code

This package contains all numerical code for the current index rebalance paper,
including its finite-player and mean-field illustrations and all nine manuscript
figures. No external data files are required.

## Setup and Run

Use Python 3.10 or later. From this folder, create a virtual environment and
install the dependencies:

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the model checks and reproduce all nine figures:

```sh
python -m unittest -v test_models
python figures.py
```

The figure generator creates `paper/figures/` beneath this folder, with
`finite_player_game/` and `mean_field_game/` subfolders. It uses a noninteractive
Matplotlib backend, so no display server or LaTeX installation is required.
The first run computes the solutions and can take several minutes. Numerical
caches are created automatically in the system temporary directory beneath
`index-rebalance/cache/`.

To select output and cache locations and recompute all solutions:

```sh
python figures.py --output-root results/figures --cache-root results/cache --force
```

Relative paths are resolved from this package folder. Repeated runs reuse
matching cached solutions unless `--force` is supplied.

## Files

- `_index_rebalance_common.py`: parameter validation and shared numerical tools.
- `index_rebalance_finite_closed_loop.py`: finite-player equilibrium solver.
- `index_rebalance_mean_field.py`: diagonal mean-field paths used in the paper.
- `experiments.py`: calibration, scenarios, priors, and numerical metrics.
- `figures.py`: generator for the nine manuscript figures.
- `test_models.py`: model checks against coefficients, boundary conditions, and
  independent dynamics.
- `requirements.txt`: NumPy, SciPy, and Matplotlib dependencies.

The numerical illustration solvers use zero initial inventories and impact.
The mean-field module computes the diagonal common paths used in the figures.
The manuscript, separate research extensions, empirical research, and formal
proof tooling are outside this numerical reproduction package.

Validated environment: Python 3.12.8, NumPy 1.26.4, SciPy 1.13.1, and
Matplotlib 3.9.2.
