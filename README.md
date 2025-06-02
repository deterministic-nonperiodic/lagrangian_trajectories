# Lagrangian Trajectory Model with Ensemble Support

This repository contains a flexible and efficient Python implementation for computing Lagrangian trajectories using 3D wind fields from reanalyses or models, such as ICON. The tool supports ensemble simulations with noise perturbations and early stopping when particles approach a specified target region.

---

## 🚀 Features

- ✅ Vectorized integration using `scipy.solve_ivp` with `dense_output=True`
- ✅ Native support for `xarray` and `dask` datasets
- ✅ Lognormal or Gaussian wind perturbations for ensemble spread
- ✅ Early termination via event detection based on spatial proximity
- ✅ Compatible with both forward and backward trajectory calculations
- ✅ CF-compliant `xarray.Dataset` output

---

## 📦 Requirements

Install dependencies with:

```bash
pip install numpy scipy pandas xarray dask joblib pyproj
```

## 🚀 Basic Usage

```python
from lagrangian import LagrangianTrajectories

# Create the model
model = LagrangianTrajectories(
    data=wind_data,  # xarray.Dataset with 'u', 'v', 'w'
    timestep=600,    # in seconds
    noise_type='lognormal',
    start_time="2025-02-20T00:00:00"
)

# Run the simulation
trajectories = model.advect_particles(
    start_positions=[(11.4, 54.1, 96000)],
    duration="3h",
    ensemble_size=20,
    target=target_ds,
    distance_tolerance=500e3
)
```
