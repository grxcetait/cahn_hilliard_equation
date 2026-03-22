# Cahn-Hilliard Phase-Field Simulation

A 2-D numerical simulation of the **Cahn-Hilliard equation**, modelling 
phase separation in a binary fluid (e.g. oil and water) on a
periodic square lattice.

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install numpy matplotlib
```

## cahn_hilliard.py
This script has all the functions and classes to either run an animation or measurements of the simulation. 
The user needs to put in different arguements and customise the animation or measurement conditions.

### Arguments

- 'phi', Mean initial composition. Use `0` for a symmetric 50/50 mixture; positive/negative values bias towards one phase., Default = 0
- 'l', Lattice side length. The simulation grid is `l × l`, Default = 100
- 'dx', Spatial step size, Default = 1.0
- 'dt', Time step size, Default = 0.01
- 'mode', Choose between animation 'ani' or measurements 'mea', Default = 'ani'
- 'steps', Total number of simulation steps, Default = 100000

## Command line examples

### Animation (`--mode ani`)

```
python3 cahn_hilliard.py --mode ani --steps 100000 --phi 0
```

### Measurements (`--mode mea`)

```
python3 poisson.py --type e --alg j --tol 1e-3 --l 100
```

## Output
All outputs are saved relative to the scipt's directory:

```
outputs/
├── datafiles/     # Raw measurement data (.txt)
└── plots/         # Saved figures (.png, 300 dpi)
```

