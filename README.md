# High-fidelity Rydberg controlled-Z gates with optimal pulses

Numerical optimization of laser pulses for high-fidelity Rydberg controlled-Z (CZ)
gates in neutral-atom quantum computing. The repository contains the simulation
code, optimization routines, parameter scans, and figure-generating notebooks that
accompany the manuscript of the same title.

Two ground-state atoms are coupled to a Rydberg state and the gate is realized
through the Rydberg blockade. The time-dependent master equation is integrated with
[QuTiP](https://qutip.org/), and the free parameters of the driving pulses are tuned
with a **differential-evolution (DE)** optimizer to maximize the fidelity of the
entangling gate against decoherence.

## Physical model

Each atom is described by a 5-level system (`num_levels = 5`),
labelled `|0⟩, |1⟩, |p⟩, |r⟩, |d⟩`, where `|0⟩` and `|1⟩` are the qubit states,
`|r⟩` is the Rydberg state, and the intermediate/auxiliary levels mediate the
two-photon excitation. The two-atom Hamiltonian is built as a tensor product and
includes:

- `Ω₁(t)` — the time-shaped drive on the `|1⟩ ↔ |p⟩` transition (the optimized pulse),
- `Ω₂` — a (constant) coupling on the `|p⟩ ↔ |r⟩` transition,
- `Δ₁` — the intermediate-state detuning,
- `B / Brr` — the Rydberg–Rydberg (blockade) interaction strength.

Dissipation (spontaneous decay / dephasing) is included through collapse operators
passed to `qutip.mesolve`. The gate fidelity is evaluated by preparing a Bell-type
input state and measuring overlap with the target entangled state:

```
F = ½ (b₁ + b₂) + |b₃|
```

where `b₁, b₂, b₃` are the relevant density-matrix expectation values at the end of
the gate (see `rdquantum/fidelity.py`).

## Pulse shapes

Several pulse ansätze are explored across the experiment directories:

- **Gaussian** — a smooth Gaussian-profile `Ω₁(t)` pulse (the `Gaussian*` directories).
- **Segmented / erf (Saffman)** — a piecewise pulse with error-function-smoothed
  segment boundaries, following Saffman *et al.*, *Phys. Rev. A* **101**, 062309 (2020)
  (`Saffman_shape` in `rdquantum/pulse_shape.py`, used by the `Segmented*` and
  `Saffman_*` directories).

## The `rdquantum` package

A small helper package is vendored into each experiment directory so that runs are
self-contained:

| Module | Purpose |
| --- | --- |
| `rdquantum/fidelity.py` | `fidelity` class — runs the gate via `mesolve` and returns the Bell-state fidelity for a given set of pulse parameters. |
| `rdquantum/pulse_shape.py` | Pulse-shape functions (e.g. `Saffman_shape`, an erf-smoothed segmented pulse). |
| `rdquantum/optimizer/de.py` | `de` / `DE` — differential-evolution optimizer over the pulse parameters (mutation, crossover, selection). |

### Differential evolution

The optimizer treats the pulse parameters (`Omega1`, `Omega2`, `Delta1`, `T_gate`, …)
as a real-valued vector and evolves a population of candidate pulses. Population size
is `c × K`, where `K` is the total number of control parameters and `c` is a
multiplier (default `15`). Each iteration performs mutation (factor `mu`), crossover
(rate `xi`), and greedy selection by fidelity, evaluated in parallel with QuTiP's
`parfor`. Progress is checkpointed to:

- `out.npz` — populations, fidelity history, pulse history, best fidelity,
- `out-op_pulse.npz` — the best (optimal) pulse parameters found.

A standalone copy of the optimizer also lives at the repository root in
[`de.py`](de.py).

## Repository layout

```
de.py                          Standalone differential-evolution optimizer
Clean/                         Cleaned, canonical reference run (Gaussian, Brr=500)
Gaussian500_53S_5L_BrrX/       Gaussian-pulse scan over Rydberg interaction Brr
Gaussian500_DeltaX/            Gaussian-pulse scan over detuning Delta
Segmented500_53S_5L_BrrX/      Segmented/erf-pulse scan over Brr
Segmented500_DeltaX/           Segmented/erf-pulse scan over detuning Delta
Saffman_r2_deltaR2X/           Saffman-style adiabatic-pulse comparison runs
Manuscript-figures/            Notebooks that generate the manuscript figures (fig1–6)
archived/                      Earlier exploratory runs kept for reference
```

Each scan directory (e.g. `Gaussian500_53S_5L_BrrX/Gaussian500_53S_5L_Brr500/`)
typically contains:

- a main `*.ipynb` notebook (model, gate operation, optimization, analysis),
- `*_robustness.ipynb` / `*_Brr_robustness.ipynb` — robustness sweeps,
- `test.py` — the script form of the run, launched on a cluster,
- `*.qsub` — a PBS batch script for the run,
- `out.npz`, `out-op_pulse.npz`, and `*.npy` result files,
- a vendored copy of the `rdquantum` package.

Directory naming convention: `Gaussian500` denotes a Gaussian pulse with a fixed
parameter (≈500), `53S`/`5L` refer to the Rydberg level and the 5-level model, and
`BrrX` / `DeltaX` indicate which physical quantity is being scanned.

## Requirements

- Python 3.8+
- [QuTiP](https://qutip.org/) (the code uses the QuTiP 4.x `mesolve` / `Options` /
  `parfor` API, including `rhs_reuse`)
- NumPy
- Matplotlib
- Jupyter (to run the `.ipynb` notebooks)

A conda environment named `rdqc` is referenced by the cluster scripts. To reproduce
locally:

```bash
conda create -n rdqc python=3.8 qutip numpy matplotlib jupyter
conda activate rdqc
```

## Usage

### Interactive (notebook)

Open a run directory and launch the notebook, e.g.:

```bash
cd Clean/Gaussian500_53S_5L_Brr500
jupyter notebook Gaussian500_53S_5L_Brr500.ipynb
```

The notebook defines the Hamiltonian and pulse shapes, evaluates the gate fidelity
for a given set of `Pulses`, runs the DE optimizer, and plots the optimization history
and the optimal pulse.

### Batch (script)

Each run also ships a `test.py` that performs the same optimization headlessly:

```bash
cd Gaussian500_53S_5L_BrrX/Gaussian500_53S_5L_Brr500
python3 test.py
```

On an HPC cluster with a PBS scheduler, submit the accompanying batch script:

```bash
qsub Gaussian500_53S_5L_Brr500.qsub
```

Results are written to `out.npz` and `out-op_pulse.npz` in the working directory.

## Reproducing the manuscript figures

The notebooks in `Manuscript-figures/` read the `out*.npz` result files from the scan
directories and produce the published figures (`fig1`–`fig6`, plus appendix figures on
Doppler/dephasing and `Brr` fluctuations) as `.eps`/`.png` files.

## References

1. M. Saffman, I. I. Beterov, A. Dalal, E. J. Páez, and B. C. Sanders,
   "Symmetric Rydberg controlled-Z gates with adiabatic pulses,"
   *Phys. Rev. A* **101**, 062309 (2020).

## License

Released under the [MIT License](LICENSE). Copyright (c) 2023 QFoam.
