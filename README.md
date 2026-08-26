# DA_FIREbox_TNG

Domain adaptation between the FIREbox and IllustrisTNG100 cosmological simulations,
predicting halo mass (`log M_halo`) across simulations with a GraphSAGE model and a
heteroscedastic (mean, variance) output head. Uses SIDDA (Sinkhorn Dynamic Domain
Adaptation), with three reweighting variants, against a No-DA source-only baseline.

## Setup

```bash
pip install -r requirements.txt
git clone --branch da-paper-shiftkit https://github.com/abdelrahman-helal/ShiftKit.git
pip install -r ShiftKit/requirements.txt
```

`ShiftKit` has no packaging metadata, so it's cloned directly into the repo root rather
than pip-installed; each notebook's setup cell auto-detects the repo root and puts
`./ShiftKit` on `sys.path`, so no separate install step is needed beyond cloning it into
place and installing its own requirements.

### Data

Not included in this repo. Each notebook expects, relative to the repo root:

```
data/firebox_data/FIREbox_z=0.txt
data/tng-data/TNG100/subhalos_99.parquet
```

## Notebooks

| Notebook | Description |
|---|---|
| `domain_shift_tng100_4methods.ipynb` | No-DA / SIDDA / SIDDA+OT-reweight / SIDDA+Loss-reweight comparison, single run per method. |
| `domain_shift_tng100_20seeds.ipynb` | Same 4 methods, 20 recorded seeds each, deterministic (`torch.use_deterministic_algorithms`). Reports mean/SD/95% CI per method and a representative-seed scatter + pull-distribution figure. |
| `domain_shift_tng100_5runs.ipynb` | Earlier, unseeded version of the 20-run sweep; kept for comparison against the seeded version. |
| `domain_shift_tng100_seed2_repro.ipynb` | Standalone rerun of seed 2 only, to check that a recorded seed reproduces the sweep's result in a fresh process. |

`scripts/process_data.py` and `scripts/process_tng_data.py` build the FIREbox and TNG100
graphs; `scripts/analysis/plot_style.py` holds the shared pred-vs-true plotting style.

## Reproducibility notes

- Seeding only fixes weight initialization; CUDA's scatter-add is otherwise
  non-deterministic, so `torch.use_deterministic_algorithms(True)` is required for a
  recorded seed to actually reproduce a run (the 20-seed and seed2-repro notebooks
  enable this; the others do not).
- `data/`, `output/`, and `ShiftKit/` are gitignored.
