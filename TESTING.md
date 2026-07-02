# NEMtropy — testing guide

## Layout

```
tests/
  conftest.py           # shared fixtures (ensemble output dir, MPL cache)
  unit/                 # fast: graph init, math, initial guesses, motifs
  integration/          # solver smoke tests (parametrized)
  slow/                 # ensemble sampling, motif z-scores
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

### Default CI / local quick run (~30s)

Excludes `@pytest.mark.slow` tests (see `pytest.ini`):

```bash
pytest
```

Equivalent:

```bash
pytest tests -m "not slow"
```

### Run everything including slow tests

```bash
pytest --override-ini 'addopts=' 
```

### Slow tests only

```bash
pytest --override-ini 'addopts=' -m slow
```

Override ensemble sample count (default `10`; legacy tests used `100`–`1000`):

```bash
NEMTROPY_ENSEMBLE_N=100 pytest --override-ini 'addopts=' -m slow
```

Directed/weighted ensemble statistics are skipped in CI; use a large `NEMTROPY_ENSEMBLE_N` for manual runs.

### By category

```bash
pytest tests/unit              # graph init, log-likelihood, etc.
pytest tests/integration       # parametrized _solve_problem / solve_tool
pytest tests/unit -m math      # log-likelihood derivatives
```

### Single file or test

```bash
pytest tests/unit/test_graph_init.py
pytest tests/integration/test_solve_models.py -k "cm and fixed-point"
```

### Verbose with timings

```bash
pytest --durations=15 -v
```

## Markers

| Marker | Meaning |
|--------|---------|
| `slow` | Ensemble I/O, large graphs — skipped by default |
| `integration` | Full solver path on graph objects |
| `math` | Log-likelihood / Hessian unit tests |

## GitHub Actions (suggested)

```yaml
- run: pip install -e ".[dev]"
- run: pytest
```

For a nightly job:

```yaml
- run: pytest -m slow
  env:
    NEMTROPY_ENSEMBLE_N: 20
```

## Notes

- Install the package editable (`pip install -e .`) so imports resolve without `sys.path` hacks.
- Ensemble tests write to temporary directories via `ensemble_output_dir` — no leftover `sample/` files.
- BICM is tested in the [bicm](https://github.com/mat701/BiCM) package, not here.
