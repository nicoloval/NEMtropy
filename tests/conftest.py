import inspect
import os
import warnings
from pathlib import Path

import pytest

import NEMtropy

# Fast ensemble size for slow-marked tests; override with NEMTROPY_ENSEMBLE_N.
ENSEMBLE_N = int(os.environ.get("NEMTROPY_ENSEMBLE_N", "10"))
_REPO_NEMTROPY = (Path(__file__).resolve().parents[1] / "src" / "NEMtropy").resolve()


def pytest_configure(config):
    pkg = Path(inspect.getfile(NEMtropy)).resolve().parent
    if pkg != _REPO_NEMTROPY:
        warnings.warn(
            f"NEMtropy imported from {pkg}, not {_REPO_NEMTROPY}; "
            "run `pip install -e \".[dev]\"` from the repo root.",
            UserWarning,
            stacklevel=1,
        )


@pytest.fixture(scope="session", autouse=True)
def _mpl_cache_dir():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-nemtropy-cache")


@pytest.fixture
def ensemble_output_dir(tmp_path):
    """Writable directory for ensemble_sampler file output."""
    out = tmp_path / "ensemble"
    out.mkdir()
    return str(out) + os.sep
