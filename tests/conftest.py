import os

import pytest

# Fast ensemble size for slow-marked tests; override with NEMTROPY_ENSEMBLE_N.
ENSEMBLE_N = int(os.environ.get("NEMTROPY_ENSEMBLE_N", "10"))


@pytest.fixture(scope="session", autouse=True)
def _mpl_cache_dir():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-nemtropy-cache")


@pytest.fixture
def ensemble_output_dir(tmp_path):
    """Writable directory for ensemble_sampler file output."""
    out = tmp_path / "ensemble"
    out.mkdir()
    return str(out) + os.sep
