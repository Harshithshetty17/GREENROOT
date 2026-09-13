"""Shared pytest fixtures for the GREENROOT verification suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, List

import pytest

# Allow `import src...` when pytest is invoked from anywhere in the tree.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import FEATURE_NAMES, MODEL_PATH  # noqa: E402
from src.models.inference import CropRecommender  # noqa: E402

#: A canonical in-distribution reading: the first row of the benchmark corpus,
#: which the ensemble classifies as rice with high confidence.
VALID_SAMPLE: List[float] = [90.0, 42.0, 43.0, 20.88, 82.0, 6.5, 202.94]


artefacts_required = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Pre-trained artefacts absent from models/; run train.py first.",
)


@pytest.fixture(scope="session")
def recommender() -> CropRecommender:
    """A loaded :class:`CropRecommender`, shared across the session."""
    if not MODEL_PATH.exists():
        pytest.skip("Pre-trained artefacts absent from models/.")
    return CropRecommender().load()


@pytest.fixture
def valid_sample() -> List[float]:
    """A mutable copy of the canonical in-distribution reading."""
    return list(VALID_SAMPLE)


@pytest.fixture
def feature_names() -> List[str]:
    """The canonical ordered feature contract."""
    return list(FEATURE_NAMES)


@pytest.fixture
def temp_db(tmp_path: Path) -> Iterator[Path]:
    """An isolated SQLite database, initialised and torn down per test."""
    from src.database import db_manager

    path = tmp_path / "test_audit.db"
    db_manager.init_db(path)
    yield path
    db_manager.close_connections()
