"""Placeholder until T1-T6 land; keeps the suite discoverable from an empty repo."""

import model_ci


def test_version() -> None:
    assert model_ci.__version__ == "0.1.0"
