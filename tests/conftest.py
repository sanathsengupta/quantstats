"""Shared fixtures for the quantstats test suite."""

import numpy as np
import pandas as pd
import pytest

from quantstats._compat import get_frequency_alias


@pytest.fixture
def daily_returns():
    """A well-behaved daily return series."""
    rng = np.random.RandomState(7)
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    return pd.Series(rng.normal(0.0005, 0.01, 400), index=dates, name="Strategy")


@pytest.fixture
def weekly_returns():
    """Five years of weekly returns."""
    rng = np.random.RandomState(11)
    dates = pd.date_range("2018-01-07", periods=260, freq="W")
    return pd.Series(rng.normal(0.002, 0.02, 260), index=dates, name="Weekly")


@pytest.fixture
def monthly_returns():
    """Ten years of month-end returns."""
    rng = np.random.RandomState(13)
    dates = pd.date_range("2014-01-31", periods=120, freq=get_frequency_alias("ME"))
    return pd.Series(rng.normal(0.008, 0.04, 120), index=dates, name="Monthly")


@pytest.fixture
def tiny_returns():
    """The five-observation example used throughout the stats docstrings."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=dates, name="Tiny")


@pytest.fixture
def paired_returns_benchmark():
    """A strategy with beta ~0.8 to its benchmark, three years of business days."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=756)
    bench = pd.Series(rng.normal(0.0003, 0.01, 756), index=idx, name="Benchmark")
    strat = (0.8 * bench + rng.normal(0, 0.006, 756)).rename("Strategy")
    return strat, bench
