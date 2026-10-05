"""
Tests for quantstats.stats module
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from quantstats import stats


@pytest.fixture
def sample_returns():
    """Generate sample daily returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.randn(500) * 0.02, index=dates, name="Strategy")
    return returns


@pytest.fixture
def sample_benchmark():
    """Generate sample benchmark returns for testing."""
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.randn(500) * 0.015, index=dates, name="Benchmark")
    return returns


@pytest.fixture
def positive_returns():
    """Generate strictly positive returns for testing edge cases."""
    rng = np.random.RandomState(1)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.Series(np.abs(rng.randn(100) * 0.01) + 0.001, index=dates)
    return returns


@pytest.fixture
def negative_returns():
    """Generate strictly negative returns for testing edge cases."""
    rng = np.random.RandomState(2)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.Series(-np.abs(rng.randn(100) * 0.01) - 0.001, index=dates)
    return returns


class TestBasicStats:
    """Test basic statistical functions."""

    def test_comp(self, sample_returns):
        """Test compound returns calculation."""
        result = stats.comp(sample_returns)
        assert isinstance(result, float)
        # Compound return should be finite
        assert np.isfinite(result)

    def test_compsum(self, sample_returns):
        """Test cumulative compound returns."""
        result = stats.compsum(sample_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_returns)
        # Final value should match comp()
        np.testing.assert_almost_equal(
            result.iloc[-1], stats.comp(sample_returns), decimal=10
        )

    def test_exposure(self, sample_returns):
        """Test exposure calculation."""
        result = stats.exposure(sample_returns)
        assert 0 <= result <= 1

    def test_win_rate(self, sample_returns):
        """Test win rate calculation."""
        result = stats.win_rate(sample_returns)
        assert 0 <= result <= 1

    def test_win_rate_all_positive(self, positive_returns):
        """Test win rate with all positive returns."""
        result = stats.win_rate(positive_returns)
        assert result == 1.0

    def test_win_rate_all_negative(self, negative_returns):
        """Test win rate with all negative returns."""
        result = stats.win_rate(negative_returns)
        assert result == 0.0


class TestRiskMetrics:
    """Test risk-related metrics."""

    def test_volatility(self, sample_returns):
        """Test volatility calculation."""
        result = stats.volatility(sample_returns)
        assert result > 0
        assert np.isfinite(result)

    def test_volatility_annualized(self, sample_returns):
        """Test annualized volatility."""
        daily_vol = stats.volatility(sample_returns, annualize=False)
        annual_vol = stats.volatility(sample_returns, annualize=True)
        # Annualized should be approximately sqrt(252) times daily
        expected_ratio = np.sqrt(252)
        actual_ratio = annual_vol / daily_vol
        np.testing.assert_almost_equal(actual_ratio, expected_ratio, decimal=5)

    def test_max_drawdown(self, sample_returns):
        """Test maximum drawdown calculation."""
        result = stats.max_drawdown(sample_returns)
        assert result <= 0  # Drawdown is always negative or zero
        assert result >= -1  # Cannot lose more than 100%

    def test_var(self, sample_returns):
        """Test Value at Risk calculation."""
        result = stats.var(sample_returns)
        assert result < 0  # VaR is typically negative (loss)

    def test_cvar(self, sample_returns):
        """Test Conditional Value at Risk (Expected Shortfall)."""
        var = stats.var(sample_returns)
        cvar = stats.cvar(sample_returns)
        # CVaR should be more extreme than VaR
        assert cvar <= var


class TestRatios:
    """Test risk-adjusted return ratios."""

    def test_sharpe(self, sample_returns):
        """Test Sharpe ratio calculation."""
        result = stats.sharpe(sample_returns)
        assert np.isfinite(result)

    def test_sharpe_with_rf(self, sample_returns):
        """Test Sharpe ratio with risk-free rate."""
        result_no_rf = stats.sharpe(sample_returns, rf=0)
        result_with_rf = stats.sharpe(sample_returns, rf=0.02)
        # Handle both scalar and Series results
        if isinstance(result_no_rf, pd.Series):
            result_no_rf = result_no_rf.iloc[0]
            result_with_rf = result_with_rf.iloc[0]
        # Higher rf should lower Sharpe ratio
        assert result_with_rf < result_no_rf

    def test_sortino(self, sample_returns):
        """Test Sortino ratio calculation."""
        result = stats.sortino(sample_returns)
        # Handle both scalar and Series results
        if isinstance(result, pd.Series):
            result = result.iloc[0]
        assert np.isfinite(result)

    def test_sortino_vs_sharpe(self, sample_returns):
        """Test that Sortino uses downside deviation."""
        sharpe = stats.sharpe(sample_returns)
        sortino = stats.sortino(sample_returns)
        # Handle both scalar and Series results
        if hasattr(sharpe, "values"):
            sharpe = float(sharpe.values[0]) if len(sharpe) > 0 else float(sharpe)
        if hasattr(sortino, "values"):
            sortino = float(sortino.values[0]) if len(sortino) > 0 else float(sortino)
        # They should be different (Sortino only penalizes downside)
        assert sharpe != sortino

    def test_calmar(self, sample_returns):
        """Test Calmar ratio calculation."""
        result = stats.calmar(sample_returns)
        assert np.isfinite(result)

    def test_omega(self, sample_returns):
        """Test Omega ratio calculation."""
        result = stats.omega(sample_returns)
        assert result > 0  # Omega is always positive

    def test_cagr(self, sample_returns):
        """Test CAGR calculation."""
        result = stats.cagr(sample_returns)
        assert np.isfinite(result)


class TestBenchmarkComparison:
    """Test benchmark comparison functions."""

    def test_greeks(self, sample_returns, sample_benchmark):
        """Test alpha/beta calculation."""
        result = stats.greeks(sample_returns, sample_benchmark)
        assert "alpha" in result
        assert "beta" in result
        assert np.isfinite(result["alpha"])
        assert np.isfinite(result["beta"])

    def test_r_squared(self, sample_returns, sample_benchmark):
        """Test R-squared calculation."""
        result = stats.r_squared(sample_returns, sample_benchmark)
        assert 0 <= result <= 1

    def test_information_ratio(self, sample_returns, sample_benchmark):
        """Test Information Ratio calculation."""
        result = stats.information_ratio(sample_returns, sample_benchmark)
        assert np.isfinite(result)

    def test_treynor_ratio(self, sample_returns, sample_benchmark):
        """Test Treynor Ratio calculation."""
        result = stats.treynor_ratio(sample_returns, sample_benchmark)
        assert np.isfinite(result)


class TestDrawdown:
    """Test drawdown-related functions."""

    def test_to_drawdown_series(self, sample_returns):
        """Test drawdown series conversion."""
        result = stats.to_drawdown_series(sample_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_returns)
        # All values should be <= 0 (drawdowns are negative)
        assert (result <= 0).all()

    def test_drawdown_details(self, sample_returns):
        """Test drawdown details extraction."""
        dd = stats.to_drawdown_series(sample_returns)
        result = stats.drawdown_details(dd)
        assert isinstance(result, pd.DataFrame)
        # Should have standard columns
        assert "start" in result.columns
        assert "end" in result.columns
        assert "max drawdown" in result.columns


class TestConsecutive:
    """Test consecutive wins/losses functions."""

    def test_consecutive_wins(self, sample_returns):
        """Test consecutive wins calculation."""
        result = stats.consecutive_wins(sample_returns)
        assert isinstance(result, (int, np.integer))
        assert result >= 0

    def test_consecutive_losses(self, sample_returns):
        """Test consecutive losses calculation."""
        result = stats.consecutive_losses(sample_returns)
        assert isinstance(result, (int, np.integer))
        assert result >= 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_returns(self):
        """Test handling of empty returns."""
        empty = pd.Series([], dtype=float)
        with pytest.raises(Exception):
            stats.sharpe(empty)

    def test_single_return(self):
        """Test handling of single return value."""
        single = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1))
        # Should not raise error
        result = stats.comp(single)
        assert np.isfinite(result)

    def test_nan_handling(self, sample_returns):
        """Test that NaN values are handled properly."""
        returns_with_nan = sample_returns.copy()
        returns_with_nan.iloc[10:20] = np.nan
        # Should still compute without error
        result = stats.sharpe(returns_with_nan)
        assert np.isfinite(result)

    def test_dataframe_input(self, sample_returns, sample_benchmark):
        """Test DataFrame input handling."""
        df = pd.DataFrame({"A": sample_returns, "B": sample_benchmark})
        result = stats.sharpe(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2


def _excess(returns, rf, periods=252):
    """Excess returns with an annual rf de-annualized to one period."""
    return returns - ((1 + rf) ** (1.0 / periods) - 1)


def _autocorr_penalty(x):
    """Lo (2002)-style penalty, written out independently of stats.py."""
    x = np.asarray(x.dropna())
    n = len(x)
    coef = abs(np.corrcoef(x[:-1], x[1:])[0, 1])
    k = np.arange(1, n)
    return np.sqrt(1 + 2 * np.sum((n - k) / n * coef**k))


class TestReferenceValues:
    """
    Exact expectations for the headline metrics.

    Each metric is checked twice: against its closed form evaluated with
    numpy on `daily_returns`, and against a hand-checkable literal on the
    five-point `tiny_returns`, so a formula change cannot hide behind an
    equally changed reference.
    """

    # --- sharpe -----------------------------------------------------------

    @pytest.mark.parametrize("rf", [0.0, 0.05])
    def test_sharpe_matches_closed_form(self, daily_returns, rf):
        x = _excess(daily_returns, rf)
        expected = x.mean() / x.std(ddof=1) * np.sqrt(252)

        assert stats.sharpe(daily_returns, rf=rf) == pytest.approx(expected, rel=1e-12)

    def test_sharpe_unannualized_is_the_per_period_ratio(self, daily_returns):
        expected = daily_returns.mean() / daily_returns.std(ddof=1)

        result = stats.sharpe(daily_returns, annualize=False)

        assert result == pytest.approx(expected, rel=1e-12)

    def test_smart_sharpe_applies_the_autocorrelation_penalty(self, daily_returns):
        x = daily_returns
        expected = x.mean() / (x.std(ddof=1) * _autocorr_penalty(x)) * np.sqrt(252)

        assert stats.sharpe(x, smart=True) == pytest.approx(expected, rel=1e-12)
        assert stats.smart_sharpe(x) == pytest.approx(expected, rel=1e-12)

    def test_sharpe_literal(self, tiny_returns):
        # mean 0.006, sample std sqrt(1.72e-3 / 4) = 0.0207364
        assert stats.sharpe(tiny_returns) == pytest.approx(4.593220484431883, rel=1e-12)
        assert stats.sharpe(tiny_returns, rf=0.05) == pytest.approx(
            4.444989216253766, rel=1e-12
        )

    def test_sharpe_requires_periods_with_rf(self, daily_returns):
        with pytest.raises(ValueError, match="periods"):
            stats.sharpe(daily_returns, rf=0.05, periods=None)

    # --- sortino ----------------------------------------------------------

    @pytest.mark.parametrize("rf", [0.0, 0.05])
    def test_sortino_matches_closed_form(self, daily_returns, rf):
        x = _excess(daily_returns, rf)
        downside = np.sqrt((x[x < 0] ** 2).sum() / x.count())
        expected = x.mean() / downside * np.sqrt(252)

        assert stats.sortino(daily_returns, rf=rf) == pytest.approx(expected, rel=1e-12)

    def test_smart_sortino_applies_the_autocorrelation_penalty(self, daily_returns):
        x = daily_returns
        downside = np.sqrt((x[x < 0] ** 2).sum() / x.count())
        expected = x.mean() / (downside * _autocorr_penalty(x)) * np.sqrt(252)

        assert stats.sortino(x, smart=True) == pytest.approx(expected, rel=1e-12)
        assert stats.smart_sortino(x) == pytest.approx(expected, rel=1e-12)

    def test_adjusted_sortino_is_sortino_over_root_two(self, daily_returns):
        expected = stats.sortino(daily_returns) / np.sqrt(2)

        assert stats.adjusted_sortino(daily_returns) == pytest.approx(
            expected, rel=1e-12
        )

    def test_sortino_literal(self, tiny_returns):
        # downside deviation sqrt((0.02^2 + 0.01^2) / 5) = 0.01
        expected = 0.006 / 0.01 * np.sqrt(252)

        assert stats.sortino(tiny_returns) == pytest.approx(expected, rel=1e-12)
        assert expected == pytest.approx(9.524704719832526, rel=1e-12)

    def test_sortino_without_losses_is_nan(self, positive_returns):
        assert np.isnan(stats.sortino(positive_returns))

    def test_sortino_requires_periods_with_rf(self, daily_returns):
        with pytest.raises(ValueError, match="periods"):
            stats.sortino(daily_returns, rf=0.05, periods=None)

    # --- cagr -------------------------------------------------------------

    def test_cagr_compounded_matches_closed_form(self, daily_returns):
        n = daily_returns.count()
        expected = (1 + daily_returns).prod() ** (252 / n) - 1

        assert stats.cagr(daily_returns) == pytest.approx(expected, rel=1e-12)

    def test_cagr_uncompounded_matches_closed_form(self, daily_returns):
        n = daily_returns.count()
        expected = (1 + daily_returns.sum()) ** (252 / n) - 1

        result = stats.cagr(daily_returns, compounded=False)

        assert result == pytest.approx(expected, rel=1e-12)

    def test_cagr_years_come_from_periods_not_the_calendar(self, weekly_returns):
        one_year = weekly_returns.iloc[:52] * 0 + 0.01

        assert stats.cagr(one_year, periods=52) == pytest.approx(
            1.01**52 - 1, rel=1e-12
        )

    def test_cagr_ignores_rf(self, daily_returns):
        # Documented long-standing behaviour: rf is accepted but not applied.
        assert stats.cagr(daily_returns, rf=0.05) == stats.cagr(daily_returns)

    def test_cagr_literal(self, tiny_returns):
        total = 1.01 * 0.98 * 1.03 * 0.99 * 1.02
        expected = total ** (252 / 5) - 1

        assert stats.cagr(tiny_returns) == pytest.approx(expected, rel=1e-12)
        assert expected == pytest.approx(3.325636719291219, rel=1e-12)

    # --- calmar -----------------------------------------------------------

    @pytest.mark.parametrize("compounded", [True, False])
    def test_calmar_is_cagr_over_max_drawdown(self, daily_returns, compounded):
        expected = stats.cagr(daily_returns, compounded=compounded) / abs(
            stats.max_drawdown(daily_returns)
        )

        result = stats.calmar(daily_returns, compounded=compounded)

        assert result == pytest.approx(expected, rel=1e-12)

    def test_calmar_literal(self, tiny_returns):
        # Max drawdown is the -2% second day: 1.01 -> 0.9898.
        expected = stats.cagr(tiny_returns) / 0.02

        assert stats.calmar(tiny_returns) == pytest.approx(expected, rel=1e-9)
        assert stats.calmar(tiny_returns) == pytest.approx(166.2818359645608, rel=1e-9)

    # --- omega ------------------------------------------------------------

    @pytest.mark.parametrize("rf", [0.0, 0.05])
    @pytest.mark.parametrize("required_return", [0.0, 0.05])
    def test_omega_matches_closed_form(self, daily_returns, rf, required_return):
        threshold = (1 + required_return) ** (1 / 252) - 1
        deviations = _excess(daily_returns, rf) - threshold
        expected = deviations[deviations > 0].sum() / -deviations[deviations < 0].sum()

        result = stats.omega(daily_returns, rf=rf, required_return=required_return)

        assert result == pytest.approx(expected, rel=1e-12)

    def test_omega_with_one_period_uses_the_threshold_as_is(self, daily_returns):
        deviations = daily_returns - 0.001
        expected = deviations[deviations > 0].sum() / -deviations[deviations < 0].sum()

        result = stats.omega(daily_returns, required_return=0.001, periods=1)

        assert result == pytest.approx(expected, rel=1e-12)

    def test_omega_literal(self, tiny_returns):
        # gains 0.01 + 0.03 + 0.02 over losses 0.02 + 0.01
        assert stats.omega(tiny_returns) == pytest.approx(2.0, rel=1e-12)

    # --- drawdown ---------------------------------------------------------

    def test_max_drawdown_literal(self):
        # Equity 1.10, 0.88, 0.924, 0.8316, 1.0811 -> trough 0.8316 / peak 1.10
        returns = pd.Series(
            [0.10, -0.20, 0.05, -0.10, 0.30],
            index=pd.date_range("2024-01-01", periods=5),
        )

        assert stats.max_drawdown(returns) == pytest.approx(-0.244, rel=1e-12)

    def test_max_drawdown_matches_the_equity_curve(self, daily_returns):
        equity = (1 + daily_returns).cumprod()
        expected = min((equity / equity.cummax()).min() - 1, equity.iloc[0] - 1, 0.0)

        assert stats.max_drawdown(daily_returns) == pytest.approx(expected, rel=1e-12)

    # --- tail risk --------------------------------------------------------

    def test_value_at_risk_is_the_normal_quantile(self, daily_returns):
        mu, sd = daily_returns.mean(), daily_returns.std(ddof=1)
        expected = norm.ppf(0.05, mu, sd)

        assert stats.value_at_risk(daily_returns) == pytest.approx(expected, rel=1e-12)
        assert stats.var(daily_returns) == pytest.approx(expected, rel=1e-12)

    def test_value_at_risk_accepts_a_percentage(self, daily_returns):
        assert stats.value_at_risk(daily_returns, confidence=95) == pytest.approx(
            stats.value_at_risk(daily_returns, confidence=0.95), rel=1e-12
        )

    def test_cvar_is_the_normal_expected_shortfall(self, daily_returns):
        mu, sd = daily_returns.mean(), daily_returns.std(ddof=1)
        expected = mu - sd * norm.pdf(norm.ppf(0.05)) / 0.05

        assert stats.cvar(daily_returns) == pytest.approx(expected, rel=1e-12)
        assert stats.conditional_value_at_risk(
            daily_returns, confidence=95
        ) == pytest.approx(expected, rel=1e-12)

    def test_var_and_cvar_literals(self, tiny_returns):
        assert stats.var(tiny_returns) == pytest.approx(-0.028108410770087598, rel=1e-9)
        assert stats.cvar(tiny_returns) == pytest.approx(-0.03677332316163571, rel=1e-9)

    # --- period statistics ------------------------------------------------

    def test_win_rate_ignores_flat_periods(self):
        returns = pd.Series(
            [0.01, 0.0, -0.02, 0.03, 0.0, -0.01],
            index=pd.date_range("2024-01-01", periods=6),
        )

        # two wins out of four non-zero periods
        assert stats.win_rate(returns) == pytest.approx(0.5)

    def test_win_rate_literal(self, tiny_returns):
        assert stats.win_rate(tiny_returns) == pytest.approx(0.6)

    def test_ghpr_is_the_geometric_mean(self, daily_returns):
        n = daily_returns.count()
        expected = (1 + daily_returns).prod() ** (1 / n) - 1

        assert stats.ghpr(daily_returns) == pytest.approx(expected, rel=1e-12)
        assert stats.ghpr(daily_returns) == pytest.approx(
            stats.geometric_mean(daily_returns), rel=1e-12
        )

    def test_ghpr_literal(self, tiny_returns):
        assert stats.ghpr(tiny_returns) == pytest.approx(
            0.0058286643890854695, rel=1e-9
        )

    def test_exposure_counts_non_zero_periods(self):
        returns = pd.Series(
            [0.01, 0.0, 0.02, 0.0, 0.03],
            index=pd.date_range("2024-01-01", periods=5),
        )

        assert stats.exposure(returns) == pytest.approx(0.6)

    def test_kelly_criterion_literal(self, tiny_returns):
        # p = 0.6, b = 0.02 / 0.015 = 4/3  ->  f* = p - (1 - p) / b = 0.3
        assert stats.kelly_criterion(tiny_returns) == pytest.approx(0.3, rel=1e-12)

    def test_compsum_is_the_cumulative_product(self, tiny_returns):
        expected = [
            0.01,
            1.01 * 0.98 - 1,
            1.01 * 0.98 * 1.03 - 1,
            1.01 * 0.98 * 1.03 * 0.99 - 1,
            1.01 * 0.98 * 1.03 * 0.99 * 1.02 - 1,
        ]

        np.testing.assert_allclose(stats.compsum(tiny_returns), expected, rtol=1e-12)
        assert stats.comp(tiny_returns) == pytest.approx(expected[-1], rel=1e-12)


class TestKnownBugs:
    """
    Behaviour believed to be wrong, pinned as strict xfails and NOT fixed.

    Each test asserts the intended behaviour. `strict=True` makes the suite
    fail loudly once the bug is fixed, so the marker must then be removed.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="std() of constant returns is float noise (~1e-19), not 0, "
        "so sharpe reports ~1e17 instead of NaN (sortino already returns NaN)",
    )
    def test_sharpe_of_constant_returns_is_nan(self):
        constant = pd.Series([0.01] * 10, index=pd.date_range("2024-01-01", periods=10))

        assert np.isnan(stats.sharpe(constant))

    @pytest.mark.xfail(
        strict=True,
        reason="same root cause: volatility of constant returns is ~2.9e-17, not 0",
    )
    def test_volatility_of_constant_returns_is_zero(self):
        constant = pd.Series([0.01] * 10, index=pd.date_range("2024-01-01", periods=10))

        assert stats.volatility(constant) == 0.0

    @pytest.mark.xfail(
        strict=True,
        reason="calmar divides by a zero max drawdown and returns inf; other "
        "ratios (sortino, omega, payoff_ratio) return NaN for a zero denominator",
    )
    def test_calmar_without_drawdown_is_nan(self, positive_returns):
        assert np.isnan(stats.calmar(positive_returns))

    @pytest.mark.xfail(
        strict=True,
        reason="exposure rounds up with ceil(ex * 100) on a float, so 7/100 "
        "becomes 7.000000000000001 and is reported as 0.08",
    )
    @pytest.mark.parametrize("invested", [7, 14, 28, 55, 56])
    def test_exposure_is_not_rounded_past_the_true_share(self, invested):
        returns = pd.Series(
            [0.01] * invested + [0.0] * (100 - invested),
            index=pd.date_range("2024-01-01", periods=100),
        )

        assert stats.exposure(returns) == pytest.approx(invested / 100)

    @pytest.mark.xfail(
        strict=True,
        reason="ulcer_index divides by returns.shape[0] - 1, which counts NaN "
        "rows, so gaps change the result (and serenity_index, built on it)",
    )
    @pytest.mark.parametrize("metric", ["ulcer_index", "serenity_index"])
    @pytest.mark.parametrize(
        "gap",
        [slice(50, 90), slice(0, 40), slice(360, 400), slice(3, 400, 7)],
        ids=["middle", "leading", "trailing", "scattered"],
    )
    def test_gap_metric_matches_dropping_the_gap(self, daily_returns, metric, gap):
        gapped = daily_returns.copy()
        gapped.iloc[gap] = np.nan
        baseline = daily_returns.drop(daily_returns.index[gap])
        fn = getattr(stats, metric)

        assert float(fn(gapped)) == pytest.approx(float(fn(baseline)), rel=1e-9)

    @pytest.mark.xfail(
        strict=True,
        reason="value_at_risk returns an unlabelled ndarray for DataFrame input; "
        "conditional_value_at_risk returns a Series indexed by column",
    )
    @pytest.mark.parametrize("metric", ["value_at_risk", "var"])
    def test_value_at_risk_dataframe_is_labelled_per_column(
        self, daily_returns, metric
    ):
        frame = pd.DataFrame({"a": daily_returns, "b": daily_returns * 2})

        result = getattr(stats, metric)(frame)

        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b"]

    @pytest.mark.xfail(
        strict=True,
        reason="treynor_ratio silently keeps only the first column of a DataFrame",
    )
    def test_treynor_ratio_dataframe_is_per_column(self, paired_returns_benchmark):
        strat, bench = paired_returns_benchmark
        frame = pd.DataFrame({"a": strat, "b": strat * 2})

        result = stats.treynor_ratio(frame, bench)

        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b"]

    @pytest.mark.xfail(
        strict=True,
        reason="profit_factor maps inf to 0 for DataFrame columns with no losses, "
        "while the Series path returns inf",
    )
    def test_profit_factor_dataframe_matches_series(self, positive_returns):
        """Test profit_factor gives the same value for Series and DataFrame input."""
        df = pd.DataFrame({"A": positive_returns})
        assert stats.profit_factor(df)["A"] == stats.profit_factor(positive_returns)


class TestAverageReturns:
    """Test average and expected return statistics."""

    def test_avg_return(self, sample_returns):
        """Test avg_return is the mean of non-zero returns."""
        result = stats.avg_return(sample_returns)
        assert np.isfinite(result)
        assert np.isclose(result, sample_returns[sample_returns != 0].mean())

    def test_avg_return_aggregated(self, sample_returns):
        """Test avg_return with monthly aggregation."""
        result = stats.avg_return(sample_returns, aggregate="ME")
        assert np.isfinite(result)

    def test_avg_win_and_loss_signs(self, sample_returns):
        """Test avg_win > 0 and avg_loss < 0 on mixed returns."""
        assert stats.avg_win(sample_returns) > 0
        assert stats.avg_loss(sample_returns) < 0

    def test_avg_win_loss_aggregated(self, sample_returns):
        """Test avg_win/avg_loss with monthly aggregation."""
        assert stats.avg_win(sample_returns, aggregate="ME") > 0
        assert stats.avg_loss(sample_returns, aggregate="ME") < 0

    def test_avg_loss_all_positive_is_nan(self, positive_returns):
        """Test avg_loss is NaN when there are no losses."""
        assert np.isnan(stats.avg_loss(positive_returns))
        assert stats.avg_win(positive_returns) > 0

    def test_avg_win_all_negative_is_nan(self, negative_returns):
        """Test avg_win is NaN when there are no wins."""
        assert np.isnan(stats.avg_win(negative_returns))
        assert stats.avg_loss(negative_returns) < 0

    def test_expected_return(self, sample_returns):
        """Test expected_return is the geometric mean per period."""
        result = stats.expected_return(sample_returns)
        assert np.isfinite(result)
        expected = (1 + sample_returns).prod() ** (1 / len(sample_returns)) - 1
        assert np.isclose(result, expected)

    def test_expected_return_sign(self, positive_returns, negative_returns):
        """Test expected_return sign on one-sided returns."""
        assert stats.expected_return(positive_returns) > 0
        assert stats.expected_return(negative_returns) < 0

    def test_geometric_mean_and_ghpr_aliases(self, sample_returns):
        """Test geometric_mean and ghpr match expected_return."""
        expected = stats.expected_return(sample_returns)
        assert np.isclose(stats.geometric_mean(sample_returns), expected)
        assert np.isclose(stats.ghpr(sample_returns), expected)

    def test_best_worst(self, sample_returns):
        """Test best/worst are the max/min returns."""
        assert np.isclose(stats.best(sample_returns), sample_returns.max())
        assert np.isclose(stats.worst(sample_returns), sample_returns.min())
        assert stats.best(sample_returns, aggregate="ME") > stats.worst(
            sample_returns, aggregate="ME"
        )


class TestProfitMetrics:
    """Test payoff, profit and Kelly metrics."""

    def test_payoff_ratio(self, sample_returns):
        """Test payoff_ratio equals avg_win / |avg_loss|."""
        result = stats.payoff_ratio(sample_returns)
        expected = stats.avg_win(sample_returns) / abs(stats.avg_loss(sample_returns))
        assert result > 0
        assert np.isclose(result, expected)

    def test_payoff_ratio_no_losses_is_nan(self, positive_returns):
        """Test payoff_ratio is NaN without losses."""
        assert np.isnan(stats.payoff_ratio(positive_returns))

    def test_win_loss_ratio_alias(self, sample_returns):
        """Test win_loss_ratio matches payoff_ratio."""
        assert np.isclose(
            stats.win_loss_ratio(sample_returns), stats.payoff_ratio(sample_returns)
        )

    def test_profit_factor(self, sample_returns):
        """Test profit_factor equals gross wins / gross losses."""
        result = stats.profit_factor(sample_returns)
        wins = sample_returns[sample_returns >= 0].sum()
        losses = abs(sample_returns[sample_returns < 0].sum())
        assert result > 0
        assert np.isclose(result, wins / losses)

    def test_profit_factor_one_sided(self, positive_returns, negative_returns):
        """Test profit_factor is inf without losses and 0 without wins."""
        assert stats.profit_factor(positive_returns) == float("inf")
        assert stats.profit_factor(negative_returns) == 0

    def test_profit_ratio(self, sample_returns):
        """Test profit_ratio is positive and finite on mixed returns."""
        result = stats.profit_ratio(sample_returns)
        assert np.isfinite(result)
        assert result > 0

    def test_profit_ratio_one_sided(self, positive_returns, negative_returns):
        """Test profit_ratio is NaN without losses and 0 without wins."""
        assert np.isnan(stats.profit_ratio(positive_returns))
        assert stats.profit_ratio(negative_returns) == 0

    def test_profit_ratio_dataframe(self, sample_returns, sample_benchmark):
        """Test profit_ratio returns one value per column."""
        df = pd.DataFrame({"A": sample_returns, "B": sample_benchmark})
        result = stats.profit_ratio(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_kelly_criterion(self, sample_returns):
        """Test kelly_criterion is consistent with win rate and payoff ratio."""
        result = stats.kelly_criterion(sample_returns)
        win_prob = stats.win_rate(sample_returns)
        payoff = stats.payoff_ratio(sample_returns)
        assert np.isfinite(result)
        assert result <= 1
        assert np.isclose(result, win_prob - (1 - win_prob) / payoff)

    def test_kelly_criterion_no_losses_is_nan(self, positive_returns):
        """Test kelly_criterion is NaN when payoff ratio is undefined."""
        assert np.isnan(stats.kelly_criterion(positive_returns))

    def test_cpc_index(self, sample_returns):
        """Test cpc_index = profit_factor * win_rate * win_loss_ratio."""
        expected = (
            stats.profit_factor(sample_returns)
            * stats.win_rate(sample_returns)
            * stats.win_loss_ratio(sample_returns)
        )
        assert np.isclose(stats.cpc_index(sample_returns), expected)

    def test_common_sense_ratio(self, sample_returns):
        """Test common_sense_ratio = profit_factor * tail_ratio."""
        expected = stats.profit_factor(sample_returns) * stats.tail_ratio(
            sample_returns
        )
        assert np.isclose(stats.common_sense_ratio(sample_returns), expected)

    def test_risk_return_ratio(self, sample_returns):
        """Test risk_return_ratio = mean / std."""
        expected = sample_returns.mean() / sample_returns.std()
        assert np.isclose(stats.risk_return_ratio(sample_returns), expected)

    def test_risk_return_ratio_dataframe(self, sample_returns, sample_benchmark):
        """Test risk_return_ratio returns one value per column."""
        df = pd.DataFrame({"A": sample_returns, "B": sample_benchmark})
        result = stats.risk_return_ratio(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_risk_of_ruin(self, sample_returns, positive_returns, negative_returns):
        """Test risk_of_ruin is a probability with expected extremes."""
        result = stats.risk_of_ruin(sample_returns)
        assert 0 <= result <= 1
        assert stats.risk_of_ruin(positive_returns) == 0
        assert stats.risk_of_ruin(negative_returns) == 1
        assert stats.ror(sample_returns) == result


class TestDistributionStats:
    """Test distribution shape statistics."""

    def test_skew(self, sample_returns):
        """Test skew is near zero for symmetric random data."""
        result = stats.skew(sample_returns)
        assert np.isfinite(result)
        assert abs(result) < 0.5

    def test_skew_one_sided(self, positive_returns, negative_returns):
        """Test skew sign on half-normal returns."""
        assert stats.skew(positive_returns) > 0
        assert stats.skew(negative_returns) < 0

    def test_kurtosis(self, sample_returns):
        """Test excess kurtosis is finite and near zero for normal data."""
        result = stats.kurtosis(sample_returns)
        assert np.isfinite(result)
        assert abs(result) < 1

    def test_tail_ratio(self, sample_returns):
        """Test tail_ratio = |95th pct / 5th pct|."""
        result = stats.tail_ratio(sample_returns)
        expected = abs(sample_returns.quantile(0.95) / sample_returns.quantile(0.05))
        assert result > 0
        assert np.isclose(result, expected)

    def test_tail_ratio_dataframe(self, sample_returns, sample_benchmark):
        """Test tail_ratio returns positive per-column values for DataFrames."""
        df = pd.DataFrame({"A": sample_returns, "B": sample_benchmark})
        result = stats.tail_ratio(df)
        assert isinstance(result, pd.Series)
        assert (result > 0).all()

    def test_distribution(self, sample_returns):
        """Test distribution splits each period into values and outliers."""
        result = stats.distribution(sample_returns)
        assert set(result) == {"Daily", "Weekly", "Monthly", "Quarterly", "Yearly"}
        daily = result["Daily"]
        assert len(daily["values"]) + len(daily["outliers"]) == len(sample_returns)

    def test_distribution_dataframe_warns(self, sample_returns):
        """Test distribution warns and uses the first DataFrame column."""
        with pytest.warns(UserWarning):
            result = stats.distribution(sample_returns.to_frame())
        assert "Daily" in result


class TestRiskAliases:
    """Test VaR / CVaR aliases."""

    def test_value_at_risk_matches_var(self, sample_returns):
        """Test value_at_risk matches var."""
        assert np.isclose(
            stats.value_at_risk(sample_returns), stats.var(sample_returns)
        )

    def test_conditional_value_at_risk_matches_cvar(self, sample_returns):
        """Test conditional_value_at_risk and expected_shortfall match cvar."""
        expected = stats.cvar(sample_returns)
        assert np.isclose(stats.conditional_value_at_risk(sample_returns), expected)
        assert np.isclose(stats.expected_shortfall(sample_returns), expected)

    def test_cvar_beyond_var(self, sample_returns):
        """Test CVaR is at least as severe as VaR."""
        assert stats.conditional_value_at_risk(sample_returns) <= stats.value_at_risk(
            sample_returns
        )


class TestPainRecovery:
    """Test pain, ulcer and recovery metrics."""

    def test_gain_to_pain_ratio(self, sample_returns):
        """Test gain_to_pain_ratio = sum / |sum of losses|."""
        result = stats.gain_to_pain_ratio(sample_returns)
        losses = abs(sample_returns[sample_returns < 0].sum())
        assert np.isclose(result, sample_returns.sum() / losses)

    def test_gain_to_pain_ratio_monthly(self, sample_returns):
        """Test gain_to_pain_ratio with monthly resolution is finite."""
        assert np.isfinite(stats.gain_to_pain_ratio(sample_returns, resolution="ME"))

    def test_gain_to_pain_ratio_one_sided(self, positive_returns, negative_returns):
        """Test gain_to_pain_ratio is NaN without losses and -1 without wins."""
        assert np.isnan(stats.gain_to_pain_ratio(positive_returns))
        assert np.isclose(stats.gain_to_pain_ratio(negative_returns), -1)

    def test_recovery_factor(self, sample_returns):
        """Test recovery_factor = |total return| / |max drawdown|."""
        result = stats.recovery_factor(sample_returns)
        expected = abs(sample_returns.sum()) / abs(stats.max_drawdown(sample_returns))
        assert result > 0
        assert np.isclose(result, expected)

    def test_recovery_factor_no_drawdown_is_nan(self, positive_returns):
        """Test recovery_factor is NaN when there is no drawdown."""
        assert np.isnan(stats.recovery_factor(positive_returns))

    def test_ulcer_index(self, sample_returns, negative_returns):
        """Test ulcer_index is non-negative and grows with drawdown depth."""
        result = stats.ulcer_index(sample_returns)
        assert result >= 0
        assert stats.ulcer_index(negative_returns) > result

    def test_ulcer_index_no_drawdown(self, positive_returns):
        """Test ulcer_index is zero without drawdowns."""
        assert stats.ulcer_index(positive_returns) == 0

    def test_upi_sign_matches_cagr(self, sample_returns, negative_returns):
        """Test upi/ulcer_performance_index sign is consistent with CAGR."""
        for returns in (sample_returns, negative_returns):
            result = stats.upi(returns)
            assert np.isfinite(result)
            assert np.sign(result) == np.sign(stats.cagr(returns))
            assert np.isclose(result, stats.ulcer_performance_index(returns))

    def test_upi_no_drawdown_is_nan(self, positive_returns):
        """Test upi is NaN when ulcer index is zero."""
        assert np.isnan(stats.upi(positive_returns))

    def test_upi_dataframe(self, sample_returns, sample_benchmark):
        """Test upi returns one value per column for DataFrames."""
        df = pd.DataFrame({"A": sample_returns, "B": sample_benchmark})
        result = stats.upi(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_serenity_index(self, sample_returns):
        """Test serenity_index is finite with sign matching total return."""
        result = stats.serenity_index(sample_returns)
        assert np.isfinite(result)
        assert np.sign(result) == np.sign(sample_returns.sum())

    def test_serenity_index_dataframe(self, sample_returns, sample_benchmark):
        """Test serenity_index returns one value per column for DataFrames."""
        df = pd.DataFrame({"A": sample_returns, "B": sample_benchmark})
        result = stats.serenity_index(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_rar(self, sample_returns):
        """Test rar = CAGR / exposure."""
        expected = stats.cagr(sample_returns) / stats.exposure(sample_returns)
        assert np.isclose(stats.rar(sample_returns), expected)


class TestMonthlyReturns:
    """Test monthly returns table."""

    def test_shape_and_columns(self, sample_returns):
        """Test table has one row per year and 12 months + EOY."""
        result = stats.monthly_returns(sample_returns)
        months = [
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
            "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
        ]  # fmt: skip
        assert list(result.columns) == months + ["EOY"]
        assert list(result.index) == ["2020", "2021"]

    def test_months_are_compounded(self, sample_returns):
        """Test each populated month equals the compounded monthly return."""
        result = stats.monthly_returns(sample_returns)
        monthly = (1 + sample_returns).resample("ME").prod() - 1
        assert len(monthly) == 17
        for ts, value in monthly.items():
            cell = result.loc[str(ts.year), ts.strftime("%b").upper()]
            assert np.isclose(cell, value)

    def test_missing_months_are_zero(self, sample_returns):
        """Test months without data are filled with zero."""
        result = stats.monthly_returns(sample_returns)
        assert (
            result.loc["2021", ["JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]] == 0
        ).all()

    def test_eoy_is_compounded_year(self, sample_returns):
        """Test EOY equals the compounded yearly return."""
        result = stats.monthly_returns(sample_returns)
        for year in ("2020", "2021"):
            expected = (1 + sample_returns[year]).prod() - 1
            assert np.isclose(result.loc[year, "EOY"], expected)

    def test_no_eoy_and_simple_sum(self, sample_returns):
        """Test eoy=False drops EOY and compounded=False sums returns."""
        result = stats.monthly_returns(sample_returns, eoy=False, compounded=False)
        assert "EOY" not in result.columns
        assert np.isclose(result.loc["2020", "JAN"], sample_returns["2020-01"].sum())

    def test_dataframe_warns(self, sample_returns):
        """Test DataFrame input warns and uses the first column."""
        with pytest.warns(UserWarning):
            result = stats.monthly_returns(sample_returns.to_frame())
        pd.testing.assert_frame_equal(result, stats.monthly_returns(sample_returns))


class TestOutliers:
    """Test outlier filtering and outlier ratios."""

    def test_outliers_above_quantile(self, sample_returns):
        """Test outliers returns only values above the quantile."""
        threshold = sample_returns.quantile(0.95)
        result = stats.outliers(sample_returns)
        assert (result > threshold).all()
        assert len(result) == (sample_returns > threshold).sum()

    def test_outliers_custom_quantile(self, sample_returns):
        """Test lower quantile yields more outliers."""
        assert len(stats.outliers(sample_returns, 0.9)) > len(
            stats.outliers(sample_returns, 0.95)
        )

    def test_remove_outliers_complements_outliers(self, sample_returns):
        """Test remove_outliers keeps only values below the quantile."""
        threshold = sample_returns.quantile(0.95)
        result = stats.remove_outliers(sample_returns)
        assert (result < threshold).all()
        assert len(result) + len(stats.outliers(sample_returns)) == len(sample_returns)

    def test_outlier_win_ratio(self, sample_returns):
        """Test outlier_win_ratio = 99th pct / mean positive return."""
        result = stats.outlier_win_ratio(sample_returns)
        expected = (
            sample_returns.quantile(0.99) / sample_returns[sample_returns >= 0].mean()
        )
        assert result > 1
        assert np.isclose(result, expected)

    def test_outlier_loss_ratio(self, sample_returns):
        """Test outlier_loss_ratio = 1st pct / mean negative return."""
        result = stats.outlier_loss_ratio(sample_returns)
        expected = (
            sample_returns.quantile(0.01) / sample_returns[sample_returns < 0].mean()
        )
        assert result > 1
        assert np.isclose(result, expected)

    def test_outlier_ratios_one_sided(self, positive_returns, negative_returns):
        """Test outlier ratios are NaN when the relevant side is empty."""
        assert np.isnan(stats.outlier_loss_ratio(positive_returns))
        assert np.isnan(stats.outlier_win_ratio(negative_returns))

    def test_outlier_ratios_dataframe(self, sample_returns, sample_benchmark):
        """Test outlier ratios return per-column Series for DataFrames."""
        df = pd.DataFrame({"A": sample_returns, "B": sample_benchmark})
        assert len(stats.outlier_win_ratio(df)) == 2
        assert len(stats.outlier_loss_ratio(df)) == 2


class TestRollingAndSmart:
    """Test rolling, smart and probabilistic ratio variants."""

    def test_rolling_volatility(self, sample_returns):
        """Test rolling_volatility has the window warm-up and is positive."""
        result = stats.rolling_volatility(sample_returns, rolling_period=126)
        assert len(result) == len(sample_returns)
        assert result.iloc[:125].isna().all()
        assert (result.dropna() > 0).all()

    def test_implied_volatility(self, sample_returns):
        """Test implied_volatility rolling and non-annualized outputs."""
        rolling = stats.implied_volatility(sample_returns)
        assert isinstance(rolling, pd.Series)
        assert (rolling.dropna() > 0).all()
        assert stats.implied_volatility(sample_returns, annualize=False) > 0

    def test_autocorr_penalty(self, sample_returns):
        """Test autocorr_penalty is >= 1 and 1 for tiny inputs."""
        assert stats.autocorr_penalty(sample_returns) >= 1
        assert stats.autocorr_penalty(sample_returns.iloc[:1]) == 1.0
        assert stats.autocorr_penalty(sample_returns.to_frame()) >= 1

    def test_autocorr_penalty_constant_series(self):
        """Test autocorr_penalty is 1 when correlation is undefined."""
        constant = pd.Series([0.01] * 10, index=pd.date_range("2020-01-01", periods=10))
        assert stats.autocorr_penalty(constant) == 1.0

    def test_smart_sharpe(self, sample_returns):
        """Test smart_sharpe is sharpe deflated by autocorr penalty."""
        expected = stats.sharpe(sample_returns) / stats.autocorr_penalty(sample_returns)
        assert np.isclose(stats.smart_sharpe(sample_returns), expected)

    def test_smart_sortino(self, sample_returns):
        """Test smart_sortino has smaller magnitude than sortino."""
        result = stats.smart_sortino(sample_returns)
        assert np.isfinite(result)
        assert abs(result) <= abs(stats.sortino(sample_returns))

    def test_adjusted_sortino(self, sample_returns):
        """Test adjusted_sortino = sortino / sqrt(2)."""
        expected = stats.sortino(sample_returns) / np.sqrt(2)
        assert np.isclose(stats.adjusted_sortino(sample_returns), expected)

    def test_rolling_sharpe(self, sample_returns):
        """Test rolling_sharpe length and finiteness."""
        result = stats.rolling_sharpe(sample_returns)
        assert len(result) == len(sample_returns)
        assert np.isfinite(result.dropna()).all()

    def test_rolling_sortino(self, sample_returns):
        """Test rolling_sortino length and finiteness."""
        result = stats.rolling_sortino(sample_returns)
        assert len(result) == len(sample_returns)
        assert np.isfinite(result.dropna()).all()

    def test_rolling_requires_periods_with_rf(self, sample_returns):
        """Test rolling ratios require periods_per_year when rf != 0."""
        with pytest.raises(Exception):
            stats.rolling_sharpe(sample_returns, rf=0.02, periods_per_year=None)
        with pytest.raises(Exception):
            stats.rolling_sortino(sample_returns, rf=0.02, periods_per_year=None)

    def test_probabilistic_ratios(self, sample_returns):
        """Test probabilistic ratios are probabilities."""
        for fn in (
            stats.probabilistic_sharpe_ratio,
            stats.probabilistic_sortino_ratio,
            stats.probabilistic_adjusted_sortino_ratio,
        ):
            assert 0 <= fn(sample_returns) <= 1

    def test_probabilistic_ratio_matches_wrapper(self, sample_returns):
        """Test probabilistic_ratio(base='sharpe') matches the PSR wrapper."""
        assert np.isclose(
            stats.probabilistic_ratio(sample_returns, base="sharpe"),
            stats.probabilistic_sharpe_ratio(sample_returns),
        )

    def test_probabilistic_ratio_invalid_base(self, sample_returns):
        """Test probabilistic_ratio rejects unknown bases."""
        with pytest.raises(ValueError):
            stats.probabilistic_ratio(sample_returns, base="invalid")

    def test_pct_rank(self, sample_returns):
        """Test pct_rank stays within (0, 100]."""
        prices = (1 + sample_returns).cumprod()
        result = stats.pct_rank(prices, window=60).dropna()
        assert len(result) > 0
        assert ((result > 0) & (result <= 100)).all()


class TestBenchmarkExtras:
    """Test additional benchmark comparison functions."""

    def test_r2_alias(self, sample_returns, sample_benchmark):
        """Test r2 matches r_squared."""
        assert np.isclose(
            stats.r2(sample_returns, sample_benchmark),
            stats.r_squared(sample_returns, sample_benchmark),
        )

    def test_rolling_greeks(self, sample_returns, sample_benchmark):
        """Test rolling_greeks returns beta/alpha aligned to returns."""
        result = stats.rolling_greeks(sample_returns, sample_benchmark, periods=60)
        assert list(result.columns) == ["beta", "alpha"]
        assert result.index.equals(sample_returns.index)
        assert np.isfinite(result.dropna().values).all()

    def test_compare(self, sample_returns, sample_benchmark):
        """Test compare aggregates both series and flags outperformance."""
        result = stats.compare(sample_returns, sample_benchmark, aggregate="ME")
        assert list(result.columns) == ["Benchmark", "Returns", "Multiplier", "Won"]
        assert set(result["Won"]) <= {"+", "-"}
        won = result["Returns"] >= result["Benchmark"]
        assert ((result["Won"] == "+") == won).all()

    def test_compare_round_vals(self, sample_returns, sample_benchmark):
        """Test compare rounds numeric output."""
        result = stats.compare(
            sample_returns, sample_benchmark, aggregate="ME", round_vals=2
        )
        assert np.allclose(result["Returns"], result["Returns"].round(2))

    def test_compare_dataframe(self, sample_returns, sample_benchmark):
        """Test compare with multi-column returns."""
        df = pd.DataFrame({"A": sample_returns, "B": sample_returns * 0.5})
        result = stats.compare(df, sample_benchmark, aggregate="ME")
        assert list(result.columns) == ["Benchmark", "Returns_0", "Returns_1"]

    def test_compare_tz_aware(self, sample_returns, sample_benchmark):
        """Test compare handles timezone-aware inputs."""
        returns = sample_returns.tz_localize("UTC")
        benchmark = sample_benchmark.tz_localize("UTC")
        result = stats.compare(returns, benchmark, aggregate="ME")
        assert len(result) > 0


class TestMonteCarloWrappers:
    """Test stats-level Monte Carlo wrappers."""

    def test_montecarlo_sharpe(self, sample_returns):
        """Test montecarlo_sharpe summary is internally ordered."""
        result = stats.montecarlo_sharpe(sample_returns, sims=50, seed=42)
        assert result["min"] <= result["percentile_5"] <= result["median"]
        assert result["median"] <= result["percentile_95"] <= result["max"]

    def test_montecarlo_drawdown(self, sample_returns):
        """Test montecarlo_drawdown values are non-positive."""
        result = stats.montecarlo_drawdown(sample_returns, sims=50, seed=42)
        assert result["max"] <= 0
        assert result["min"] <= result["max"]

    def test_montecarlo_cagr(self, sample_returns):
        """Test montecarlo_cagr summary is finite and ordered."""
        result = stats.montecarlo_cagr(sample_returns, sims=50, seed=42)
        assert np.isfinite(result["mean"])
        assert result["min"] <= result["median"] <= result["max"]

    def test_montecarlo_dataframe_uses_first_column(
        self, sample_returns, sample_benchmark
    ):
        """Test Monte Carlo wrappers accept DataFrames."""
        df = pd.DataFrame({"A": sample_returns, "B": sample_benchmark})
        mc = stats.montecarlo(df, sims=20, seed=1)
        assert mc.data.shape[1] == 20
        assert np.isfinite(stats.montecarlo_sharpe(df, sims=20, seed=1)["mean"])
        assert np.isfinite(stats.montecarlo_cagr(df, sims=20, seed=1)["mean"])


class TestNewFunctionEdgeCases:
    """Edge cases for the functions covered above."""

    @pytest.fixture
    def empty(self):
        return pd.Series([], dtype=float, index=pd.DatetimeIndex([]))

    @pytest.mark.parametrize(
        "fn",
        [
            "avg_return",
            "avg_win",
            "avg_loss",
            "payoff_ratio",
            "kelly_criterion",
            "skew",
            "kurtosis",
            "tail_ratio",
            "gain_to_pain_ratio",
            "outlier_win_ratio",
            "outlier_loss_ratio",
        ],
    )
    def test_empty_returns_nan(self, empty, fn):
        """Test these functions return NaN on empty input."""
        assert np.isnan(getattr(stats, fn)(empty))

    @pytest.mark.parametrize("fn", ["ulcer_index", "recovery_factor", "rar", "upi"])
    def test_empty_returns_raise(self, empty, fn):
        """Test these functions reject empty input."""
        with pytest.raises(Exception):
            getattr(stats, fn)(empty)

    def test_empty_profit_factor_zero(self, empty):
        """Test profit_factor is 0 on empty input."""
        assert stats.profit_factor(empty) == 0

    def test_one_sided_finite_metrics(self, positive_returns, negative_returns):
        """Test sign of always-defined metrics on one-sided inputs."""
        assert stats.avg_return(positive_returns) > 0
        assert stats.avg_return(negative_returns) < 0
        assert stats.tail_ratio(positive_returns) > 1
        assert stats.tail_ratio(negative_returns) < 1
        assert stats.ulcer_index(negative_returns) > 0
