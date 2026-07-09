"""
Tests for helpers/df_ops.py — prepare_df, split_df, clean_df, downsample_min_gap.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from helpers.df_ops import prepare_df, split_df, clean_df, downsample_min_gap


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_raw_df(n=200, jd_start=2440000.0, step=5.0, seed=0):
    """Minimal two-column (JD, sind) raw DataFrame."""
    np.random.seed(seed)
    jd   = jd_start + np.arange(n) * step
    sind = 0.2 + 0.01 * np.sin(2 * np.pi * np.arange(n) / 50) + np.random.normal(0, 1e-4, n)
    return pd.DataFrame({"jd": jd, "sind": sind})


def _prepared(n=200, **kw):
    return prepare_df(_make_raw_df(n=n), **kw)


# ─────────────────────────────────────────────────────────────────────────────
# prepare_df
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareDF:

    def test_output_columns_present(self):
        df = _prepared()
        for col in ("JD", "sind", "year", "day"):
            assert col in df.columns, f"Missing column: {col}"

    def test_length_preserved_clean_data(self):
        df = _prepared(n=150)
        assert len(df) == 150

    def test_relative_jd_starts_at_zero(self):
        df = prepare_df(_make_raw_df(), relative=True)
        assert df["JD"].iloc[0] == pytest.approx(0.0)

    def test_non_relative_jd_unchanged(self):
        df = prepare_df(_make_raw_df(jd_start=2440000.0), relative=False)
        assert df["JD"].iloc[0] == pytest.approx(2440000.0)

    def test_drops_nan_sind_rows(self):
        raw = _make_raw_df(n=50)
        raw.iloc[5, 1] = np.nan
        df = prepare_df(raw)
        assert len(df) == 49
        assert not df["sind"].isna().any()

    def test_drops_nan_jd_rows(self):
        raw = _make_raw_df(n=50)
        raw.iloc[3, 0] = np.nan
        df = prepare_df(raw)
        assert len(df) == 49

    def test_year_column_in_reasonable_range(self):
        df = prepare_df(_make_raw_df(jd_start=2440000.0, n=5))
        assert 1960 < df["year"].iloc[0] < 2100

    def test_extra_columns_stripped(self):
        raw = _make_raw_df()
        raw["extra"] = 99
        df = prepare_df(raw)
        assert "extra" not in df.columns

    def test_year_monotonically_increasing(self):
        df = _prepared()
        assert (df["year"].diff().dropna() > 0).all()

    def test_day_monotonically_increasing(self):
        df = _prepared()
        assert (df["day"].diff().dropna() > 0).all()

    def test_non_numeric_sind_dropped(self):
        raw = _make_raw_df(n=20)
        raw = raw.astype(object)
        raw.iloc[2, 1] = "bad"
        df = prepare_df(raw)
        assert len(df) == 19


# ─────────────────────────────────────────────────────────────────────────────
# split_df
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitDF:

    def setup_method(self):
        self.df = _prepared(n=200)

    def test_returns_three_dataframes(self):
        parts = split_df(self.df)
        assert len(parts) == 3
        for p in parts:
            assert isinstance(p, pd.DataFrame)

    def test_lengths_sum_to_total(self):
        train, valid, test = split_df(self.df, train_split=0.7, valid_split=0.2)
        assert len(train) + len(valid) + len(test) == len(self.df)

    def test_partitions_are_non_overlapping_and_contiguous(self):
        """Verify the three parts cover the full df without overlap."""
        train, valid, test = split_df(self.df, train_split=0.7, valid_split=0.2)
        combined = pd.concat([train, valid, test])
        assert len(combined) == len(self.df)
        # Datetime index should span the same range
        assert combined.index.min() == self.df.index.min()
        assert combined.index.max() == self.df.index.max()

    def test_train_fraction_respected(self):
        n = len(self.df)
        train, _, _ = split_df(self.df, train_split=0.8, valid_split=0.19)
        assert len(train) == pytest.approx(0.8 * n, abs=1)

    def test_valid_fraction_respected(self):
        n = len(self.df)
        _, valid, _ = split_df(self.df, train_split=0.8, valid_split=0.19)
        assert len(valid) == pytest.approx(0.19 * n, abs=1)

    def test_train_before_valid_before_test(self):
        """Chronological ordering must be preserved across the three splits."""
        train, valid, test = split_df(self.df)
        if len(valid) > 0:
            assert train.index[-1] < valid.index[0]
        if len(test) > 0 and len(valid) > 0:
            assert valid.index[-1] < test.index[0]

    def test_copies_are_independent(self):
        """Modifying a split must not affect the original df."""
        train, _, _ = split_df(self.df)
        original_first = self.df["sind"].iloc[0]
        train.iloc[0, train.columns.get_loc("sind")] = -9999.0
        assert self.df["sind"].iloc[0] == pytest.approx(original_first)


# ─────────────────────────────────────────────────────────────────────────────
# clean_df
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanDF:

    def _make_dirty_split(self, n=200, n_outliers=5):
        df = _prepared(n=n)
        train, valid, _ = split_df(df, train_split=0.7, valid_split=0.2)
        train = train.copy()
        train.iloc[:n_outliers, train.columns.get_loc("sind")] = 999.0
        return train, valid

    def test_returns_three_values(self):
        train, valid = self._make_dirty_split()
        result = clean_df(train, valid, plot=False, verbose=False)
        assert len(result) == 3

    def test_train_outliers_removed(self):
        train, valid = self._make_dirty_split(n_outliers=5)
        clean_train, _, _ = clean_df(train, valid, tol=4, plot=False, verbose=False)
        assert (clean_train["sind"] == 999.0).sum() == 0

    def test_mad_returned_is_positive(self):
        train, valid = self._make_dirty_split()
        _, _, mad = clean_df(train, valid, plot=False, verbose=False)
        assert mad > 0

    def test_valid_outliers_removed(self):
        """Outliers in the validation set should also be filtered using train MAD."""
        train, valid = self._make_dirty_split()
        valid = valid.copy()
        valid.iloc[0, valid.columns.get_loc("sind")] = 999.0
        _, clean_valid, _ = clean_df(train, valid, tol=4, plot=False, verbose=False)
        assert (clean_valid["sind"] == 999.0).sum() == 0

    def test_clean_data_unchanged(self):
        """With no outliers, clean_df should return the same length as input."""
        df = _prepared(n=200)
        train, valid, _ = split_df(df, train_split=0.7, valid_split=0.2)
        clean_train, clean_valid, _ = clean_df(train, valid, tol=4, plot=False, verbose=False)
        assert len(clean_train) == len(train)
        assert len(clean_valid) == len(valid)

    def test_tighter_tol_removes_more(self):
        """A stricter tolerance should remove at least as many points as a looser one."""
        train, valid = self._make_dirty_split(n_outliers=0)
        ct_loose, _, _ = clean_df(train, valid, tol=10, plot=False, verbose=False)
        ct_tight, _, _ = clean_df(train, valid, tol=1,  plot=False, verbose=False)
        assert len(ct_tight) <= len(ct_loose)


# ─────────────────────────────────────────────────────────────────────────────
# downsample_min_gap
# ─────────────────────────────────────────────────────────────────────────────

class TestDownsampleMinGap:

    def _df_with_gaps(self, gaps, jd_start=2440000.0):
        """Build a prepared df where consecutive gaps (in days) follow `gaps`."""
        days = np.cumsum([0.0] + list(gaps))
        jd   = jd_start + days
        sind = np.ones(len(jd)) * 0.2
        raw  = pd.DataFrame({"jd": jd, "sind": sind})
        return prepare_df(raw, relative=False)

    def test_all_gaps_at_least_minimum(self):
        df = self._df_with_gaps([1.0] * 100)
        result = downsample_min_gap(df, minimum_gap=3.0)
        gaps = np.diff(result["day"].values)
        assert np.all(gaps >= 3.0 - 1e-9)

    def test_first_point_always_kept(self):
        df = self._df_with_gaps([1.0] * 50)
        result = downsample_min_gap(df, minimum_gap=5.0)
        assert result["day"].iloc[0] == pytest.approx(df["day"].iloc[0])

    def test_length_reduced_when_too_dense(self):
        df = self._df_with_gaps([1.0] * 100)
        result = downsample_min_gap(df, minimum_gap=5.0)
        assert len(result) < len(df)

    def test_already_sparse_unchanged(self):
        """If all gaps are already >= minimum_gap, no rows should be dropped."""
        df = self._df_with_gaps([10.0] * 50)
        result = downsample_min_gap(df, minimum_gap=5.0)
        assert len(result) == len(df)

    def test_returns_dataframe(self):
        df = self._df_with_gaps([2.0] * 20)
        result = downsample_min_gap(df, minimum_gap=3.0)
        assert isinstance(result, pd.DataFrame)

    def test_index_reset(self):
        """Output index should be 0, 1, 2, … (reset_index applied)."""
        df = self._df_with_gaps([1.0] * 30)
        result = downsample_min_gap(df, minimum_gap=4.0)
        assert list(result.index) == list(range(len(result)))

    def test_sind_values_preserved(self):
        """sind values in the kept rows must be unchanged."""
        gaps  = [5.0] * 20
        sind_vals = np.linspace(0.1, 0.3, 21)
        jd = 2440000.0 + np.cumsum([0.0] + gaps)
        raw = pd.DataFrame({"jd": jd, "sind": sind_vals})
        df = prepare_df(raw, relative=False)
        result = downsample_min_gap(df, minimum_gap=3.0)
        # All rows should survive (gaps are 5 d > 3 d); values unchanged
        assert len(result) == len(df)
        np.testing.assert_allclose(result["sind"].values, df["sind"].values)

    def test_minimum_gap_zero_keeps_all(self):
        """minimum_gap=0 should keep every row."""
        df = self._df_with_gaps([0.5] * 40)
        result = downsample_min_gap(df, minimum_gap=0.0)
        assert len(result) == len(df)
