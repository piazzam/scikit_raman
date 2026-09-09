"""Tests for scikit_raman.module.preprocessing.

Covers the fundamentals: resampling, SNV normalisation, spike removal and
polynomial baseline removal, all on small synthetic spectra.
"""

import numpy as np
import pandas as pd
import pytest

import scikit_raman.module.preprocessing as pp
from conftest import make_spectra


class TestResampling:
    def test_resample_one_shift_point_count_and_grid(self):
        x = np.linspace(300, 1800, 200)
        y = np.sin(x / 100.0)
        y_new, x_new = pp.resample_one_shift(y, x, start=400, end=1600, points=991)
        assert len(y_new) == 991
        assert len(x_new) == 991
        assert x_new[0] == 400 and x_new[-1] == 1600
        assert np.all(np.isfinite(y_new))

    def test_resample_one_shift_matches_source_on_overlap(self):
        x = np.linspace(400, 1600, 400)
        y = 2.0 * x + 5.0  # linear -> linear interpolation is exact
        y_new, x_new = pp.resample_one_shift(y, x, points=101)
        assert np.allclose(y_new, 2.0 * np.array(x_new) + 5.0)

    def test_resample_shift_rewrites_dataframe(self):
        df = pd.DataFrame({
            "spectra": [list(s) for s in make_spectra(4, 60, seed=3)],
            "x-axis": [list(np.linspace(390, 1620, 60))] * 4,
        })
        out = pp.resample_shift(df.copy(), points=77)
        assert all(len(s) == 77 for s in out["spectra"])
        assert all(len(x) == 77 for x in out["x-axis"])
        assert out["x-axis"].iloc[0][0] == 400


class TestSNV:
    def test_snv_gives_zero_mean_unit_std(self, spectra_df):
        out = pp.snv_normalization(spectra_df.copy())
        for spectrum in out["spectra"]:
            arr = np.asarray(spectrum)
            assert arr.mean() == pytest.approx(0.0, abs=1e-9)
            assert arr.std() == pytest.approx(1.0, rel=1e-6)

    def test_snv_is_affine_invariant(self):
        base = make_spectra(1, 50, seed=7)[0]
        df = pd.DataFrame({"spectra": [list(base), list(3.0 * base + 10.0)]})
        out = pp.snv_normalization(df.copy())
        assert np.allclose(out["spectra"].iloc[0], out["spectra"].iloc[1], atol=1e-9)


class TestSpikeRemoval:
    def test_spike_is_suppressed(self):
        clean = make_spectra(1, 80, seed=5)[0]
        spiked = clean.copy()
        spiked[40] += 60.0
        df = pd.DataFrame({"spectra": [list(spiked)]})
        out = np.asarray(pp.spike_removal(df.copy())["spectra"].iloc[0])
        # the spike is pulled back close to the surrounding baseline
        assert abs(out[40] - clean[40]) < abs(spiked[40] - clean[40])
        assert abs(out[40] - clean[40]) < 5.0

    def test_spike_free_spectrum_barely_changes(self):
        # a smooth, noise-free spectrum has no sharp jumps to mistake for spikes
        axis = np.arange(200, dtype=float)
        clean = 5.0 + 3.0 * np.exp(-((axis - 100) ** 2) / (2 * 40.0**2))
        df = pd.DataFrame({"spectra": [list(clean)]})
        out = np.asarray(pp.spike_removal(df.copy())["spectra"].iloc[0])
        assert np.allclose(out, clean, atol=1e-6)

    def test_modified_z_score_flags_outlier(self):
        data = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 8.0, 0.95, 1.0])
        scores = np.abs(pp.modified_z_score(data))
        assert scores.argmax() == 5


class TestBaselineRemoval:
    def _signal(self, n_points=80):
        axis = np.arange(n_points, dtype=float)
        baseline = 5.0 + 0.03 * axis + 1e-3 * axis**2
        peak = 12.0 * np.exp(-((axis - 40) ** 2) / (2 * 4.0**2))
        return axis, baseline, peak

    def test_polynomial_baseline_is_removed_and_peak_kept(self):
        axis, baseline, peak = self._signal()
        df = pd.DataFrame({"spectra": [list(baseline + peak)]})
        out = np.asarray(
            pp.remove_baseline_polynomial(df.copy(), deg=5)["spectra"].iloc[0])

        off_peak = np.abs(axis - 40) > 15
        # the baseline offset (>= 5 everywhere) is largely removed off-peak
        assert np.abs(out[off_peak]).mean() < 3.0
        assert np.abs(out[off_peak]).mean() < 0.5 * baseline[off_peak].mean()
        # the peak survives
        assert out.max() > 9.0
        assert 30 < out.argmax() < 50

    @pytest.mark.xfail(reason="all-zero spectra are silently dropped from X_out, "
                              "which then length-mismatches the DataFrame index; "
                              "slated for the Stage 3 cleanup",
                       strict=True, raises=ValueError)
    def test_polynomial_baseline_handles_all_zero_spectra(self):
        _, baseline, peak = self._signal()
        df = pd.DataFrame({"spectra": [list(baseline + peak), [0.0] * 80]})
        pp.remove_baseline_polynomial(df.copy(), deg=5)
