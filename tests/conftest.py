"""Shared fixtures and helpers for the scikit_raman test suite.

All tests use small synthetic spectra, never real patient data.
"""

import numpy as np
import pandas as pd
import pytest

from scikit_raman.Classes.Dataset import Dataset

DEFAULT_LABELS = {"cov": 0, "covNeg": 1, "ctrl": 2}


def make_spectra(n_spectra, n_points, seed=0):
    """Return an ``(n_spectra, n_points)`` array of smooth-ish synthetic spectra.

    Each spectrum is a broad Gaussian peak on a sloping baseline plus a little
    noise, which is enough structure to exercise the preprocessing routines.
    """
    rng = np.random.default_rng(seed)
    axis = np.linspace(400, 1600, n_points)
    out = np.empty((n_spectra, n_points))
    for i in range(n_spectra):
        centre = rng.uniform(700, 1300)
        width = rng.uniform(40, 120)
        peak = rng.uniform(5, 15) * np.exp(-((axis - centre) ** 2) / (2 * width**2))
        baseline = rng.uniform(0.5, 2.0) + rng.uniform(1e-4, 5e-4) * axis
        noise = rng.normal(scale=0.05, size=n_points)
        out[i] = peak + baseline + noise
    return out


def make_dataset(n_patients=4, spectra_per_patient=3, n_points=40, seed=0,
                 categories=("cov", "covNeg", "ctrl")):
    """Build a small :class:`~scikit_raman.Classes.Dataset.Dataset`.

    Patients are assigned round-robin to ``categories`` so that group-aware
    splitting has something meaningful to preserve.
    """
    n = n_patients * spectra_per_patient
    axis = np.linspace(400, 1600, n_points)
    spectra = make_spectra(n, n_points, seed=seed)
    x_axis = np.tile(axis, (n, 1))
    raw = np.array([i % 2 == 0 for i in range(n)])
    user = np.array([f"patient_{p:02d}" for p in range(n_patients)
                     for _ in range(spectra_per_patient)])
    name = np.array(["raw" if r else "dark" for r in raw])
    category = np.array([categories[p % len(categories)] for p in range(n_patients)
                         for _ in range(spectra_per_patient)])
    labels = np.empty(shape=(0, 0))  # triggers create_label from category
    return Dataset(spectra, x_axis, raw, user, name, category, labels)


def make_dataframe(n_spectra=6, n_points=40, seed=1):
    """Build the plain ``DataFrame`` layout the preprocessing functions expect."""
    axis = np.linspace(400, 1600, n_points)
    spectra = make_spectra(n_spectra, n_points, seed=seed)
    return pd.DataFrame({
        "spectra": [list(s) for s in spectra],
        "x-axis": [list(axis) for _ in range(n_spectra)],
        "category": [("cov", "ctrl")[i % 2] for i in range(n_spectra)],
    })


@pytest.fixture
def dataset():
    return make_dataset()


@pytest.fixture
def spectra_df():
    return make_dataframe()
