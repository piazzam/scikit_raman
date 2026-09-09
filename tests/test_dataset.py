"""Tests for scikit_raman.Classes.Dataset.

Focus areas (the parts most likely to silently break): loading/constructing,
filtering helpers, and the cross-validation split generators.
"""

import numpy as np
import pytest

from scikit_raman.Classes.Dataset import Dataset
from conftest import make_dataset


class TestConstruction:
    def test_basic_attributes(self, dataset):
        assert len(dataset) == 12
        assert dataset.n_dims == 40
        assert dataset.spectra.shape == (12, 40)
        assert dataset.x_axis.shape == (12, 40)

    def test_labels_derived_from_category(self):
        ds = make_dataset(n_patients=3, spectra_per_patient=2,
                          categories=("cov", "covNeg", "ctrl"))
        # patients 0, 1, 2 -> cov, covNeg, ctrl -> 0, 1, 2
        assert list(ds.labels) == [0, 0, 1, 1, 2, 2]

    def test_explicit_labels_are_kept(self):
        ds = make_dataset()
        labels = np.arange(len(ds))
        ds2 = Dataset(ds.spectra, ds.x_axis, ds._raw, ds.user, ds._name,
                      ds.category, labels)
        assert list(ds2.labels) == list(labels)

    def test_getitem_and_len_agree(self, dataset):
        spectrum, x, raw, user = dataset[0][:4]
        assert spectrum.shape == (40,)
        assert x.shape == (40,)
        assert isinstance(user, (str, np.str_))

    def test_unique_accessors(self, dataset):
        assert sorted(dataset.get_unique_user()) == [
            "patient_00", "patient_01", "patient_02", "patient_03"]
        assert sorted(dataset.get_unique_category()) == ["cov", "covNeg", "ctrl"]
        assert sorted(dataset.get_unique_labels()) == [0, 1, 2]


class TestFiltering:
    def test_get_raw_data_returns_only_raw(self, dataset):
        raw_ds = dataset.get_raw_data()
        assert len(raw_ds) == 6
        assert bool(np.all(raw_ds._raw))

    def test_get_by_indices(self, dataset):
        sub = dataset.get_by_indices([0, 1, 5])
        assert len(sub) == 3
        assert np.allclose(sub.spectra[2], dataset.spectra[5])

    def test_search_by_name(self, dataset):
        sub = dataset.search_by_name("patient_01")
        assert len(sub) == 3
        assert set(sub.user) == {"patient_01"}

    def test_search_by_name_indices(self, dataset):
        assert dataset.search_by_name_indices("patient_02") == [6, 7, 8]

    def test_search_by_category_name(self, dataset):
        sub = dataset.search_by_category_name("cov")
        assert set(sub.category) == {"cov"}
        # patients 0 and 3 are "cov", 3 spectra each
        assert len(sub) == 6

    def test_search_by_category_label(self, dataset):
        sub = dataset.search_by_category_label(2)
        assert set(sub.labels) == {2}
        assert set(sub.category) == {"ctrl"}

    def test_remove_elements(self, dataset):
        n = len(dataset)
        dataset.remove_elements([0, 1])
        assert len(dataset) == n - 2
        assert dataset.spectra.shape[0] == n - 2

    @pytest.mark.xfail(reason="get_dark_data references self.raw instead of "
                              "self._raw; slated for the Stage 3 cleanup",
                       strict=True, raises=AttributeError)
    def test_get_dark_data(self, dataset):
        dark = dataset.get_dark_data()
        assert not np.any(dark._raw)


class TestSplits:
    def _assert_no_patient_leak(self, dataset, folds):
        for train_idx, test_idx in folds:
            train_patients = set(dataset.user[train_idx])
            test_patients = set(dataset.user[test_idx])
            assert not (train_patients & test_patients)

    def test_k_fold_preserves_patients(self, dataset):
        folds = dataset.k_fold(2)
        assert len(folds) == 2
        self._assert_no_patient_leak(dataset, folds)

    def test_k_fold_covers_every_sample_once_as_test(self, dataset):
        folds = dataset.k_fold(2)
        seen = np.concatenate([test for _, test in folds])
        assert sorted(seen) == list(range(len(dataset)))

    def test_stratified_k_fold_preserves_patients(self, dataset):
        folds = dataset.stratified_k_fold(2)
        self._assert_no_patient_leak(dataset, folds)

    def test_simple_k_fold_splits_all_indices(self, dataset):
        folds = dataset.simple_k_fold(3)
        assert len(folds) == 3
        for train_idx, test_idx in folds:
            assert len(train_idx) + len(test_idx) == len(dataset)

    def test_leave_one_patient_cv(self, dataset):
        folds = dataset.leave_one_patient_cv()
        assert len(folds) == len(dataset.get_unique_user())
        for _, test_idx in folds:
            assert len(set(dataset.user[test_idx])) == 1

    def test_split_without_labels_raises(self):
        ds = make_dataset()
        del ds.labels
        with pytest.raises(Exception, match="labels"):
            ds.k_fold(2)
