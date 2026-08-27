"""Smoke tests for functionality that depends on optional extras.

These are skipped automatically when the optional dependency is not installed,
so the core test run does not require tensorflow / torch.
"""

import pytest


def test_keras_cnn_builders_produce_equivalent_architectures():
    pytest.importorskip("tensorflow")
    from scikit_raman.module.models import (
        create_model_benchmark,
        create_model_benchmark_weight_initialization,
    )

    a = create_model_benchmark(n_dims=200, number_classes=3)
    b = create_model_benchmark_weight_initialization(n_dims=200, number_classes=3)

    assert a.input_shape == b.input_shape == (None, 200)
    assert a.output_shape == b.output_shape == (None, 3)
    assert len(a.layers) == len(b.layers)


def test_torch_importable_if_installed():
    torch = pytest.importorskip("torch")
    assert hasattr(torch, "tensor")
