from __future__ import annotations

import os

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    if 'requires_vitis' in item.keywords and 'XILINX_VITIS' not in os.environ:
        pytest.skip('needs AMD Vitis (XILINX_VITIS is not set)')


@pytest.fixture(autouse=True)
def _transport_checked(monkeypatch):
    """Every design a test lowers moves the elements its ports declare (`transport_check`)."""
    from aie4ml.model import AIEModel
    from transport_check import check_transport

    run_pipeline = AIEModel.run_pipeline

    def checked(self):
        run_pipeline(self)
        check_transport(self)
        return self

    monkeypatch.setattr(AIEModel, 'run_pipeline', checked)
