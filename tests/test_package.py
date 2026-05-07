from __future__ import annotations

import importlib.metadata

import luminareionization as m


def test_version() -> None:
    assert importlib.metadata.version("luminareionization") == m.__version__
