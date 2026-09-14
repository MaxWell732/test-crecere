import pytest

from extraction import paths


@pytest.fixture(autouse=True)
def _isolated_logs(tmp_path, monkeypatch):
    """Tests never write into the real data/logs."""
    monkeypatch.setattr(paths, "LOGS", tmp_path / "logs")
