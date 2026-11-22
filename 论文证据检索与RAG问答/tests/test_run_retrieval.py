from __future__ import annotations

from scripts import run_retrieval


def test_installed_package_versions_include_optional_retrieval_dependencies(monkeypatch):
    available = {
        "datasets": "5.1.0",
        "numpy": "2.4.6",
        "pandas": "3.1.0",
        "rank-bm25": "0.2.2",
        "torch": "2.14.1",
    }

    def fake_version(name: str) -> str:
        if name not in available:
            raise run_retrieval.importlib.metadata.PackageNotFoundError(name)
        return available[name]

    monkeypatch.setattr(run_retrieval.importlib.metadata, "version", fake_version)

    versions = run_retrieval.installed_package_versions()

    assert versions["torch"] == "2.14.1"
    assert versions["datasets"] == "5.1.0"
    assert versions["chromadb"] is None
    assert versions["sentence-transformers"] is None
    assert set(versions) == set(run_retrieval.RUNTIME_DISTRIBUTIONS)
