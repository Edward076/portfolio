from scientific_rag import data_loader


def test_loader_uses_pinned_parquet_conversion_commit(monkeypatch):
    calls = []
    dataset = object()
    conversion_revision = "conversion-commit-123"

    def fake_load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        if name == "allenai/qasper":
            raise RuntimeError("Dataset scripts are no longer supported")
        return dataset

    queried_urls = []

    def fake_get_json(url):
        queried_urls.append(url)
        assert "parquet?dataset=allenai%2Fqasper" in url
        return {
            "parquet_files": [
                {
                    "split": split,
                    "url": (
                        "https://huggingface.co/datasets/allenai/qasper/resolve/"
                        f"{('refs%2Fconvert%2Fparquet' if split == 'train' else 'latest-commit')}/"
                        f"data/{split}-00000-of-00001.parquet"
                    ),
                }
                for split in ("train", "validation", "test")
            ]
        }

    monkeypatch.setattr(data_loader, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(data_loader, "_get_json", fake_get_json)

    result = data_loader.load_qasper(
        revision="source-commit-456",
        parquet_conversion_revision=conversion_revision,
    )

    assert result == (dataset, "source-commit-456", "hub_parquet_conversion", conversion_revision)
    assert calls[0] == ("allenai/qasper", {"revision": "source-commit-456"})
    assert calls[1][0] == "parquet"
    for split in ("train", "validation", "test"):
        assert calls[1][1]["data_files"][split].endswith(
            f"/resolve/{conversion_revision}/data/{split}-00000-of-00001.parquet"
        )
    assert len(queried_urls) == 1
