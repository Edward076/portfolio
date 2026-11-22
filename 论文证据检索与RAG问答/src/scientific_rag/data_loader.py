from __future__ import annotations

import json
from urllib.request import urlopen

from datasets import DatasetDict, load_dataset


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def _pin_hub_resolve_url(url: str, revision: str) -> str:
    repository, separator, remainder = url.partition("/resolve/")
    _, path_separator, resource_path = remainder.partition("/")
    if not separator or not path_separator or not resource_path:
        raise ValueError(f"Unexpected Hugging Face resolve URL: {url}")
    return f"{repository}/resolve/{revision}/{resource_path}"


def load_qasper(
    dataset_id: str = "allenai/qasper",
    revision: str | None = None,
    parquet_conversion_revision: str | None = None,
) -> tuple[DatasetDict, str, str, str | None]:
    """Load Qasper and return (splits, source revision, loading method).

    The source repository currently contains a legacy dataset script, which
    recent ``datasets`` versions refuse to execute. In that case this loads
    Hugging Face's official Parquet conversion and still records the source
    repository SHA for provenance.
    """
    source_revision = revision or _get_json(
        f"https://huggingface.co/api/datasets/{dataset_id}"
    ).get("sha", "unknown")
    try:
        return (
            load_dataset(dataset_id, revision=source_revision),
            source_revision,
            "dataset_script",
            None,
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        encoded_id = dataset_id.replace("/", "%2F")
        conversion_revision = parquet_conversion_revision
        if conversion_revision is None:
            refs = _get_json(f"https://huggingface.co/api/datasets/{dataset_id}/refs")
            conversion_revision = next(
                item["targetCommit"]
                for item in refs.get("converts", [])
                if item["name"] == "parquet"
            )
        files = _get_json(f"https://datasets-server.huggingface.co/parquet?dataset={encoded_id}")[
            "parquet_files"
        ]
        data_files = {
            split: next(
                _pin_hub_resolve_url(item["url"], conversion_revision)
                for item in files
                if item["split"] == split
            )
            for split in ("train", "validation", "test")
        }
        return (
            load_dataset("parquet", data_files=data_files),
            source_revision,
            "hub_parquet_conversion",
            conversion_revision,
        )
