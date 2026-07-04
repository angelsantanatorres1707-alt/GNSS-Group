from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient


NOTEBOOK_PATH = Path(__file__).parent / "GNSS_Analysis_ipyleaflet.ipynb"
NGF_CACHE_PATH = Path(__file__).parent / "data" / "NGF" / "P049.cwu.igs14.csv"


def test_ipyleaflet_notebook_runs():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )

    client.execute()


def test_ngf_cached_reader_runs():
    if not NGF_CACHE_PATH.exists():
        pytest.skip("P049 NGF cache file is not available")

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
_times, _tseries = Read_NGF("P049", False)
assert len(_times) > 0
assert _tseries.shape[0] == 7
"""
        )
    )

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )

    client.execute()
