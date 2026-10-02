"""Shared pytest configuration.

Regenerates `docs/traceability-matrix.md` at the end of every session. Because the
matrix is built by scanning the test sources rather than by reading the collected
items (see `traceability.py` for why), the output is the same whether you ran the
whole suite or a single file — so this is safe to do unconditionally.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from traceability import parse_requirement_ids, render_matrix, scan_req_markers

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = REPO_ROOT / "tests"
REQUIREMENTS = REPO_ROOT / "requirements.md"
MATRIX = REPO_ROOT / "docs" / "traceability-matrix.md"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write the traceability matrix.

    A failure to write is warned about, not raised: the matrix is a reporting
    artifact, and a read-only checkout should not turn a green suite red. The
    assertion that every requirement is actually covered lives in
    tests/test_traceability.py, which does not depend on this file being written.
    """
    requirement_ids = parse_requirement_ids(REQUIREMENTS.read_text(encoding="utf-8"))
    markers = scan_req_markers(TEST_ROOT)
    try:
        MATRIX.parent.mkdir(parents=True, exist_ok=True)
        MATRIX.write_text(render_matrix(requirement_ids, markers), encoding="utf-8")
    except OSError as exc:
        warnings.warn(f"could not write {MATRIX.name}: {exc}", stacklevel=1)
