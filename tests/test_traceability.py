"""Tests for the requirements traceability machinery.

The coverage test here is the one that keeps `requirements.md` honest: a requirement
that nobody wrote a test for fails the suite by name, rather than quietly sitting in
a document that claims otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from traceability import parse_requirement_ids, render_matrix, scan_req_markers

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = REPO_ROOT / "tests"
REQUIREMENTS = REPO_ROOT / "requirements.md"


@pytest.fixture(scope="module")
def requirement_ids() -> list[str]:
    return parse_requirement_ids(REQUIREMENTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def markers() -> dict[str, list[str]]:
    return scan_req_markers(TEST_ROOT)


# --------------------------------------------------------------------------- REQ-030


@pytest.mark.req("REQ-030")
def test_every_requirement_has_at_least_one_test(
    requirement_ids: list[str], markers: dict[str, list[str]]
) -> None:
    untested = [req_id for req_id in requirement_ids if not markers.get(req_id)]
    assert not untested, (
        f"{len(untested)} of {len(requirement_ids)} requirements have no test: "
        f"{', '.join(untested)}"
    )


@pytest.mark.req("REQ-030")
def test_no_test_claims_an_undeclared_requirement(
    requirement_ids: list[str], markers: dict[str, list[str]]
) -> None:
    """Catches a typo'd marker, which would otherwise look like coverage but verify nothing."""
    undeclared = sorted(set(markers) - set(requirement_ids))
    assert not undeclared, (
        f"tests reference requirement IDs that requirements.md does not declare: "
        f"{', '.join(undeclared)}"
    )


@pytest.mark.req("REQ-030")
def test_requirements_file_declares_unique_ids(requirement_ids: list[str]) -> None:
    duplicates = sorted({r for r in requirement_ids if requirement_ids.count(r) > 1})
    assert not duplicates, f"duplicate requirement IDs in requirements.md: {duplicates}"


# --------------------------------------------------------------------------- REQ-031


@pytest.mark.req("REQ-031")
def test_parse_requirement_ids_reads_table_rows_only() -> None:
    """IDs mentioned in prose must not be mistaken for declarations.

    REQ-021's acceptance criterion refers to REQ-024 and REQ-025 in running text;
    only the row that *declares* an ID should count.
    """
    document = (
        "| ID | Requirement | Acceptance criterion |\n"
        "|---|---|---|\n"
        "| REQ-001 | Validate inputs. | Raises ValueError. |\n"
        "| REQ-002 | Be deterministic. | Same as REQ-001 but for repeat calls. |\n"
        "\nSome prose mentioning REQ-999 which is not declared anywhere.\n"
    )
    assert parse_requirement_ids(document) == ["REQ-001", "REQ-002"]


@pytest.mark.req("REQ-031")
def test_scan_req_markers_finds_decorators(tmp_path: Path) -> None:
    (tmp_path / "test_example.py").write_text(
        "import pytest\n"
        "\n"
        "@pytest.mark.req('REQ-010')\n"
        "def test_one() -> None:\n"
        "    pass\n"
        "\n"
        '@pytest.mark.req("REQ-010")\n'
        "def test_two() -> None:\n"
        "    pass\n"
        "\n"
        "@pytest.mark.req('REQ-011')\n"
        "def test_three() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )
    # A non-test module must be ignored, so that helper code mentioning a marker
    # in a docstring cannot inflate the matrix.
    (tmp_path / "helper.py").write_text(
        "import pytest\n@pytest.mark.req('REQ-999')\ndef f(): ...\n", encoding="utf-8"
    )

    assert scan_req_markers(tmp_path) == {
        "REQ-010": ["test_example.py"],
        "REQ-011": ["test_example.py"],
    }


@pytest.mark.req("REQ-031")
def test_scan_req_markers_ignores_string_literals(tmp_path: Path) -> None:
    """A marker mentioned inside a string must not count as coverage.

    Regression test. The first version of the scanner used a regular expression over
    the file text, which counted the marker-shaped strings inside this suite's own
    fixtures as real coverage — so a requirement could look verified by a test that
    only ever quoted its ID. Matching AST call nodes instead makes a marker have to
    be code to count.
    """
    (tmp_path / "test_fixture_holder.py").write_text(
        "import pytest\n"
        "\n"
        "SAMPLE_FILE = '''\n"
        "@pytest.mark.req('REQ-404')\n"
        "def test_generated() -> None:\n"
        "    pass\n"
        "'''\n"
        "\n"
        "@pytest.mark.req('REQ-010')\n"
        "def test_real() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )
    assert scan_req_markers(tmp_path) == {"REQ-010": ["test_fixture_holder.py"]}


@pytest.mark.req("REQ-031")
def test_render_matrix_marks_untested_requirements() -> None:
    matrix = render_matrix(
        ["REQ-001", "REQ-002"],
        {"REQ-001": ["spectral/test_psd.py"]},
    )
    assert "**1 of 2 requirements have at least one test.**" in matrix
    assert "| REQ-001 | `spectral/test_psd.py` | 1 |" in matrix
    assert "| REQ-002 | **NONE** | 0 |" in matrix


@pytest.mark.req("REQ-031")
def test_render_matrix_reports_undeclared_ids() -> None:
    matrix = render_matrix(["REQ-001"], {"REQ-001": ["a.py"], "REQ-404": ["b.py"]})
    assert "Markers referencing undeclared requirements" in matrix
    assert "- REQ-404" in matrix
