"""Requirements-to-tests traceability.

Three pure functions, so that each one can be unit-tested on its own rather than
only through the side effect of writing a file:

    parse_requirement_ids  requirements.md  -> the declared requirement IDs
    scan_req_markers       the test tree    -> {requirement ID: [test files]}
    render_matrix          both of those    -> the markdown matrix

WHY A STATIC SCAN RATHER THAN PYTEST'S COLLECTED MARKERS
--------------------------------------------------------
The obvious alternative is a `pytest_collection_modifyitems` hook reading markers
off collected items. It was rejected for one practical reason: the collected set
depends on how the suite was invoked. Running `pytest tests/spectral` would then
regenerate the matrix from a fraction of the tests and quietly report every other
requirement as untested, which is precisely the failure mode a traceability
matrix exists to prevent.

Scanning the source is invocation-independent and reproducible without executing
anything. Its one limitation is that markers applied dynamically at runtime are
invisible to it; this suite applies every marker as a literal decorator, and
`--strict-markers` plus the unknown-ID check in `test_traceability.py` catch the
typos that would otherwise slip through.

WHY THE AST RATHER THAN A REGULAR EXPRESSION
--------------------------------------------
A regex over the file text is shorter, and it is wrong. This very test suite
contains fixtures whose *string literals* hold marker-shaped text — files written
to a temporary directory to exercise the scanner itself. A text scan counts those
as real coverage, so a requirement could appear verified by a test that only ever
mentioned it inside a quoted string. Parsing the AST and matching call nodes means
a marker has to actually be code to count. `test_scan_req_markers_ignores_string_literals`
pins this behaviour.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# A requirement is declared by a table row in requirements.md whose first cell is
# the ID. Anchoring to the row start avoids picking up the IDs that appear inside
# other requirements' prose (REQ-021 refers to REQ-024, for example).
_DECLARATION = re.compile(r"^\|\s*(REQ-\d{3})\s*\|", re.MULTILINE)


def parse_requirement_ids(requirements_markdown: str) -> list[str]:
    """Return the requirement IDs declared in requirements.md, in document order."""
    return _DECLARATION.findall(requirements_markdown)


def _is_req_marker(func: ast.expr) -> bool:
    """True for the attribute chain `pytest.mark.req`."""
    if not isinstance(func, ast.Attribute) or func.attr != "req":
        return False
    mark = func.value
    if not isinstance(mark, ast.Attribute) or mark.attr != "mark":
        return False
    root = mark.value
    return isinstance(root, ast.Name) and root.id == "pytest"


def _req_ids_in_source(source: str) -> list[str]:
    """Every requirement ID passed to a `pytest.mark.req(...)` call in `source`.

    Walks the whole tree rather than only decorator lists, so that markers attached
    through `pytest.param(..., marks=pytest.mark.req("REQ-0XX"))` are counted too.
    """
    ids: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not _is_req_marker(node.func):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str):
                ids.append(value)
    return ids


def scan_req_markers(test_root: Path) -> dict[str, list[str]]:
    """Map each requirement ID to the test files that claim to verify it.

    Paths are returned relative to `test_root` and sorted, so the generated matrix
    is stable across machines and does not leak absolute local paths.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(test_root.rglob("test_*.py")):
        relative = path.relative_to(test_root).as_posix()
        for req_id in _req_ids_in_source(path.read_text(encoding="utf-8")):
            found.setdefault(req_id, set()).add(relative)
    return {req_id: sorted(files) for req_id, files in sorted(found.items())}


def render_matrix(requirement_ids: list[str], markers: dict[str, list[str]]) -> str:
    """Render the traceability matrix as markdown.

    Requirements with no test are rendered explicitly rather than omitted; a matrix
    that silently drops them would be worse than no matrix at all.
    """
    covered = sum(1 for req_id in requirement_ids if markers.get(req_id))
    lines = [
        "# Traceability matrix",
        "",
        "Generated from `@pytest.mark.req(...)` markers in the test tree. Do not edit by hand.",
        "",
        f"**{covered} of {len(requirement_ids)} requirements have at least one test.**",
        "",
        "| Requirement | Verified by | Tests |",
        "|---|---|---|",
    ]
    for req_id in requirement_ids:
        files = markers.get(req_id, [])
        if files:
            lines.append(f"| {req_id} | {', '.join(f'`{f}`' for f in files)} | {len(files)} |")
        else:
            lines.append(f"| {req_id} | **NONE** | 0 |")

    unknown = sorted(set(markers) - set(requirement_ids))
    if unknown:
        lines += [
            "",
            "## Markers referencing undeclared requirements",
            "",
            "These IDs are marked in tests but are not declared in `requirements.md`:",
            "",
        ]
        lines += [f"- {req_id}" for req_id in unknown]

    return "\n".join(lines) + "\n"
