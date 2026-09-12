"""Finding the LLM in somebody else's codebase.

    from livingeval.discover import scan
    for site in scan(Path(".")):
        print(site.ident, site.archetype, site.confidence)

Static only -- see `sites` for why.
"""

from pathlib import Path

from livingeval.discover.archetype import Archetype, classify, classify_all
from livingeval.discover.sites import CallSite, Evidence, scan_file, scan_tree

__all__ = ["Archetype", "CallSite", "Evidence", "classify", "classify_all", "scan",
           "scan_file", "scan_tree"]


def scan(root: Path, include_tests: bool = False) -> list[CallSite]:
    """Every model call under `root`, classified."""
    return classify_all(scan_tree(root, include_tests=include_tests))
