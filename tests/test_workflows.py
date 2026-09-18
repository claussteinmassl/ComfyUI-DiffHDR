import json
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).parent.parent / "workflows").glob("*.json"))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_links_consistent(path):
    wf = json.loads(path.read_text())
    ids = {n["id"] for n in wf["nodes"]}
    for link in wf["links"]:
        _, src, _, dst, _, _ = link[:6]
        assert src in ids and dst in ids
    assert any(n["type"].startswith("DiffHDR") for n in wf["nodes"])


def test_four_workflows():
    assert len(WORKFLOWS) == 4
