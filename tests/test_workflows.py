import asyncio
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


def _diffhdr_node_classes():
    """Maps each DiffHDR node id (schema.node_id) to its node class."""
    from diffhdr.nodes import comfy_entrypoint
    ext = asyncio.run(comfy_entrypoint())
    nodes = asyncio.run(ext.get_node_list())
    return {n.define_schema().node_id: n for n in nodes}


def _widget_inputs(schema_inputs, io):
    """Returns the schema's widget inputs (socket-only inputs excluded), in serialisation order."""
    return [inp for inp in schema_inputs if isinstance(inp, io.WidgetInput)]


def check_node_widgets_values(node: dict, node_cls, io) -> None:
    """Asserts a DiffHDR node's ``widgets_values`` matches its schema.

    The frontend serialises one value per widget input (``Int``/``Float``/``String``/``Boolean``/
    ``Combo``; socket-only inputs such as ``Model``/``Vae``/``Clip``/``Image``/``Mask``/
    ``Conditioning``/``Latent`` never contribute a widget value), plus one extra synthetic value
    right after any widget with ``control_after_generate`` set (the frontend's
    "randomize/fixed/..." control, always placed immediately after its owning widget).

    Args:
        node: A single node dict from a workflow's ``nodes`` list.
        node_cls: The ``io.ComfyNode`` subclass for this node's type.
        io: The ``comfy_api.latest.io`` module (for ``io.WidgetInput`` / ``io.Combo``).

    Raises:
        AssertionError: If the number of ``widgets_values`` entries does not match the schema, or
            a Combo widget's stored value is not one of the schema's declared options.
    """
    schema = node_cls.define_schema()
    widgets = _widget_inputs(schema.inputs, io)
    expected = len(widgets) + sum(1 for inp in widgets if getattr(inp, "control_after_generate", None))
    actual = node.get("widgets_values", [])
    assert len(actual) == expected, (
        f"node {node.get('id')} ({node['type']}): {len(actual)} widgets_values, "
        f"expected {expected} (schema widgets: {[w.id for w in widgets]})"
    )

    idx = 0
    for inp in widgets:
        value = actual[idx]
        if isinstance(inp, io.Combo.Input) and inp.options:
            assert value in inp.options, (
                f"node {node.get('id')} ({node['type']}): widget '{inp.id}' value {value!r} "
                f"is not one of the schema's options {inp.options}"
            )
        idx += 1
        if getattr(inp, "control_after_generate", None):
            idx += 1  # skip the synthetic control_after_generate value


@pytest.mark.requires_comfy
@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_diffhdr_widgets_values_match_schema(path):
    from comfy_api.latest import io
    classes = _diffhdr_node_classes()
    wf = json.loads(path.read_text())
    for node in wf["nodes"]:
        if not node["type"].startswith("DiffHDR"):
            continue
        check_node_widgets_values(node, classes[node["type"]], io)
