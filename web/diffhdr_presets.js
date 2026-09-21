// Keeps the preset dropdown of the all-in-one DiffHDR nodes in sync with the
// sampler / scheduler / shift widgets. Purely cosmetic: the Python side resolves
// the preset again on execution, so the node behaves identically without this file.
import { app } from "../../scripts/app.js";

// Mirrors diffhdr/sampling.py PRESETS.
const PRESETS = {
    fast: { sampler: "res_multistep", scheduler: "simple", shift: 8.0 },
    original: { sampler: "euler", scheduler: "simple", shift: 5.0 },
};
const TARGETS = ["sampler", "scheduler", "shift"];
const NODES = new Set(["DiffHDRVideo", "DiffHDRPano"]);

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function chain(widget, after) {
    if (!widget) return;
    const original = widget.callback;
    widget.callback = function (...args) {
        const result = original?.apply(this, args);
        after();
        return result;
    };
}

/** Copies the selected preset's values into the three widgets. */
function applyPreset(node) {
    const values = PRESETS[findWidget(node, "preset")?.value];
    if (!values) return; // "custom": the user's widget values stand
    node._diffhdrApplyingPreset = true;
    try {
        for (const name of TARGETS) {
            const widget = findWidget(node, name);
            if (widget) widget.value = values[name];
        }
    } finally {
        node._diffhdrApplyingPreset = false;
    }
    node.setDirtyCanvas(true, true);
}

/** Switches the preset to "custom" after the user edited one of the three widgets. */
function markCustom(node) {
    if (node._diffhdrApplyingPreset) return;
    const preset = findWidget(node, "preset");
    if (!preset || preset.value === "custom") return;
    preset.value = "custom";
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "DiffHDR.presets",
    nodeCreated(node) {
        if (!NODES.has(node.comfyClass) || !findWidget(node, "preset")) return;
        chain(findWidget(node, "preset"), () => applyPreset(node));
        for (const name of TARGETS) chain(findWidget(node, name), () => markCustom(node));
    },
});
