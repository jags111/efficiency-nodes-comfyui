// Detect and update Efficiency Nodes from v1.92 to v2.00 changes (Final update?)
import { app } from '../../scripts/app.js'
import { addNode } from "./node_options/common/utils.js";

const ext = {
    name: "efficiency.WorkflowFix",
};

function reloadHiResFixNode(originalNode) {

    // Safeguard against missing 'pos' property
    const position = originalNode.pos && originalNode.pos.length === 2 ? { x: originalNode.pos[0], y: originalNode.pos[1] } : { x: 0, y: 0 };

    // Recreate the node
    const newNode = addNode("HighRes-Fix Script", originalNode, position);

    // Transfer input connections from old node to new node
    originalNode.inputs.forEach((input, index) => {
        if (input && input.link !== null) {
            const originLinkInfo = originalNode.graph.links[input.link];
            if (originLinkInfo) {
                const originNode = originalNode.graph.getNodeById(originLinkInfo.origin_id);
                if (originNode) {
                    originNode.connect(originLinkInfo.origin_slot, newNode, index);
                }
            }
        }
    });

    // Transfer output connections from old node to new node
    originalNode.outputs.forEach((output, index) => {
        if (output && output.links) {
            output.links.forEach(link => {
                const targetLinkInfo = originalNode.graph.links[link];
                if (targetLinkInfo) {
                    const targetNode = originalNode.graph.getNodeById(targetLinkInfo.target_id);
                    if (targetNode) {
                        newNode.connect(index, targetNode, targetLinkInfo.target_slot);
                    }
                }
            });
        }
    });

    // Remove the original node after all connections are transferred
    originalNode.graph.remove(originalNode);

    return newNode;
}

ext.loadedGraphNode = function(node, app) {
    const originalNode = node; // This line ensures that originalNode refers to the provided node
    const kSamplerTypes = [
        "KSampler (Efficient)",
        "KSampler Adv. (Efficient)",
        "KSampler SDXL (Eff.)"
    ];

    // EFFICIENT LOADER & EFF. LOADER SDXL
    /*  Changes:
            Added "token_normalization" & "weight_interpretation" widget below prompt text boxes,
            below code fixes the widget values for empty_latent_width, empty_latent_height, and batch_size
            by shifting down by 2 widget values starting from the "token_normalization" widget.
            Logic triggers when "token_normalization" is a number instead of a string.
    */
    if (node.comfyClass === "Efficient Loader" || node.comfyClass === "Eff. Loader SDXL") {
        const tokenWidget = node.widgets.find(w => w.name === "token_normalization");
        const weightWidget = node.widgets.find(w => w.name === "weight_interpretation");
        
        if (typeof tokenWidget.value === 'number') {
            console.log("[EfficiencyUpdate]", `Fixing '${node.comfyClass}' token and weight widgets:`, node);
            const index = node.widgets.indexOf(tokenWidget);
            if (index !== -1) {
                for (let i = node.widgets.length - 1; i > index + 1; i--) { 
                    node.widgets[i].value = node.widgets[i - 2].value;
                }
            }
            tokenWidget.value = "none";
            weightWidget.value = "comfy";
        }
    }
    
    // KSAMPLER (EFFICIENT), KSAMPLER ADV. (EFFICIENT), & KSAMPLER SDXL (EFF.)
    /*  Changes:
            Removed the "sampler_state" widget which cause all widget values to shift down by a factor of 1.
            Fix involves moving all widget values by -1. "vae_decode" value is lost in this process, so in
            below fix I manually set it to its default value of "true".
    */
    else if (kSamplerTypes.includes(node.comfyClass)) {

        const seedWidgetName = (node.comfyClass === "KSampler (Efficient)") ? "seed" : "noise_seed";
        const stepsWidgetName = (node.comfyClass === "KSampler (Efficient)") ? "steps" : "start_at_step";

        const seedWidget = node.widgets.find(w => w.name === seedWidgetName);
        const stepsWidget = node.widgets.find(w => w.name === stepsWidgetName);

        if (isNaN(seedWidget.value) && isNaN(stepsWidget.value)) {
            console.log("[EfficiencyUpdate]", `Fixing '${node.comfyClass}' node widgets:`, node);
            for (let i = 0; i < node.widgets.length - 1; i++) {
                node.widgets[i].value = node.widgets[i + 1].value;
            }
            node.widgets[node.widgets.length - 1].value = "true";
        }
    }

    // HIGHRES-FIX SCRIPT
    /*  Changes:
            Many new changes where added, so in order to properly update, aquired the values of the original
            widgets, reload a new node, transffer the known original values, and transffer connection.
            This fix is triggered when the upscale_type widget is neither "latent" or "pixel".
    */
    // Check if the current node is "HighRes-Fix Script" and if any of the above fixes were applied
    else if (node.comfyClass === "HighRes-Fix Script") {
        const upscaleTypeWidget = node.widgets.find(w => w.name === "upscale_type");
        
        if (upscaleTypeWidget && upscaleTypeWidget.value !== "latent" && upscaleTypeWidget.value !== "pixel" && upscaleTypeWidget.value !== "both") {
            console.log("[EfficiencyUpdate]", "Reloading 'HighRes-Fix Script' node:", node);

            // Reload the node and get the new node instance
            const newNode = reloadHiResFixNode(node);

            // Update the widgets of the new node
            const targetWidgetNames = ["latent_upscaler", "upscale_by", "hires_steps", "denoise", "iterations"];

            // Extract the first five values of the original node
            const originalValues = originalNode.widgets.slice(0, 5).map(w => w.value);

            targetWidgetNames.forEach((name, index) => {
                const widget = newNode.widgets.find(w => w.name === name);
                if (widget && originalValues[index] !== undefined) {
                    if (name === "latent_upscaler" && typeof originalValues[index] === 'string') {
                        widget.value = originalValues[index].replace("SD-Latent-Upscaler", "city96");
                    } else {
                        widget.value = originalValues[index];
                    }
                }
            });
        }
    }
}

app.registerExtension(ext);
