import { app } from "../../scripts/app.js";

let origProps = {};
let initialized = false;

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

const doesInputWithNameExist = (node, name) => {
    // return node.inputs ? node.inputs.some((input) => input.name === name) : false;
    return false;
};

const HIDDEN_TAG = "tschide";
// Toggle Widget + change size
function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;

    // Store the original properties of the widget if not already stored
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }

    const origSize = node.size;

    // Set the widget type and computeSize based on the show flag
    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

    // Recursively handle linked widgets if they exist
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));

    // Calculate the new height for the node based on its computeSize method
    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

const WIDGET_HEIGHT = 24;
// Use for Multiline Widget Nodes (aka Efficient Loaders)
function toggleWidget_2(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
    
    const isCurrentlyVisible = widget.type !== HIDDEN_TAG + suffix;
    if (isCurrentlyVisible === show) return; // Early exit if widget is already in the desired state

    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }

    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

    if (initialized){
        const adjustment = show ? WIDGET_HEIGHT : -WIDGET_HEIGHT;
        node.setSize([node.size[0], node.size[1] + adjustment]);
    }
}

// New function to handle widget visibility based on input_mode
function handleInputModeWidgetsVisibility(node, inputModeValue) {
    // Utility function to generate widget names up to a certain count
    function generateWidgetNames(baseName, count) {
        return Array.from({ length: count }, (_, i) => `${baseName}_${i + 1}`);
    }

    // Common widget groups
    const batchWidgets = ["batch_path", "subdirectories", "batch_sort", "batch_max"];
    const xbatchWidgets = ["X_batch_path", "X_subdirectories", "X_batch_sort", "X_batch_max"];
    const ckptWidgets = [...generateWidgetNames("ckpt_name", 50)];
    const clipSkipWidgets = [...generateWidgetNames("clip_skip", 50)];
    const vaeNameWidgets = [...generateWidgetNames("vae_name", 50)];
    const loraNameWidgets = [...generateWidgetNames("lora_name", 50)];
    const loraWtWidgets = [...generateWidgetNames("lora_wt", 50)];
    const modelStrWidgets = [...generateWidgetNames("model_str", 50)];
    const clipStrWidgets = [...generateWidgetNames("clip_str", 50)];
    const xWidgets = ["X_batch_count", "X_first_value", "X_last_value"]
    const yWidgets = ["Y_batch_count", "Y_first_value", "Y_last_value"]

    const nodeVisibilityMap = {
        "LoRA Stacker": {
            "simple": [...modelStrWidgets, ...clipStrWidgets],
            "advanced": [...loraWtWidgets]
        },
        "XY Input: Steps": {
            "steps": ["first_start_step", "last_start_step", "first_end_step", "last_end_step", "first_refine_step", "last_refine_step"],
            "start_at_step": ["first_step", "last_step", "first_end_step", "last_end_step", "first_refine_step", "last_refine_step"],
            "end_at_step": ["first_step", "last_step", "first_start_step", "last_start_step", "first_refine_step", "last_refine_step"],
            "refine_at_step": ["first_step", "last_step", "first_start_step", "last_start_step", "first_end_step", "last_end_step"]
        },
        "XY Input: VAE": {
            "VAE Names": [...batchWidgets],
            "VAE Batch": [...vaeNameWidgets, "vae_count"]
        },
        "XY Input: Checkpoint": {
            "Ckpt Names": [...clipSkipWidgets, ...vaeNameWidgets, ...batchWidgets],
            "Ckpt Names+ClipSkip": [...vaeNameWidgets, ...batchWidgets],
            "Ckpt Names+ClipSkip+VAE": [...batchWidgets],
            "Checkpoint Batch": [...ckptWidgets, ...clipSkipWidgets, ...vaeNameWidgets, "ckpt_count"]
        },
        "XY Input: LoRA": {
            "LoRA Names": [...modelStrWidgets, ...clipStrWidgets, ...batchWidgets],
            "LoRA Names+Weights": [...batchWidgets, "model_strength", "clip_strength"],
            "LoRA Batch": [...loraNameWidgets, ...modelStrWidgets, ...clipStrWidgets, "lora_count"]
        },
        "XY Input: LoRA Plot": {
            "X: LoRA Batch, Y: LoRA Weight": ["lora_name", "model_strength", "clip_strength",  "X_first_value", "X_last_value"],
            "X: LoRA Batch, Y: Model Strength": ["lora_name", "model_strength", "model_strength", "X_first_value", "X_last_value"],
            "X: LoRA Batch, Y: Clip Strength": ["lora_name", "clip_strength", "X_first_value", "X_last_value"],
            "X: Model Strength, Y: Clip Strength": [...xbatchWidgets, "model_strength", "clip_strength"],
        },
        "XY Input: Control Net": {
            "strength": ["first_start_percent", "last_start_percent", "first_end_percent", "last_end_percent", "strength"],
            "start_percent": ["first_strength", "last_strength", "first_end_percent", "last_end_percent", "start_percent"],
            "end_percent": ["first_strength", "last_strength", "first_start_percent", "last_start_percent", "end_percent"]
        },
        "XY Input: Control Net Plot": {
            "X: Strength, Y: Start%": ["strength", "start_percent"],
            "X: Strength, Y: End%": ["strength","end_percent"],
            "X: Start%, Y: Strength": ["start_percent", "strength"],
            "X: Start%, Y: End%": ["start_percent", "end_percent"],
            "X: End%, Y: Strength": ["end_percent", "strength"],
            "X: End%, Y: Start%": ["end_percent", "start_percent"],
        }
    };

    const inputModeVisibilityMap = nodeVisibilityMap[node.comfyClass];
    
    if (!inputModeVisibilityMap || !inputModeVisibilityMap[inputModeValue]) return;

    // Reset all widgets to visible
    for (const key in inputModeVisibilityMap) {
        for (const widgetName of inputModeVisibilityMap[key]) {
            const widget = findWidgetByName(node, widgetName);
            toggleWidget(node, widget, true);
        }
    }

    // Hide the specific widgets for the current input_mode value
    for (const widgetName of inputModeVisibilityMap[inputModeValue]) {
        const widget = findWidgetByName(node, widgetName);
        toggleWidget(node, widget, false);
    }
}

// Handle multi-widget visibilities
function handleVisibility(node, countValue, node_type) {
    const inputModeValue = findWidgetByName(node, "input_mode").value;
    const baseNamesMap = {
        "LoRA": ["lora_name", "model_str", "clip_str"],
        "Checkpoint": ["ckpt_name", "clip_skip", "vae_name"],
        "LoRA Stacker": ["lora_name", "model_str", "clip_str", "lora_wt"]
    };

    const baseNames = baseNamesMap[node_type];

    const isBatchMode = inputModeValue.includes("Batch");
    if (isBatchMode) {countValue = 0;}

    for (let i = 1; i <= 50; i++) {
        const nameWidget = findWidgetByName(node, `${baseNames[0]}_${i}`);
        const firstWidget = findWidgetByName(node, `${baseNames[1]}_${i}`);
        const secondWidget = findWidgetByName(node, `${baseNames[2]}_${i}`);
        const thirdWidget = node_type === "LoRA Stacker" ? findWidgetByName(node, `${baseNames[3]}_${i}`) : null;

        if (i <= countValue) {
            toggleWidget(node, nameWidget, true);

            if (node_type === "LoRA Stacker") {
                if (inputModeValue === "simple") {
                    toggleWidget(node, firstWidget, false);   // model_str
                    toggleWidget(node, secondWidget, false); // clip_str
                    toggleWidget(node, thirdWidget, true);  // lora_wt
                } else if (inputModeValue === "advanced") {
                    toggleWidget(node, firstWidget, true);   // model_str
                    toggleWidget(node, secondWidget, true);  // clip_str
                    toggleWidget(node, thirdWidget, false);   // lora_wt
                }
            } else if (node_type === "Checkpoint") {
                if (inputModeValue.includes("ClipSkip")){toggleWidget(node, firstWidget, true);}
                if (inputModeValue.includes("VAE")){toggleWidget(node, secondWidget, true);}
            } else if (node_type === "LoRA") {
                if (inputModeValue.includes("Weights")){
                    toggleWidget(node, firstWidget, true);
                    toggleWidget(node, secondWidget, true);
                }
            }
        }
        else {
            toggleWidget(node, nameWidget, false);
            toggleWidget(node, firstWidget, false);
            toggleWidget(node, secondWidget, false);
            if (thirdWidget) {toggleWidget(node, thirdWidget, false);}
        }
    }
}

// Sampler & Scheduler XY input visibility logic
function handleSamplerSchedulerVisibility(node, countValue, targetParameter) {
    for (let i = 1; i <= 50; i++) {
        const samplerWidget = findWidgetByName(node, `sampler_${i}`);
        const schedulerWidget = findWidgetByName(node, `scheduler_${i}`);

        if (i <= countValue) {
            if (targetParameter === "sampler") {
                toggleWidget(node, samplerWidget, true);
                toggleWidget(node, schedulerWidget, false);
            } else if (targetParameter === "scheduler") {
                toggleWidget(node, samplerWidget, false);
                toggleWidget(node, schedulerWidget, true);
            } else { // targetParameter is "sampler & scheduler"
                toggleWidget(node, samplerWidget, true);
                toggleWidget(node, schedulerWidget, true);
            }
        } else {
            toggleWidget(node, samplerWidget, false);
            toggleWidget(node, schedulerWidget, false);
        }
    }
}

// Handle simple widget visibility based on a count
function handleWidgetVisibility(node, thresholdValue, widgetNamePrefix, maxCount) {
    for (let i = 1; i <= maxCount; i++) {
        const widget = findWidgetByName(node, `${widgetNamePrefix}${i}`);
        if (widget) {
            toggleWidget(node, widget, i <= thresholdValue);
        }
    }
}

// Disable the 'Ckpt Name+ClipSkip+VAE' option if 'target_ckpt' is "Refiner"
let last_ckpt_input_mode;
let last_target_ckpt;
function xyCkptRefinerOptionsRemove(widget, node) {

    let target_ckpt = findWidgetByName(node, "target_ckpt").value
    let input_mode = widget.value

    if ((input_mode === "Ckpt Names+ClipSkip+VAE") && (target_ckpt === "Refiner")) {
        if (last_ckpt_input_mode === "Ckpt Names+ClipSkip") {
            if (last_target_ckpt === "Refiner"){
                widget.value = "Checkpoint Batch";
            } else {widget.value = "Ckpt Names+ClipSkip";}
        } else if (last_ckpt_input_mode === "Checkpoint Batch") {
            if (last_target_ckpt === "Refiner"){
                widget.value = "Ckpt Names+ClipSkip";
            } else {widget.value = "Checkpoint Batch";}
        } else if (last_ckpt_input_mode !== 'undefined') {
            widget.value = last_ckpt_input_mode;
        } else {
            widget.value = "Ckpt Names";
        }
    } else if (input_mode !== "Ckpt Names+ClipSkip+VAE"){
        last_ckpt_input_mode = input_mode;
    }
    last_target_ckpt = target_ckpt
}

// Create a map of node titles to their respective widget handlers
const nodeWidgetHandlers = {
    "Efficient Loader": {
        'lora_name': handleEfficientLoaderLoraName
    },
    "Eff. Loader SDXL": {
        'refiner_ckpt_name': handleEffLoaderSDXLRefinerCkptName
    },
    "LoRA Stacker": {
        'input_mode': handleLoRAStackerInputMode,
        'lora_count': handleLoRAStackerLoraCount
    },
    "XY Input: Steps": {
        'target_parameter': handleXYInputStepsTargetParameter
    },
    "XY Input: Sampler/Scheduler": {
        'target_parameter': handleXYInputSamplerSchedulerTargetParameter,
        'input_count': handleXYInputSamplerSchedulerInputCount
    },
    "XY Input: VAE": {
        'input_mode': handleXYInputVAEInputMode,
        'vae_count': handleXYInputVAEVaeCount
    },
    "XY Input: Prompt S/R": {
        'replace_count': handleXYInputPromptSRReplaceCount
    },
    "XY Input: Checkpoint": {
        'input_mode': handleXYInputCheckpointInputMode,
        'ckpt_count': handleXYInputCheckpointCkptCount,
        'target_ckpt': handleXYInputCheckpointTargetCkpt
    },
    "XY Input: LoRA": {
        'input_mode': handleXYInputLoRAInputMode,
        'lora_count': handleXYInputLoRALoraCount
    },
    "XY Input: LoRA Plot": {
        'input_mode': handleXYInputLoRAPlotInputMode
    },
    "XY Input: LoRA Stacks": {
        'node_state': handleXYInputLoRAStacksNodeState
    },
    "XY Input: Control Net": {
        'target_parameter': handleXYInputControlNetTargetParameter
    },
    "XY Input: Control Net Plot": {
        'plot_type': handleXYInputControlNetPlotPlotType
    },
    "Noise Control Script": {
        'add_seed_noise': handleNoiseControlScript
    },
    "HighRes-Fix Script": {
        'upscale_type': handleHiResFixScript,
        'use_same_seed': handleHiResFixScript,
        'use_controlnet':handleHiResFixScript
    },
    "Tiled Upscaler Script": {
        'use_controlnet':handleTiledUpscalerScript
    },
};

// In the main function where widgetLogic is called
function widgetLogic(node, widget) {
    // Retrieve the handler for the current node title and widget name
    const handler = nodeWidgetHandlers[node.comfyClass]?.[widget.name];
    if (handler) {
        handler(node, widget);
    }
}

// Efficient Loader Handlers
function handleEfficientLoaderLoraName(node, widget) {
    if (widget.value === 'None') {       
        toggleWidget_2(node, findWidgetByName(node, 'lora_model_strength'));
        toggleWidget_2(node, findWidgetByName(node, 'lora_clip_strength'));
    } else {
        toggleWidget_2(node, findWidgetByName(node, 'lora_model_strength'), true);
        toggleWidget_2(node, findWidgetByName(node, 'lora_clip_strength'), true);
    }
}

// Eff. Loader SDXL Handlers
function handleEffLoaderSDXLRefinerCkptName(node, widget) {
    if (widget.value === 'None') {
        toggleWidget_2(node, findWidgetByName(node, 'refiner_clip_skip'));
        toggleWidget_2(node, findWidgetByName(node, 'positive_ascore'));
        toggleWidget_2(node, findWidgetByName(node, 'negative_ascore'));
    } else {
        toggleWidget_2(node, findWidgetByName(node, 'refiner_clip_skip'), true);
        toggleWidget_2(node, findWidgetByName(node, 'positive_ascore'), true);
        toggleWidget_2(node, findWidgetByName(node, 'negative_ascore'), true);
    }
}

// Noise Control Script Seed Handler
function handleNoiseControlScript(node, widget) {

    function ensureSeedControlExists(callback) {
        if (node.seedControl && node.seedControl.lastSeedButton) {
            callback();
        } else {
            setTimeout(() => ensureSeedControlExists(callback), 0);
        }
    }

    ensureSeedControlExists(() => {
        if (widget.value === false) {
            toggleWidget(node, findWidgetByName(node, 'seed'));
            toggleWidget(node, findWidgetByName(node, 'weight'));
            toggleWidget(node, node.seedControl.lastSeedButton);
            node.seedControl.lastSeedButton.disabled = true; // Disable the button
        } else {
            toggleWidget(node, findWidgetByName(node, 'seed'), true);
            toggleWidget(node, findWidgetByName(node, 'weight'), true);
            node.seedControl.lastSeedButton.disabled = false; // Enable the button
            toggleWidget(node, node.seedControl.lastSeedButton, true);
        }
    });

}

/// HighRes-Fix Script Handlers
function handleHiResFixScript(node, widget) {

    function ensureSeedControlExists(callback) {
        if (node.seedControl && node.seedControl.lastSeedButton) {
            callback();
        } else {
            setTimeout(() => ensureSeedControlExists(callback), 0);
        }
    }

    if (findWidgetByName(node, 'upscale_type').value === "latent") {     
        toggleWidget(node, findWidgetByName(node, 'pixel_upscaler'));

        toggleWidget(node, findWidgetByName(node, 'hires_ckpt_name'), true);
        toggleWidget(node, findWidgetByName(node, 'latent_upscaler'), true);
        toggleWidget(node, findWidgetByName(node, 'use_same_seed'), true);
        toggleWidget(node, findWidgetByName(node, 'hires_steps'), true);
        toggleWidget(node, findWidgetByName(node, 'denoise'), true);
        toggleWidget(node, findWidgetByName(node, 'iterations'), true);

        ensureSeedControlExists(() => {
            if (findWidgetByName(node, 'use_same_seed').value == true) {
                toggleWidget(node, findWidgetByName(node, 'seed'));
                toggleWidget(node, node.seedControl.lastSeedButton);
                node.seedControl.lastSeedButton.disabled = true; // Disable the button
            } else {
                toggleWidget(node, findWidgetByName(node, 'seed'), true);
                node.seedControl.lastSeedButton.disabled = false; // Enable the button
                toggleWidget(node, node.seedControl.lastSeedButton, true);
            }
        });

        if (findWidgetByName(node, 'use_controlnet').value == '_'){
            toggleWidget(node, findWidgetByName(node, 'use_controlnet'));
            toggleWidget(node, findWidgetByName(node, 'control_net_name'));
            toggleWidget(node, findWidgetByName(node, 'strength'));
            toggleWidget(node, findWidgetByName(node, 'preprocessor'));
            toggleWidget(node, findWidgetByName(node, 'preprocessor_imgs'));
        }
        else{
            toggleWidget(node, findWidgetByName(node, 'use_controlnet'), true);

            if (findWidgetByName(node, 'use_controlnet').value == true){
                toggleWidget(node, findWidgetByName(node, 'control_net_name'), true);
                toggleWidget(node, findWidgetByName(node, 'strength'), true);
                toggleWidget(node, findWidgetByName(node, 'preprocessor'), true);
                toggleWidget(node, findWidgetByName(node, 'preprocessor_imgs'), true);
            }
            else{
                toggleWidget(node, findWidgetByName(node, 'control_net_name'));
                toggleWidget(node, findWidgetByName(node, 'strength'));
                toggleWidget(node, findWidgetByName(node, 'preprocessor'));
                toggleWidget(node, findWidgetByName(node, 'preprocessor_imgs'));
            }
        }

    } else if (findWidgetByName(node, 'upscale_type').value === "pixel") {
        toggleWidget(node, findWidgetByName(node, 'hires_ckpt_name'));
        toggleWidget(node, findWidgetByName(node, 'latent_upscaler'));
        toggleWidget(node, findWidgetByName(node, 'use_same_seed'));
        toggleWidget(node, findWidgetByName(node, 'hires_steps'));
        toggleWidget(node, findWidgetByName(node, 'denoise'));
        toggleWidget(node, findWidgetByName(node, 'iterations'));
        toggleWidget(node, findWidgetByName(node, 'seed'));
        ensureSeedControlExists(() => {
            toggleWidget(node, node.seedControl.lastSeedButton);
            node.seedControl.lastSeedButton.disabled = true; // Disable the button
        });
        toggleWidget(node, findWidgetByName(node, 'use_controlnet'));
        toggleWidget(node, findWidgetByName(node, 'control_net_name'));
        toggleWidget(node, findWidgetByName(node, 'strength'));
        toggleWidget(node, findWidgetByName(node, 'preprocessor'));
        toggleWidget(node, findWidgetByName(node, 'preprocessor_imgs'));

        toggleWidget(node, findWidgetByName(node, 'pixel_upscaler'), true);
        
    } else if (findWidgetByName(node, 'upscale_type').value === "both") {
        toggleWidget(node, findWidgetByName(node, 'pixel_upscaler'), true);
        toggleWidget(node, findWidgetByName(node, 'hires_ckpt_name'), true);
        toggleWidget(node, findWidgetByName(node, 'latent_upscaler'), true);
        toggleWidget(node, findWidgetByName(node, 'use_same_seed'), true);
        toggleWidget(node, findWidgetByName(node, 'hires_steps'), true);
        toggleWidget(node, findWidgetByName(node, 'denoise'), true);
        toggleWidget(node, findWidgetByName(node, 'iterations'), true);

        ensureSeedControlExists(() => {
            if (findWidgetByName(node, 'use_same_seed').value == true) {
                toggleWidget(node, findWidgetByName(node, 'seed'));
                toggleWidget(node, node.seedControl.lastSeedButton);
                node.seedControl.lastSeedButton.disabled = true; // Disable the button
            } else {
                toggleWidget(node, findWidgetByName(node, 'seed'), true);
                node.seedControl.lastSeedButton.disabled = false; // Enable the button
                toggleWidget(node, node.seedControl.lastSeedButton, true);
            }
        });

        if (findWidgetByName(node, 'use_controlnet').value == '_'){
            toggleWidget(node, findWidgetByName(node, 'use_controlnet'));
            toggleWidget(node, findWidgetByName(node, 'control_net_name'));
            toggleWidget(node, findWidgetByName(node, 'strength'));
            toggleWidget(node, findWidgetByName(node, 'preprocessor'));
            toggleWidget(node, findWidgetByName(node, 'preprocessor_imgs'));
        }
        else{
            toggleWidget(node, findWidgetByName(node, 'use_controlnet'), true);

            if (findWidgetByName(node, 'use_controlnet').value == true){
                toggleWidget(node, findWidgetByName(node, 'control_net_name'), true);
                toggleWidget(node, findWidgetByName(node, 'strength'), true);
                toggleWidget(node, findWidgetByName(node, 'preprocessor'), true);
                toggleWidget(node, findWidgetByName(node, 'preprocessor_imgs'), true);
            }
            else{
                toggleWidget(node, findWidgetByName(node, 'control_net_name'));
                toggleWidget(node, findWidgetByName(node, 'strength'));
                toggleWidget(node, findWidgetByName(node, 'preprocessor'));
                toggleWidget(node, findWidgetByName(node, 'preprocessor_imgs'));
            }
        }
    }
}

/// Tiled Upscaler Script Handler
function handleTiledUpscalerScript(node, widget) {
    if (findWidgetByName(node, 'use_controlnet').value == true){
        toggleWidget(node, findWidgetByName(node, 'tile_controlnet'), true);
        toggleWidget(node, findWidgetByName(node, 'strength'), true);
    }
    else{
        toggleWidget(node, findWidgetByName(node, 'tile_controlnet'));
        toggleWidget(node, findWidgetByName(node, 'strength'));
    }
}

// LoRA Stacker Handlers
function handleLoRAStackerInputMode(node, widget) {
    handleInputModeWidgetsVisibility(node, widget.value);
    handleVisibility(node, findWidgetByName(node, "lora_count").value, "LoRA Stacker");
}

function handleLoRAStackerLoraCount(node, widget) {
    handleVisibility(node, widget.value, "LoRA Stacker");
}

// XY Input: Steps Handlers
function handleXYInputStepsTargetParameter(node, widget) {
    handleInputModeWidgetsVisibility(node, widget.value);
}

// XY Input: Sampler/Scheduler Handlers
function handleXYInputSamplerSchedulerTargetParameter(node, widget) {
    handleSamplerSchedulerVisibility(node, findWidgetByName(node, 'input_count').value, widget.value);
}

function handleXYInputSamplerSchedulerInputCount(node, widget) {
    handleSamplerSchedulerVisibility(node, widget.value, findWidgetByName(node, 'target_parameter').value);
}

// XY Input: VAE Handlers
function handleXYInputVAEInputMode(node, widget) {
    handleInputModeWidgetsVisibility(node, widget.value);
    if (widget.value === "VAE Names") {
        handleWidgetVisibility(node, findWidgetByName(node, "vae_count").value, "vae_name_", 50);
    } else {
        handleWidgetVisibility(node, 0, "vae_name_", 50);
    }
}

function handleXYInputVAEVaeCount(node, widget) {
    if (findWidgetByName(node, "input_mode").value === "VAE Names") {
        handleWidgetVisibility(node, widget.value, "vae_name_", 50);
    }
}

// XY Input: Prompt S/R Handlers
function handleXYInputPromptSRReplaceCount(node, widget) {
    handleWidgetVisibility(node, widget.value, "replace_", 49);
}

// XY Input: Checkpoint Handlers
function handleXYInputCheckpointInputMode(node, widget) {
    xyCkptRefinerOptionsRemove(widget, node);
    handleInputModeWidgetsVisibility(node, widget.value);
    handleVisibility(node, findWidgetByName(node, "ckpt_count").value, "Checkpoint");
}

function handleXYInputCheckpointCkptCount(node, widget) {
    handleVisibility(node, widget.value, "Checkpoint");
}

function handleXYInputCheckpointTargetCkpt(node, widget) {
    xyCkptRefinerOptionsRemove(findWidgetByName(node, "input_mode"), node);
}

// XY Input: LoRA Handlers
function handleXYInputLoRAInputMode(node, widget) {
    handleInputModeWidgetsVisibility(node, widget.value);
    handleVisibility(node, findWidgetByName(node, "lora_count").value, "LoRA");
}

function handleXYInputLoRALoraCount(node, widget) {
    handleVisibility(node, widget.value, "LoRA");
}

// XY Input: LoRA Plot Handlers
function handleXYInputLoRAPlotInputMode(node, widget) {
    handleInputModeWidgetsVisibility(node, widget.value);
}

// XY Input: LoRA Stacks Handlers
function handleXYInputLoRAStacksNodeState(node, widget) {
    toggleWidget(node, findWidgetByName(node, "node_state"), false);
}

// XY Input: Control Net Handlers
function handleXYInputControlNetTargetParameter(node, widget) {
    handleInputModeWidgetsVisibility(node, widget.value);
}

// XY Input: Control Net Plot Handlers
function handleXYInputControlNetPlotPlotType(node, widget) {
    handleInputModeWidgetsVisibility(node, widget.value);
}

app.registerExtension({
    name: "efficiency.widgethider",
    nodeCreated(node) {
        if (!nodeWidgetHandlers[node.comfyClass]) return;
        for (const w of node.widgets || []) {
            if (!nodeWidgetHandlers[node.comfyClass][w.name]) continue;
            let widgetValue = w.value;

            // Store the original descriptor if it exists
            let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value');
            if (!originalDescriptor) {
                originalDescriptor = Object.getOwnPropertyDescriptor(w.constructor.prototype, 'value');
            }

            widgetLogic(node, w);

            Object.defineProperty(w, 'value', {
                get() {
                    // If there's an original getter, use it. Otherwise, return widgetValue.
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;

                    return valueToReturn;
                },
                set(newVal) {

                    // If there's an original setter, use it. Otherwise, set widgetValue.
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else {
                        widgetValue = newVal;
                    }

                    widgetLogic(node, w);
                }
            });
        }
        setTimeout(() => {initialized = true;}, 500);
    }
});

