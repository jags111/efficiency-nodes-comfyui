# Copilot Instructions for efficiency-nodes-comfyui

## Project Overview

This repository contains **Efficiency Nodes for ComfyUI** — a collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that streamline AI image generation workflows and reduce total node count. The project is maintained by jags111 and is a fork/continuation of the original work by Luciano Cirino.

## Repository Structure

```
efficiency-nodes-comfyui/
├── efficiency_nodes.py      # Main file containing all node class definitions
├── tsc_utils.py             # Utility functions (model caching, loading, helpers)
├── __init__.py              # Package entry point, registers WEB_DIRECTORY and exports
├── pyproject.toml           # Project metadata and ComfyUI registry config
├── requirements.txt         # Python dependencies
├── arial.ttf                # Font used for image annotation nodes
├── node_settings.json       # Default node settings
├── py/                      # Helper modules (samplers, encoders, upscalers)
│   ├── bnk_adv_encode.py    # Advanced CLIP text encoding
│   ├── bnk_tiled_samplers.py
│   ├── city96_latent_upscaler.py
│   ├── smZ_cfg_denoiser.py
│   └── ...
├── js/                      # Frontend JavaScript for ComfyUI web interface
│   ├── appearance.js        # Node color themes
│   ├── seedcontrol.js       # Seed widget behavior
│   ├── widgethider.js       # Dynamic widget visibility
│   └── node_options/        # Right-click context menu additions
└── workflows/               # Example workflow JSON files
```

## Architecture & Key Conventions

### Node Class Pattern

All nodes follow the ComfyUI node pattern:

```python
class TSC_MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "param": ("TYPE", {"default": value}) },
            "optional": { "opt_param": ("TYPE",) },
            "hidden":   { "my_unique_id": "UNIQUE_ID" }
        }

    RETURN_TYPES = ("TYPE1", "TYPE2",)
    RETURN_NAMES = ("name1", "name2",)
    FUNCTION = "execute_method"
    CATEGORY = "Efficiency Nodes/Category"

    def execute_method(self, param, opt_param=None, my_unique_id=None):
        # implementation
        return (result1, result2,)
```

- All node classes are prefixed with `TSC_`
- `FUNCTION` must match the name of the method that executes the node
- `CATEGORY` uses the `"Efficiency Nodes/..."` namespace for grouping in the ComfyUI UI
- Nodes with side effects (saving images, printing) should set `OUTPUT_NODE = True`

### Node Registration

Nodes are registered at the bottom of `efficiency_nodes.py` by updating `NODE_CLASS_MAPPINGS`:

```python
NODE_CLASS_MAPPINGS = {
    "KSampler (Efficient)": TSC_KSampler,
    "Efficient Loader": TSC_EfficientLoader,
    # ...
}
```

Optional features (requiring extra packages like `simpleeval` or `animatediff`) are wrapped in `try/except ImportError` blocks and registered conditionally.

### Model Caching (tsc_utils.py)

The `loaded_objects` and `last_helds` globals in `tsc_utils.py` implement a caching layer to avoid reloading models between node executions:

- `loaded_objects["ckpt"]` — checkpoints
- `loaded_objects["lora"]` — LoRA weights
- `loaded_objects["vae"]`  — VAE models
- `last_helds["latent"]` / `last_helds["image"]` — KSampler output caches

Always call `globals_cleanup(prompt)` at the start of loader nodes to evict stale cache entries.

### Frontend JavaScript

- JS files in `js/` are served by ComfyUI via `WEB_DIRECTORY = "js"` in `__init__.py`
- Extensions are registered with `app.registerExtension({ name: "efficiency.<name>", ... })`
- Use `import { app } from "../../scripts/app.js"` for ComfyUI's app object
- Widget helpers (`findWidgetByName`, `addMenuHandler`) are in `js/node_options/common/utils.js`

## Dependencies

- **Python**: `torch`, `PIL` (Pillow), `numpy`, `comfy` (ComfyUI), `folder_paths`
- **Optional Python**: `simpleeval` (Evaluate nodes), `clip_interrogator` (CLIP Interrogator node)
- **ComfyUI extras**: `comfy_extras.nodes_align_your_steps`, `comfy_extras.nodes_gits`, `comfy_extras.nodes_upscale_model`

## Development Guidelines

- **Do not break the node contract**: Changing `RETURN_TYPES`, `RETURN_NAMES`, or `INPUT_TYPES` on existing nodes is a breaking change for saved workflows.
- **Cache awareness**: When modifying loader nodes, preserve calls to `globals_cleanup()` and the caching functions (`load_checkpoint`, `load_lora`, `load_vae`).
- **Error messages**: Use the `warning()` and `error()` helpers from `tsc_utils.py` for consistent console output formatting.
- **Optional imports**: Wrap any new optional feature dependencies in `try/except ImportError` and print a warning when the package is missing.
- **Node naming**: Use the `TSC_` prefix for new node classes. Register them in `NODE_CLASS_MAPPINGS` with a human-readable display name (e.g., `"My New Node": TSC_MyNewNode`).
- **JS extensions**: Follow the existing pattern of using `app.registerExtension` with the `"efficiency.<extensionName>"` naming convention.

## Testing & Validation

There is no automated test suite. Validation is done by loading the nodes in a running ComfyUI instance:

1. Install ComfyUI and place this package in `ComfyUI/custom_nodes/efficiency-nodes-comfyui/`
2. Start ComfyUI and check the console for import errors
3. Open the ComfyUI web interface and verify nodes appear under the `Efficiency Nodes` menu category
4. Load example workflows from the `workflows/` directory to test node behavior

## Publishing

The package is published to the [ComfyUI Registry](https://registry.comfy.org/) via the GitHub Actions workflow `.github/workflows/publish.yml`, triggered when `pyproject.toml` is updated on `main`. Update the `version` field in `pyproject.toml` for each release.
