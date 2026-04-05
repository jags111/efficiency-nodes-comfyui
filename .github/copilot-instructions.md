# Copilot Instructions for Efficiency Nodes ComfyUI

## Repository Overview

This repository provides custom efficiency nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), a powerful node-based UI for Stable Diffusion. The nodes streamline workflows by combining multiple operations into efficient, cached, and preview-enabled nodes.

## Project Structure

- **`efficiency_nodes.py`**: Main file containing all 45+ node class definitions (primary implementation file)
- **`tsc_utils.py`**: Utility functions for caching, tensor operations, and console messaging
- **`__init__.py`**: Entry point that exports NODE_CLASS_MAPPINGS for ComfyUI
- **`py/`**: Specialized modules for upscaling, sampling, encoding, and tiling
- **`node_settings.json`**: Configuration for model caching behavior
- **`requirements.txt`**: Python dependencies (clip-interrogator, simpleeval)

## Core Architecture

### Node Pattern

All custom nodes follow the ComfyUI standard structure:

```python
class TSC_NodeName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {...},  # Required inputs
            "optional": {...},  # Optional inputs
            "hidden": {...}     # Hidden inputs (UNIQUE_ID, PROMPT)
        }
    
    RETURN_TYPES = ("TYPE1", "TYPE2")
    RETURN_NAMES = ("output1", "output2")
    FUNCTION = "method_name"
    CATEGORY = "Efficiency Nodes/SubCategory"
    
    def method_name(self, **kwargs):
        # Node logic here
        return (result1, result2)
```

### Naming Conventions

- **Classes**: Use `TSC_` prefix (creator's initials) + descriptive name in PascalCase
  - Examples: `TSC_EfficientLoader`, `TSC_KSampler`, `TSC_XYplot`
- **Methods**: Use snake_case for all methods
- **Constants**: Use UPPER_SNAKE_CASE for module-level constants

### Custom Data Types

The repository defines several custom types for workflow composition:

- `LORA_STACK`: Tuple for stacking multiple LoRA models
- `CONTROL_NET_STACK`: Tuple for stacking ControlNet configurations
- `SCRIPT`: Type for chaining script operations (XY Plot, HighRes-Fix, etc.)
- `XY`: Type for XY plot data
- `SDXL_TUPLE`: SDXL-specific configuration tuple

## Key Patterns

### 1. Wrapper Pattern

Efficiency nodes wrap base ComfyUI nodes to add features:

```python
# Wraps KSampler with caching, preview, and script support
class TSC_KSampler:
    def sample(self, ...):
        # Check cache
        # Execute base KSampler
        # Store results
        # Handle script execution
        # Return enhanced output
```

### 2. Caching System

Use the caching utilities from `tsc_utils.py`:

```python
from tsc_utils import load_ksampler_results, store_ksampler_results

# Load cached results
cached = load_ksampler_results(unique_id, prompt)

# Store results for future use
store_ksampler_results(unique_id, prompt, results)
```

**Important**: Cache operations use `unique_id` and `prompt` from hidden inputs to ensure per-instance caching.

### 3. Stack Pattern

Support stacking for composable workflows:

```python
"optional": {
    "lora_stack": ("LORA_STACK",),
    "cnet_stack": ("CONTROL_NET_STACK",),
}
```

### 4. Script System

Nodes can execute scripts for advanced workflows:

```python
"optional": {
    "script": ("SCRIPT",),
}

# In node execution:
if script:
    # Execute script logic (XY Plot, HighRes-Fix, etc.)
```

### 5. Dynamic UI Inputs

Use `folder_paths` for dynamic dropdown population:

```python
import folder_paths

"required": {
    "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
    "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
}
```

## Dependencies

### Required

- **PyTorch**: Core tensor operations
- **PIL**: Image processing
- **NumPy**: Array operations
- **clip-interrogator**: Image captioning
- **simpleeval**: Safe expression evaluation

### ComfyUI Integration

The code integrates with ComfyUI via `sys.path` manipulation:

```python
# Pattern used throughout codebase
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))
sys.path.append(comfy_dir)
from comfy import samplers, sd, utils
# ... imports ...
sys.path.remove(comfy_dir)
```

### Optional Dependencies

- **comfyui_controlnet_aux**: ControlNet preprocessing
- **ComfyUI-AnimateDiff-Evolved**: AnimateDiff support

Handle optional dependencies gracefully:

```python
try:
    import optional_module
    NODE_CLASS_MAPPINGS.update({"Node Name": NodeClass})
except ImportError:
    pass
```

## Node Registration

Nodes are registered in `NODE_CLASS_MAPPINGS` dictionary:

```python
NODE_CLASS_MAPPINGS = {
    "Display Name": TSC_ClassName,
    "KSampler (Efficient)": TSC_KSampler,
    "Efficient Loader": TSC_EfficientLoader,
    # ... more nodes
}

# Optional nodes added conditionally
try:
    from simpleeval import simple_eval
    NODE_CLASS_MAPPINGS.update({
        "Simple Eval Examples": TSC_SimpleEval,
    })
except ImportError:
    print("simpleeval not installed, skipping related nodes")
```

## Code Style Guidelines

### Imports

1. Standard library imports first
2. Third-party imports (torch, PIL, numpy)
3. ComfyUI imports (with path manipulation)
4. Local imports (tsc_utils, py modules)

### Error Handling

Use the colored messaging functions from `tsc_utils.py`:

```python
from tsc_utils import error, warning, success

try:
    # Operation
    success("Operation completed")
except Exception as e:
    error(f"Operation failed: {e}")
```

### Input Validation

Validate inputs in the INPUT_TYPES definition:

```python
"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
"sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
```

## Testing and Validation

This repository does not have formal unit tests. Changes should be validated by:

1. **Import Test**: Verify `__init__.py` imports successfully
2. **ComfyUI Integration**: Load nodes in ComfyUI UI and verify they appear
3. **Workflow Test**: Create test workflows and verify node functionality
4. **Error Testing**: Test edge cases and ensure graceful error messages

## Common Patterns to Follow

### Adding a New Node

1. Create class with `TSC_` prefix
2. Define `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`
3. Implement the function method
4. Add to `NODE_CLASS_MAPPINGS`
5. Test in ComfyUI workflow

### Adding Optional Features

1. Wrap in try/except for dependency checking
2. Use `.update()` to add to NODE_CLASS_MAPPINGS
3. Provide fallback or skip if dependency missing
4. Print informative message about missing dependency

### Working with Models

1. Use `folder_paths` for model discovery
2. Implement caching via `tsc_utils` functions
3. Store loaded models in `loaded_objects` dict with unique IDs
4. Handle model loading errors gracefully

### Handling UI Updates

1. Use hidden inputs for `UNIQUE_ID` and `PROMPT` tracking
2. Return UI update dictionaries when needed
3. Follow ComfyUI's output format for preview images

## Performance Considerations

- **Caching**: Always use caching for expensive operations (model loading, sampling)
- **Memory**: Be mindful of GPU memory with large models
- **Preview**: Implement progressive preview for long operations
- **Batching**: Support batch processing where applicable

## Documentation

- Update README.md for new nodes
- Add examples to the [project Wiki](https://github.com/jags111/efficiency-nodes-comfyui/wiki)
- Include workflow JSON examples for complex nodes
- Document any new configuration options in `node_settings.json`

## Key Files to Understand

1. **efficiency_nodes.py**: Study existing nodes for patterns
2. **tsc_utils.py**: Understand caching and utility functions
3. **py/bnk_adv_encode.py**: Advanced CLIP encoding examples
4. **py/smZ_cfg_denoiser.py**: Custom denoiser implementation
5. **__init__.py**: Entry point and version management

## ComfyUI-Specific Tips

- Nodes are instantiated fresh for each workflow execution
- Use `UNIQUE_ID` from hidden inputs for per-node-instance state
- `PROMPT` contains the full workflow graph
- Return types must match RETURN_TYPES exactly
- UI widgets are defined in INPUT_TYPES with tuples
- Use `folder_paths` for discovering models/resources

## Version Information

- Current version: 2.0+ (see `CC_VERSION` in `__init__.py`)
- Published to ComfyUI registry via `pyproject.toml`
- Auto-publishes on main branch when `pyproject.toml` changes

## Resources

- [ComfyUI Repository](https://github.com/comfyanonymous/ComfyUI)
- [Project Wiki](https://github.com/jags111/efficiency-nodes-comfyui/wiki)
- [Project README](../README.md)
- Original author: Luciano Cirino (TSC)
- Current maintainer: jags111
