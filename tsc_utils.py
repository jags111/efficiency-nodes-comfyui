# Efficiency Nodes Utility functions
from torch import Tensor
import torch
from PIL import Image
import numpy as np
import os
import sys
import io
from contextlib import contextmanager
import json
import folder_paths

# Get the absolute path of the parent directory of the current script
my_dir = os.path.dirname(os.path.abspath(__file__))

# Add the My directory path to the sys.path list
sys.path.append(my_dir)

# Construct the absolute path to the ComfyUI directory
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Add the ComfyUI directory path to the sys.path list
sys.path.append(comfy_dir)

# Import functions from ComfyUI
import comfy.sd
import comfy.utils
import latent_preview
from comfy.cli_args import args

# Cache for Efficiency Node models
loaded_objects = {
    "ckpt": [], # (ckpt_name, ckpt_model, clip, bvae, [id])
    "refn": [], # (ckpt_name, ckpt_model, clip, bvae, [id])
    "vae": [],  # (vae_name, vae, [id])
    "lora": []  # ([(lora_name, strength_model, strength_clip)], ckpt_name, lora_model, clip_lora, [id])
}

# Cache for Efficient Ksamplers
last_helds = {
    "latent": [],   # (latent, [parameters], id)    # Base sampling latent results
    "image": [],    # (image, id)                   # Base sampling image results
    "cnet_img": []  # (cnet_img, [parameters], id)  # HiRes-Fix control net preprocessor image results
}

def load_ksampler_results(key: str, my_unique_id, parameters_list=None):
    global last_helds
    for data in last_helds[key]:
        id_ = data[-1]  # ID is always the last element in the tuple
        if id_ == my_unique_id:
            if parameters_list is not None:
                # Ensure tuple has at least 3 elements and match with parameters_list
                if len(data) >= 3 and data[1] == parameters_list:
                    return data[0]
            else:
                return data[0]
    return None

def store_ksampler_results(key: str, my_unique_id, value, parameters_list=None):
    global last_helds

    for i, data in enumerate(last_helds[key]):
        id_ = data[-1]  # ID will always be the last in the tuple
        if id_ == my_unique_id:
            # Check if parameters_list is provided or not
            updated_data = (value, parameters_list, id_) if parameters_list is not None else (value, id_)
            last_helds[key][i] = updated_data
            return True

    # If parameters_list is given
    if parameters_list is not None:
        last_helds[key].append((value, parameters_list, my_unique_id))
    else:
        last_helds[key].append((value, my_unique_id))
    return True

# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Convert tensor to PIL, resize it, and convert back to tensor
def quick_resize(source_tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    resized_images = []
    for img in source_tensor:
        resized_pil = tensor2pil(img.squeeze(0)).resize((target_shape[2], target_shape[1]), Image.ANTIALIAS)
        resized_images.append(pil2tensor(resized_pil).squeeze(0))
    return torch.stack(resized_images, dim=0)

# Create a function to compute the hash of a tensor
import hashlib
def tensor_to_hash(tensor):
    byte_repr = tensor.cpu().numpy().tobytes()  # Convert tensor to bytes
    return hashlib.sha256(byte_repr).hexdigest()  # Compute hash

# Color coded messages functions
MESSAGE_COLOR = "\033[36m"  # Cyan
XYPLOT_COLOR = "\033[35m"   # Purple
SUCCESS_COLOR = "\033[92m"  # Green
WARNING_COLOR = "\033[93m"  # Yellow
ERROR_COLOR = "\033[91m"    # Red
INFO_COLOR = "\033[90m"     # Gray
def format_message(text, color_code):
    RESET_COLOR = "\033[0m"
    return f"{color_code}{text}{RESET_COLOR}"
def message(text):
    return format_message(text, MESSAGE_COLOR)
def warning(text):
    return format_message(text, WARNING_COLOR)
def error(text):
    return format_message(text, ERROR_COLOR)
def success(text):
    return format_message(text, SUCCESS_COLOR)
def xyplot_message(text):
    return format_message(text, XYPLOT_COLOR)
def info(text):
    return format_message(text, INFO_COLOR)

def extract_node_info(prompt, id, indirect_key=None):
    # Convert ID to string
    id = str(id)
    node_id = None

    # If an indirect_key (like 'script') is provided, perform a two-step lookup
    if indirect_key:
        # Ensure the id exists in the prompt and has an 'inputs' entry with the indirect_key
        if id in prompt and 'inputs' in prompt[id] and indirect_key in prompt[id]['inputs']:
            # Extract the indirect_id
            indirect_id = prompt[id]['inputs'][indirect_key][0]

            # Ensure the indirect_id exists in the prompt
            if indirect_id in prompt:
                node_id = indirect_id
                return prompt[indirect_id].get('class_type', None), node_id

        # If indirect_key is not found within the prompt
        return None, None

    # If no indirect_key is provided, perform a direct lookup
    return prompt.get(id, {}).get('class_type', None), node_id

def extract_node_value(prompt, id, key):
    # If ID is in data, return its 'inputs' value for a given key. Otherwise, return None.
    return prompt.get(str(id), {}).get('inputs', {}).get(key, None)

def print_loaded_objects_entries(id=None, prompt=None, show_id=False):
    print("-" * 40)  # Print an empty line followed by a separator line
    if id is not None:
        id = str(id)  # Convert ID to string
    if prompt is not None and id is not None:
        node_name, _ = extract_node_info(prompt, id)
        if show_id:
            print(f"\033[36m{node_name} Models Cache: (node_id:{int(id)})\033[0m")
        else:
            print(f"\033[36m{node_name} Models Cache:\033[0m")
    elif id is None:
        print(f"\033[36mGlobal Models Cache:\033[0m")
    else:
        print(f"\033[36mModels Cache: \nnode_id:{int(id)}\033[0m")
    entries_found = False
    for key in ["ckpt", "refn", "vae", "lora"]:
        entries_with_id = loaded_objects[key] if id is None else [entry for entry in loaded_objects[key] if id in entry[-1]]
        if not entries_with_id:  # If no entries with the chosen ID, print None and skip this key
            continue
        entries_found = True
        print(f"{key.capitalize()}:")
        for i, entry in enumerate(entries_with_id, 1):  # Start numbering from 1
            if key == "lora":
                base_ckpt_name = os.path.splitext(os.path.basename(entry[1]))[0]  # Split logic for base_ckpt
                if id is None:
                    associated_ids = ', '.join(map(str, entry[-1]))  # Gather all associated ids
                    print(f"  [{i}] base_ckpt: {base_ckpt_name} (ids: {associated_ids})")
                else:
                    print(f"  [{i}] base_ckpt: {base_ckpt_name}")
                for name, strength_model, strength_clip in entry[0]:
                    lora_model_info = f"{os.path.splitext(os.path.basename(name))[0]}({round(strength_model, 2)},{round(strength_clip, 2)})"
                    print(f"      lora(mod,clip): {lora_model_info}")
            else:
                name_without_ext = os.path.splitext(os.path.basename(entry[0]))[0]
                if id is None:
                    associated_ids = ', '.join(map(str, entry[-1]))  # Gather all associated ids
                    print(f"  [{i}] {name_without_ext} (ids: {associated_ids})")
                else:
                    print(f"  [{i}] {name_without_ext}")
    if not entries_found:
        print("-")

# This function cleans global variables associated with nodes that are no longer detected on UI
def globals_cleanup(prompt):
    global loaded_objects
    global last_helds

    # Step 1: Clean up last_helds
    for key in list(last_helds.keys()):
        original_length = len(last_helds[key])
        last_helds[key] = [
            (*values, id_)
            for *values, id_ in last_helds[key]
            if str(id_) in prompt.keys()
        ]

    # Step 2: Clean up loaded_objects
    for key in list(loaded_objects.keys()):
        for i, tup in enumerate(list(loaded_objects[key])):
            # Remove ids from id array in each tuple that don't exist in prompt
            id_array = [id for id in tup[-1] if str(id) in prompt.keys()]
            if len(id_array) != len(tup[-1]):
                if id_array:
                    loaded_objects[key][i] = tup[:-1] + (id_array,)
                    ###print(f'Updated tuple at index {i} in {key} in loaded_objects: {loaded_objects[key][i]}')
                else:
                    # If id array becomes empty, delete the corresponding tuple
                    loaded_objects[key].remove(tup)
                    ###print(f'Deleted tuple at index {i} in {key} in loaded_objects because its id array became empty.')

def load_checkpoint(ckpt_name, id, output_vae=True, cache=None, cache_overwrite=True, ckpt_type="ckpt"):
    global loaded_objects

    # Create copies of the arguments right at the start
    ckpt_name = ckpt_name.copy() if isinstance(ckpt_name, (list, dict, set)) else ckpt_name

    # Check if the type is valid
    if ckpt_type not in ["ckpt", "refn"]:
        raise ValueError(f"Invalid checkpoint type: {ckpt_type}")

    for entry in loaded_objects[ckpt_type]:
        if entry[0] == ckpt_name:
            _, model, clip, vae, ids = entry
            cache_full = cache and len([entry for entry in loaded_objects[ckpt_type] if id in entry[-1]]) >= cache

            if cache_full:
                clear_cache(id, cache, ckpt_type)
            elif id not in ids:
                ids.append(id)

            return model, clip, vae

    if os.path.isabs(ckpt_name):
        ckpt_path = ckpt_name
    else:
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    with suppress_output():
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

    model = out[0]
    clip = out[1]
    vae = out[2] if output_vae else None  # Load VAE from the checkpoint path only if output_vae is True

    if cache:
        cache_list = [entry for entry in loaded_objects[ckpt_type] if id in entry[-1]]
        if len(cache_list) < cache:
            loaded_objects[ckpt_type].append((ckpt_name, model, clip, vae, [id]))
        else:
            clear_cache(id, cache, ckpt_type)
            if cache_overwrite:
                for e in loaded_objects[ckpt_type]:
                    if id in e[-1]:
                        e[-1].remove(id)
                        # If the id list becomes empty, remove the entry from the ckpt_type list
                        if not e[-1]:
                            loaded_objects[ckpt_type].remove(e)
                        break
                loaded_objects[ckpt_type].append((ckpt_name, model, clip, vae, [id]))

    return model, clip, vae

def get_bvae_by_ckpt_name(ckpt_name):
    for ckpt in loaded_objects["ckpt"]:
        if ckpt[0] == ckpt_name:
            return ckpt[3]  # return 'bvae' variable
    return None  # return None if no match is found

def load_vae(vae_name, id, cache=None, cache_overwrite=False):
    global loaded_objects

    # Create copies of the argument right at the start
    vae_name = vae_name.copy() if isinstance(vae_name, (list, dict, set)) else vae_name

    for i, entry in enumerate(loaded_objects["vae"]):
        if entry[0] == vae_name:
            vae, ids = entry[1], entry[2]
            if id not in ids:
                if cache and len([entry for entry in loaded_objects["vae"] if id in entry[-1]]) >= cache:
                    return vae
                ids.append(id)
            if cache:
                clear_cache(id, cache, "vae")
            return vae

    if os.path.isabs(vae_name):
        vae_path = vae_name
    else:
        vae_path = folder_paths.get_full_path("vae", vae_name)

    sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=sd)

    if cache:
        if len([entry for entry in loaded_objects["vae"] if id in entry[-1]]) < cache:
            loaded_objects["vae"].append((vae_name, vae, [id]))
        else:
            clear_cache(id, cache, "vae")
            if cache_overwrite:
                # Find the first entry with the id, remove the id from the entry's id list
                for e in loaded_objects["vae"]:
                    if id in e[-1]:
                        e[-1].remove(id)
                        # If the id list becomes empty, remove the entry from the "vae" list
                        if not e[-1]:
                            loaded_objects["vae"].remove(e)
                        break
                loaded_objects["vae"].append((vae_name, vae, [id]))

    return vae

def load_lora(lora_params, ckpt_name, id, cache=None, ckpt_cache=None, cache_overwrite=False):
    global loaded_objects

    # Create copies of the arguments right at the start
    lora_params = lora_params.copy() if isinstance(lora_params, (list, dict, set)) else lora_params
    ckpt_name = ckpt_name.copy() if isinstance(ckpt_name, (list, dict, set)) else ckpt_name

    for entry in loaded_objects["lora"]:

        # Convert to sets and compare
        if set(entry[0]) == set(lora_params) and entry[1] == ckpt_name:

            _, _, lora_model, lora_clip, ids = entry
            cache_full = cache and len([entry for entry in loaded_objects["lora"] if id in entry[-1]]) >= cache

            if cache_full:
                clear_cache(id, cache, "lora")
            elif id not in ids:
                ids.append(id)

            # Additional cache handling for 'ckpt' just like in 'load_checkpoint' function
            for ckpt_entry in loaded_objects["ckpt"]:
                if ckpt_entry[0] == ckpt_name:
                    _, _, _, _, ckpt_ids = ckpt_entry
                    ckpt_cache_full = ckpt_cache and len(
                        [ckpt_entry for ckpt_entry in loaded_objects["ckpt"] if id in ckpt_entry[-1]]) >= ckpt_cache

                    if ckpt_cache_full:
                        clear_cache(id, ckpt_cache, "ckpt")
                    elif id not in ckpt_ids:
                        ckpt_ids.append(id)

            return lora_model, lora_clip

    def recursive_load_lora(lora_params, ckpt, clip, id, ckpt_cache, cache_overwrite, folder_paths):
        if len(lora_params) == 0:
            return ckpt, clip

        lora_name, strength_model, strength_clip = lora_params[0]
        if os.path.isabs(lora_name):
            lora_path = lora_name
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)

        try:
            lora_model, lora_clip = comfy.sd.load_lora_for_models(
                ckpt,
                clip,
                comfy.utils.load_torch_file(lora_path),
                strength_model,
                strength_clip,
            )
        except Exception as e:
            raise ValueError("Error loading Lora file: {} \n{}".format(lora_name, e))

        # Call the function again with the new lora_model and lora_clip and the remaining tuples
        return recursive_load_lora(lora_params[1:], lora_model, lora_clip, id, ckpt_cache, cache_overwrite, folder_paths)

    # Unpack lora parameters from the first element of the list for now
    lora_name, strength_model, strength_clip = lora_params[0]
    ckpt, clip, _ = load_checkpoint(ckpt_name, id, cache=ckpt_cache)

    lora_model, lora_clip = recursive_load_lora(lora_params, ckpt, clip, id, ckpt_cache, cache_overwrite, folder_paths)

    if cache:
        if len([entry for entry in loaded_objects["lora"] if id in entry[-1]]) < cache:
            loaded_objects["lora"].append((lora_params, ckpt_name, lora_model, lora_clip, [id]))
        else:
            clear_cache(id, cache, "lora")
            if cache_overwrite:
                # Find the first entry with the id, remove the id from the entry's id list
                for e in loaded_objects["lora"]:
                    if id in e[-1]:
                        e[-1].remove(id)
                        # If the id list becomes empty, remove the entry from the "lora" list
                        if not e[-1]:
                            loaded_objects["lora"].remove(e)
                        break
                loaded_objects["lora"].append((lora_params, ckpt_name, lora_model, lora_clip, [id]))

    return lora_model, lora_clip

def clear_cache(id, cache, dict_name):
    """
    Clear the cache for a specific id in a specific dictionary.
    If the cache limit is reached for a specific id, deletes the id from the oldest entry.
    If the id array of the entry becomes empty, deletes the entry.
    """
    # Get all entries associated with the id_element
    id_associated_entries = [entry for entry in loaded_objects[dict_name] if id in entry[-1]]
    while len(id_associated_entries) > cache:
        # Identify an older entry (but not necessarily the oldest) containing id
        older_entry = id_associated_entries[0]
        # Remove the id_element from the older entry
        older_entry[-1].remove(id)
        # If the id array of the older entry becomes empty after this, delete the entry
        if not older_entry[-1]:
            loaded_objects[dict_name].remove(older_entry)
        # Update the id_associated_entries
        id_associated_entries = [entry for entry in loaded_objects[dict_name] if id in entry[-1]]


def clear_cache_by_exception(node_id, vae_dict=None, ckpt_dict=None, lora_dict=None, refn_dict=None):
    global loaded_objects

    dict_mapping = {
        "vae_dict": "vae",
        "ckpt_dict": "ckpt",
        "lora_dict": "lora",
        "refn_dict": "refn"
    }

    for arg_name, arg_val in {"vae_dict": vae_dict, "ckpt_dict": ckpt_dict, "lora_dict": lora_dict, "refn_dict": refn_dict}.items():
        if arg_val is None:
            continue

        dict_name = dict_mapping[arg_name]

        for tuple_idx, tuple_item in enumerate(loaded_objects[dict_name].copy()):
            if arg_name == "lora_dict":
                # Iterate over the tuples (lora_params, ckpt_name) in arg_val
                for lora_params, ckpt_name in arg_val:
                    # Compare lists of tuples considering order inside tuples, but not order of tuples
                    if set(lora_params) == set(tuple_item[0]) and ckpt_name == tuple_item[1]:
                        break
                else:  # If no match was found in lora_dict, remove the tuple from loaded_objects
                    if node_id in tuple_item[-1]:
                        tuple_item[-1].remove(node_id)
                        if not tuple_item[-1]:
                            loaded_objects[dict_name].remove(tuple_item)
                    continue
            elif tuple_item[0] not in arg_val:  # Only remove the tuple if it's not in arg_val
                if node_id in tuple_item[-1]:
                    tuple_item[-1].remove(node_id)
                    if not tuple_item[-1]:
                        loaded_objects[dict_name].remove(tuple_item)

# Retrieve the cache number from 'node_settings' json file
def get_cache_numbers(node_name):
    # Get the directory path of the current file
    my_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the file path for node_settings.json
    settings_file = os.path.join(my_dir, 'node_settings.json')
    # Load the settings from the JSON file
    with open(settings_file, 'r') as file:
        node_settings = json.load(file)
    # Retrieve the cache numbers for the given node
    model_cache_settings = node_settings.get(node_name, {}).get('model_cache', {})
    vae_cache = int(model_cache_settings.get('vae', 1))
    ckpt_cache = int(model_cache_settings.get('ckpt', 1))
    lora_cache = int(model_cache_settings.get('lora', 1))
    refn_cache = int(model_cache_settings.get('ckpt', 1))
    return vae_cache, ckpt_cache, lora_cache, refn_cache,

def print_last_helds(id=None):
    print("\n" + "-" * 40)  # Print an empty line followed by a separator line
    if id is not None:
        id = str(id)  # Convert ID to string
        print(f"Node-specific Last Helds (node_id:{int(id)})")
    else:
        print(f"Global Last Helds:")
    for key in ["preview_images", "latent", "output_images", "vae_decode"]:
        entries_with_id = last_helds[key] if id is None else [entry for entry in last_helds[key] if id == entry[-1]]
        if not entries_with_id:  # If no entries with the chosen ID, print None and skip this key
            continue
        print(f"{key.capitalize()}:")
        for i, entry in enumerate(entries_with_id, 1):  # Start numbering from 1
            if isinstance(entry[0], bool):  # Special handling for boolean types
                output = entry[0]
            else:
                output = len(entry[0])
            if id is None:
                print(f"  [{i}] Output: {output} (id: {entry[-1]})")
            else:
                print(f"  [{i}] Output: {output}")
    print("-" * 40)  # Print a separator line
    print("\n")  # Print an empty line

# For suppressing print outputs from functions
@contextmanager
def suppress_output():
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

# Set global preview_method
def set_preview_method(method):
    if method == 'auto' or method == 'LatentPreviewMethod.Auto':
        args.preview_method = latent_preview.LatentPreviewMethod.Auto
    elif method == 'latent2rgb' or method == 'LatentPreviewMethod.Latent2RGB':
        args.preview_method = latent_preview.LatentPreviewMethod.Latent2RGB
    elif method == 'taesd' or method == 'LatentPreviewMethod.TAESD':
        args.preview_method = latent_preview.LatentPreviewMethod.TAESD
    else:
        args.preview_method = latent_preview.LatentPreviewMethod.NoPreviews

# Extract global preview_method
def global_preview_method():
    return args.preview_method

# -----------------------------------------------------------------------------------------------------------------------
# Delete efficiency nodes web extensions from 'ComfyUI\web\extensions'.
# Pull https://github.com/comfyanonymous/ComfyUI/pull/1273 now allows defining web extensions through a dir path in init
import shutil

# Destination directory
destination_dir = os.path.join(comfy_dir, 'web', 'extensions', 'efficiency-nodes-comfyui')

# Check if the directory exists and delete it
if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)

# -----------------------------------------------------------------------------------------------------------------------
# Other
class XY_Capsule:
    def pre_define_model(self, model, clip, vae):
        return model, clip, vae

    def set_result(self, image, latent):
        pass

    def get_result(self, model, clip, vae):
        return None

    def set_x_capsule(self, capsule):
        return None

    def getLabel(self):
        return "Unknown"
