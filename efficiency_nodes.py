# Efficiency Nodes - A collection of my ComfyUI custom nodes to help streamline workflows and reduce total node count.
# by Luciano Cirino (Discord: TSC#9184) - April 2023 - October 2023
# https://github.com/LucianoCirino/efficiency-nodes-comfyui

from torch import Tensor
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch

import ast
from pathlib import Path
from importlib import import_module
import os
import sys
import copy
import subprocess
import json
import psutil

from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
from comfy_extras.nodes_gits import GITSScheduler

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.abspath(os.path.join(my_dir, '..'))
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Construct the path to the font file
font_path = os.path.join(my_dir, 'arial.ttf')

# Append comfy_dir to sys.path & import files
sys.path.append(comfy_dir)
from nodes import LatentUpscaleBy, KSampler, KSamplerAdvanced, VAEDecode, VAEDecodeTiled, VAEEncode, VAEEncodeTiled, \
    ImageScaleBy, CLIPSetLastLayer, CLIPTextEncode, ControlNetLoader, ControlNetApply, ControlNetApplyAdvanced, \
    PreviewImage, MAX_RESOLUTION
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.latent_formats
sys.path.remove(comfy_dir)

# Append my_dir to sys.path & import files
sys.path.append(my_dir)
from tsc_utils import *
from .py import smZ_cfg_denoiser
from .py import smZ_rng_source
from .py import cg_mixed_seed_noise
from .py import city96_latent_upscaler
from .py import ttl_nn_latent_upscaler
from .py import bnk_tiled_samplers
from .py import bnk_adv_encode
sys.path.remove(my_dir)

from comfy import samplers
# Append custom_nodes_dir to sys.path
sys.path.append(custom_nodes_dir)

# GLOBALS
REFINER_CFG_OFFSET = 0 #Refiner CFG Offset

# Monkey patch schedulers
SCHEDULER_NAMES = samplers.SCHEDULER_NAMES + ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS"]
SCHEDULERS = samplers.KSampler.SCHEDULERS + ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS"]

########################################################################################################################
# Common function for encoding prompts
def encode_prompts(positive_prompt, negative_prompt, token_normalization, weight_interpretation, clip, clip_skip,
                   refiner_clip, refiner_clip_skip, ascore, is_sdxl, empty_latent_width, empty_latent_height,
                   return_type="both"):

    positive_encoded = negative_encoded = refiner_positive_encoded = refiner_negative_encoded = None

    # Process base encodings if needed
    if return_type in ["base", "both"]:
        clip = CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]

        positive_encoded = bnk_adv_encode.AdvancedCLIPTextEncode().encode(clip, positive_prompt, token_normalization, weight_interpretation)[0]
        negative_encoded = bnk_adv_encode.AdvancedCLIPTextEncode().encode(clip, negative_prompt, token_normalization, weight_interpretation)[0]

    # Process refiner encodings if needed
    if return_type in ["refiner", "both"] and is_sdxl and refiner_clip and refiner_clip_skip and ascore:
        refiner_clip = CLIPSetLastLayer().set_last_layer(refiner_clip, refiner_clip_skip)[0]

        refiner_positive_encoded = bnk_adv_encode.AdvancedCLIPTextEncode().encode(refiner_clip, positive_prompt, token_normalization, weight_interpretation)[0]
        refiner_positive_encoded = bnk_adv_encode.AddCLIPSDXLRParams().encode(refiner_positive_encoded, empty_latent_width, empty_latent_height, ascore[0])[0]

        refiner_negative_encoded = bnk_adv_encode.AdvancedCLIPTextEncode().encode(refiner_clip, negative_prompt, token_normalization, weight_interpretation)[0]
        refiner_negative_encoded = bnk_adv_encode.AddCLIPSDXLRParams().encode(refiner_negative_encoded, empty_latent_width, empty_latent_height, ascore[1])[0]

    # Return results based on return_type
    if return_type == "base":
        return positive_encoded, negative_encoded, clip
    elif return_type == "refiner":
        return refiner_positive_encoded, refiner_negative_encoded, refiner_clip
    elif return_type == "both":
        return positive_encoded, negative_encoded, clip, refiner_positive_encoded, refiner_negative_encoded, refiner_clip

########################################################################################################################
# TSC Efficient Loader
class TSC_EfficientLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                              "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                              "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                              "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                              "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "positive": ("STRING", {"default": "CLIP_POSITIVE","multiline": True}),
                              "negative": ("STRING", {"default": "CLIP_NEGATIVE", "multiline": True}),
                              "token_normalization": (["none", "mean", "length", "length+mean"],),
                              "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                              "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                              "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 262144})},
                "optional": {"lora_stack": ("LORA_STACK", ),
                             "cnet_stack": ("CONTROL_NET_STACK",)},
                "hidden": { "prompt": "PROMPT",
                            "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "DEPENDENCIES",)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "CLIP", "DEPENDENCIES", )
    FUNCTION = "efficientloader"
    CATEGORY = "Efficiency Nodes/Loaders"

    def efficientloader(self, ckpt_name, vae_name, clip_skip, lora_name, lora_model_strength, lora_clip_strength,
                        positive, negative, token_normalization, weight_interpretation, empty_latent_width,
                        empty_latent_height, batch_size, lora_stack=None, cnet_stack=None, refiner_name="None",
                        ascore=None, prompt=None, my_unique_id=None, loader_type="regular"):

        # Clean globally stored objects
        globals_cleanup(prompt)

        # Create Empty Latent
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8]).cpu()

        # Retrieve cache numbers
        vae_cache, ckpt_cache, lora_cache, refn_cache = get_cache_numbers("Efficient Loader")

        if lora_name != "None" or lora_stack:
            # Initialize an empty list to store LoRa parameters.
            lora_params = []

            # Check if lora_name is not the string "None" and if so, add its parameters.
            if lora_name != "None":
                lora_params.append((lora_name, lora_model_strength, lora_clip_strength))

            # If lora_stack is not None or an empty list, extend lora_params with its items.
            if lora_stack:
                lora_params.extend(lora_stack)

            # Load LoRA(s)
            model, clip = load_lora(lora_params, ckpt_name, my_unique_id, cache=lora_cache, ckpt_cache=ckpt_cache,
                                    cache_overwrite=True)

            if vae_name == "Baked VAE":
                vae = get_bvae_by_ckpt_name(ckpt_name)
                if vae is None:
                    print(
                        f"{warning('Efficiency Nodes:')} Baked VAE not found in cache, loading checkpoint to extract VAE...")
                    _, _, vae = load_checkpoint(ckpt_name, my_unique_id, output_vae=True, cache=ckpt_cache,
                                                cache_overwrite=True)
        else:
            model, clip, vae = load_checkpoint(ckpt_name, my_unique_id, cache=ckpt_cache, cache_overwrite=True)
            lora_params = None

        # Load Refiner Checkpoint if given
        if refiner_name != "None":
            refiner_model, refiner_clip, _ = load_checkpoint(refiner_name, my_unique_id, output_vae=False,
                                                             cache=refn_cache, cache_overwrite=True, ckpt_type="refn")
        else:
            refiner_model = refiner_clip = None

        # Extract clip_skips
        refiner_clip_skip = clip_skip[1] if loader_type == "sdxl" else None
        clip_skip = clip_skip[0] if loader_type == "sdxl" else clip_skip

        # Encode prompt based on loader_type
        positive_encoded, negative_encoded, clip, refiner_positive_encoded, refiner_negative_encoded, refiner_clip = \
            encode_prompts(positive, negative, token_normalization, weight_interpretation, clip, clip_skip,
                           refiner_clip, refiner_clip_skip, ascore, loader_type == "sdxl",
                           empty_latent_width, empty_latent_height)

        # Apply ControlNet Stack if given
        if cnet_stack:
            controlnet_conditioning = TSC_Apply_ControlNet_Stack().apply_cnet_stack(positive_encoded, negative_encoded, cnet_stack)
            positive_encoded, negative_encoded = controlnet_conditioning[0], controlnet_conditioning[1]

        # Check for custom VAE
        if vae_name != "Baked VAE":
            vae = load_vae(vae_name, my_unique_id, cache=vae_cache, cache_overwrite=True)

        # Data for XY Plot
        dependencies = (vae_name, ckpt_name, clip, clip_skip, refiner_name, refiner_clip, refiner_clip_skip,
                        positive, negative, token_normalization, weight_interpretation, ascore,
                        empty_latent_width, empty_latent_height, lora_params, cnet_stack)

        ### Debugging
        ###print_loaded_objects_entries()
        print_loaded_objects_entries(my_unique_id, prompt)

        if loader_type == "regular":
            return (model, positive_encoded, negative_encoded, {"samples":latent}, vae, clip, dependencies,)
        elif loader_type == "sdxl":
            return ((model, clip, positive_encoded, negative_encoded, refiner_model, refiner_clip,
                     refiner_positive_encoded, refiner_negative_encoded), {"samples":latent}, vae, dependencies,)

#=======================================================================================================================
# TSC Efficient Loader SDXL
class TSC_EfficientLoaderSDXL(TSC_EfficientLoader):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "base_ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                              "base_clip_skip": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1}),
                              "refiner_ckpt_name": (["None"] + folder_paths.get_filename_list("checkpoints"),),
                              "refiner_clip_skip": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1}),
                              "positive_ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                              "negative_ascore": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                              "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                              "positive": ("STRING", {"default": "CLIP_POSITIVE", "multiline": True}),
                              "negative": ("STRING", {"default": "CLIP_NEGATIVE", "multiline": True}),
                              "token_normalization": (["none", "mean", "length", "length+mean"],),
                              "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                              "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                              "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})},
                "optional": {"lora_stack": ("LORA_STACK", ), "cnet_stack": ("CONTROL_NET_STACK",),},
                "hidden": { "prompt": "PROMPT", "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("SDXL_TUPLE", "LATENT", "VAE", "DEPENDENCIES",)
    RETURN_NAMES = ("SDXL_TUPLE", "LATENT", "VAE", "DEPENDENCIES", )
    FUNCTION = "efficientloaderSDXL"
    CATEGORY = "Efficiency Nodes/Loaders"

    def efficientloaderSDXL(self, base_ckpt_name, base_clip_skip, refiner_ckpt_name, refiner_clip_skip, positive_ascore,
                            negative_ascore, vae_name, positive, negative, token_normalization, weight_interpretation,
                            empty_latent_width, empty_latent_height, batch_size, lora_stack=None, cnet_stack=None,
                            prompt=None, my_unique_id=None):
        clip_skip = (base_clip_skip, refiner_clip_skip)
        lora_name = "None"
        lora_model_strength = lora_clip_strength = 0
        return super().efficientloader(base_ckpt_name, vae_name, clip_skip, lora_name, lora_model_strength, lora_clip_strength,
                        positive, negative, token_normalization, weight_interpretation, empty_latent_width, empty_latent_height,
                        batch_size, lora_stack=lora_stack, cnet_stack=cnet_stack, refiner_name=refiner_ckpt_name,
                        ascore=(positive_ascore, negative_ascore), prompt=prompt, my_unique_id=my_unique_id, loader_type="sdxl")

#=======================================================================================================================
# TSC Unpack SDXL Tuple
class TSC_Unpack_SDXL_Tuple:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sdxl_tuple": ("SDXL_TUPLE",)},}

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING","CONDITIONING", "MODEL", "CLIP", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("BASE_MODEL", "BASE_CLIP", "BASE_CONDITIONING+", "BASE_CONDITIONING-",
                    "REFINER_MODEL", "REFINER_CLIP","REFINER_CONDITIONING+","REFINER_CONDITIONING-",)
    FUNCTION = "unpack_sdxl_tuple"
    CATEGORY = "Efficiency Nodes/Misc"

    def unpack_sdxl_tuple(self, sdxl_tuple):
        return (sdxl_tuple[0], sdxl_tuple[1],sdxl_tuple[2],sdxl_tuple[3],
                sdxl_tuple[4],sdxl_tuple[5],sdxl_tuple[6],sdxl_tuple[7],)

# =======================================================================================================================
# TSC Pack SDXL Tuple
class TSC_Pack_SDXL_Tuple:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base_model": ("MODEL",),
                             "base_clip": ("CLIP",),
                             "base_positive": ("CONDITIONING",),
                             "base_negative": ("CONDITIONING",),
                             "refiner_model": ("MODEL",),
                             "refiner_clip": ("CLIP",),
                             "refiner_positive": ("CONDITIONING",),
                             "refiner_negative": ("CONDITIONING",),},}

    RETURN_TYPES = ("SDXL_TUPLE",)
    RETURN_NAMES = ("SDXL_TUPLE",)
    FUNCTION = "pack_sdxl_tuple"
    CATEGORY = "Efficiency Nodes/Misc"

    def pack_sdxl_tuple(self, base_model, base_clip, base_positive, base_negative,
                        refiner_model, refiner_clip, refiner_positive, refiner_negative):
        return ((base_model, base_clip, base_positive, base_negative,
                 refiner_model, refiner_clip, refiner_positive, refiner_negative),)

########################################################################################################################
# TSC LoRA Stacker
class TSC_LoRA_Stacker:
    modes = ["simple", "advanced"]

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "input_mode": (cls.modes,),
                "lora_count": ("INT", {"default": 3, "min": 0, "max": 50, "step": 1}),
            }
        }

        for i in range(1, 51):
            inputs["required"][f"lora_name_{i}"] = (loras,)
            inputs["required"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"model_str_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"clip_str_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        inputs["optional"] = {
            "lora_stack": ("LORA_STACK",)
        }
        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "lora_stacker"
    CATEGORY = "Efficiency Nodes/Stackers"

    def lora_stacker(self, input_mode, lora_count, lora_stack=None, **kwargs):

        # Extract values from kwargs
        loras = [kwargs.get(f"lora_name_{i}") for i in range(1, lora_count + 1)]

        # Create a list of tuples using provided parameters, exclude tuples with lora_name as "None"
        if input_mode == "simple":
            weights = [kwargs.get(f"lora_wt_{i}") for i in range(1, lora_count + 1)]
            loras = [(lora_name, lora_weight, lora_weight) for lora_name, lora_weight in zip(loras, weights) if
                     lora_name != "None"]
        else:
            model_strs = [kwargs.get(f"model_str_{i}") for i in range(1, lora_count + 1)]
            clip_strs = [kwargs.get(f"clip_str_{i}") for i in range(1, lora_count + 1)]
            loras = [(lora_name, model_str, clip_str) for lora_name, model_str, clip_str in
                     zip(loras, model_strs, clip_strs) if lora_name != "None"]

        # If lora_stack is not None, extend the loras list with lora_stack
        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        return (loras,)

#=======================================================================================================================
# TSC Control Net Stacker
class TSC_Control_Net_Stacker:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"control_net": ("CONTROL_NET",),
                             "image": ("IMAGE",),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})},
                "optional": {"cnet_stack": ("CONTROL_NET_STACK",)},
                }

    RETURN_TYPES = ("CONTROL_NET_STACK",)
    RETURN_NAMES = ("CNET_STACK",)
    FUNCTION = "control_net_stacker"
    CATEGORY = "Efficiency Nodes/Stackers"

    def control_net_stacker(self, control_net, image, strength, start_percent, end_percent, cnet_stack=None):
        # If control_net_stack is None, initialize as an empty list
        cnet_stack = [] if cnet_stack is None else cnet_stack

        # Extend the control_net_stack with the new tuple
        cnet_stack.extend([(control_net, image, strength, start_percent, end_percent)])

        return (cnet_stack,)

#=======================================================================================================================
# TSC Apply ControlNet Stack
class TSC_Apply_ControlNet_Stack:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",),
                             "negative": ("CONDITIONING",)},
                "optional": {"cnet_stack": ("CONTROL_NET_STACK",)}
                }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("CONDITIONING+","CONDITIONING-",)
    FUNCTION = "apply_cnet_stack"
    CATEGORY = "Efficiency Nodes/Stackers"

    def apply_cnet_stack(self, positive, negative, cnet_stack=None):
        if cnet_stack is None:
            return (positive, negative)

        for control_net_tuple in cnet_stack:
            control_net, image, strength, start_percent, end_percent = control_net_tuple
            controlnet_conditioning = ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image,
                                                                                 strength, start_percent, end_percent)
            positive, negative = controlnet_conditioning[0], controlnet_conditioning[1]

        return (positive, negative, )


########################################################################################################################
# TSC KSampler (Efficient)
class TSC_KSampler:
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"model": ("MODEL",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "preview_method": (["auto", "latent2rgb", "taesd", "vae_decoded_only", "none"],),
                     "vae_decode": (["true", "true (tiled)", "false"],),
                     },
                "optional": { "optional_vae": ("VAE",),
                              "script": ("SCRIPT",),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE", )
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "IMAGE", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "Efficiency Nodes/Sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
               preview_method, vae_decode, denoise=1.0, prompt=None, extra_pnginfo=None, my_unique_id=None,
               optional_vae=(None,), script=None, add_noise=None, start_at_step=None, end_at_step=None,
               return_with_leftover_noise=None, sampler_type="regular"):
        
        # Rename the vae variable
        vae = optional_vae

        # If vae is not connected, disable vae decoding
        if vae == (None,) and vae_decode != "false":
            print(f"{warning('KSampler(Efficient) Warning:')} No vae input detected, proceeding as if vae_decode was false.\n")
            vae_decode = "false"

        #---------------------------------------------------------------------------------------------------------------
        # Unpack SDXL Tuple embedded in the 'model' channel
        if sampler_type == "sdxl":
            sdxl_tuple = model
            model, _, positive, negative, refiner_model, _, refiner_positive, refiner_negative = sdxl_tuple
        else:
            refiner_model = refiner_positive = refiner_negative = None

        #---------------------------------------------------------------------------------------------------------------
        def keys_exist_in_script(*keys):
            return any(key in script for key in keys) if script else False

        #---------------------------------------------------------------------------------------------------------------
        def vae_decode_latent(vae, samples, vae_decode):
            return VAEDecodeTiled().decode(vae,samples,320)[0] if "tiled" in vae_decode else VAEDecode().decode(vae,samples)[0]

        def vae_encode_image(vae, pixels, vae_decode):
            return VAEEncodeTiled().encode(vae,pixels,320)[0] if "tiled" in vae_decode else VAEEncode().encode(vae,pixels)[0]

        # ---------------------------------------------------------------------------------------------------------------
        def process_latent_image(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                denoise, sampler_type, add_noise, start_at_step, end_at_step, return_with_leftover_noise,
                                refiner_model, refiner_positive, refiner_negative, vae, vae_decode, preview_method):

            # Store originals
            original_calculation = comfy.samplers.calculate_sigmas
            original_KSampler_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS
            previous_preview_method = global_preview_method()
            original_prepare_noise = comfy.sample.prepare_noise
            original_KSampler = comfy.samplers.KSampler
            original_model_str = str(model)

            # monkey patch the sample function
            def calculate_sigmas(model_sampling, scheduler_name: str, steps):
                if scheduler_name.startswith("AYS"):
                    return AlignYourStepsScheduler().get_sigmas(scheduler_name.split(" ")[1], steps, denoise=1.0)[0]
                elif scheduler_name == "GITS":
                    return GITSScheduler().get_sigmas(1.20, steps, denoise=1.0)[0]
                return original_calculation(model_sampling, scheduler_name, steps)

            comfy.samplers.KSampler.SCHEDULERS = SCHEDULERS
            comfy.samplers.calculate_sigmas = calculate_sigmas

            # Initialize output variables
            samples = images = gifs = preview = cnet_imgs = None

            try:
                # Change the global preview method (temporarily)
                set_preview_method(preview_method)

                # ------------------------------------------------------------------------------------------------------
                # Check if "noise" exists in the script before main sampling has taken place
                if keys_exist_in_script("noise"):
                    rng_source, cfg_denoiser, add_seed_noise, m_seed, m_weight = script["noise"]
                    smZ_rng_source.rng_rand_source(rng_source) # this function monkey patches comfy.sample.prepare_noise
                    if cfg_denoiser:
                        comfy.samplers.KSampler = smZ_cfg_denoiser.SDKSampler
                    if add_seed_noise:
                        comfy.sample.prepare_noise = cg_mixed_seed_noise.get_mixed_noise_function(comfy.sample.prepare_noise, m_seed, m_weight)
                    else:
                        m_seed = m_weight = None
                else:
                    rng_source = cfg_denoiser = add_seed_noise = m_seed = m_weight = None

                # ------------------------------------------------------------------------------------------------------
                # Check if "anim" exists in the script before main sampling has taken place
                if keys_exist_in_script("anim"):
                    if preview_method != "none":
                        set_preview_method("none")  # disable preview method
                        print(f"{warning('KSampler(Efficient) Warning:')} Live preview disabled for animatediff generations.")
                    motion_model, beta_schedule, context_options, frame_rate, loop_count, format, pingpong, save_image = script["anim"]
                    model = AnimateDiffLoaderWithContext().load_mm_and_inject_params(model, motion_model, beta_schedule, context_options)[0]

                # ------------------------------------------------------------------------------------------------------
                # Store run parameters as strings. Load previous stored samples if all parameters match.
                latent_image_hash = tensor_to_hash(latent_image["samples"])
                positive_hash = tensor_to_hash(positive[0][0])
                negative_hash = tensor_to_hash(negative[0][0])
                refiner_positive_hash = tensor_to_hash(refiner_positive[0][0]) if refiner_positive is not None else None
                refiner_negative_hash = tensor_to_hash(refiner_negative[0][0]) if refiner_negative is not None else None

                # Include motion_model, beta_schedule, and context_options as unique identifiers if they exist.
                model_identifier = [original_model_str, motion_model, beta_schedule, context_options] if keys_exist_in_script("anim")\
                                    else [original_model_str]

                parameters = [model_identifier] + [seed, steps, cfg, sampler_name, scheduler, positive_hash, negative_hash,
                                                  latent_image_hash, denoise, sampler_type, add_noise, start_at_step,
                                                  end_at_step, return_with_leftover_noise, refiner_model, refiner_positive_hash,
                                                  refiner_negative_hash, rng_source, cfg_denoiser, add_seed_noise, m_seed, m_weight]

                # Convert all elements in parameters to strings, except for the hash variable checks
                parameters = [str(item) if not isinstance(item, type(latent_image_hash)) else item for item in parameters]

                # Load previous latent if all parameters match, else returns 'None'
                samples = load_ksampler_results("latent", my_unique_id, parameters)

                if samples is None: # clear stored images
                    store_ksampler_results("image", my_unique_id, None)
                    store_ksampler_results("cnet_img", my_unique_id, None)

                if samples is not None: # do not re-sample
                    images = load_ksampler_results("image", my_unique_id)
                    cnet_imgs = True # "True" will denote that it can be loaded provided the preprocessor matches

                # Sample the latent_image(s) using the Comfy KSampler nodes
                elif sampler_type == "regular":
                    samples = KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                                        latent_image, denoise=denoise)[0] if denoise>0 else latent_image

                elif sampler_type == "advanced":
                    samples = KSamplerAdvanced().sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                                        positive, negative, latent_image, start_at_step, end_at_step,
                                                        return_with_leftover_noise, denoise=1.0)[0]

                elif sampler_type == "sdxl":
                    # Disable refiner if refine_at_step is -1
                    if end_at_step == -1:
                        end_at_step = steps

                    # Perform base model sampling
                    add_noise = return_with_leftover_noise = "enable"
                    samples = KSamplerAdvanced().sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                                        positive, negative, latent_image, start_at_step, end_at_step,
                                                        return_with_leftover_noise, denoise=1.0)[0]

                    # Perform refiner model sampling
                    if refiner_model and end_at_step < steps:
                        add_noise = return_with_leftover_noise = "disable"
                        samples = KSamplerAdvanced().sample(refiner_model, add_noise, seed, steps, cfg + REFINER_CFG_OFFSET,
                                                            sampler_name, scheduler, refiner_positive, refiner_negative,
                                                            samples, end_at_step, steps,
                                                            return_with_leftover_noise, denoise=1.0)[0]

                # Cache the first pass samples in the 'last_helds' dictionary "latent" if not xyplot
                if not any(keys_exist_in_script(key) for key in ["xyplot"]):
                    store_ksampler_results("latent", my_unique_id, samples, parameters)

                # ------------------------------------------------------------------------------------------------------
                # Check if "hiresfix" exists in the script after main sampling has taken place
                if keys_exist_in_script("hiresfix"):
                    # Unpack the tuple from the script's "hiresfix" key
                    upscale_type, latent_upscaler, upscale_by, use_same_seed, hires_seed, hires_steps, hires_denoise,\
                        iterations, hires_control_net, hires_cnet_strength, preprocessor, preprocessor_imgs, \
                        latent_upscale_function, latent_upscale_model, pixel_upscale_model = script["hiresfix"]

                    # Define hires_seed
                    hires_seed = seed if use_same_seed else hires_seed

                    # Define latent_upscale_model
                    if latent_upscale_model is None:
                        latent_upscale_model = model
                    elif keys_exist_in_script("anim"):
                            latent_upscale_model = \
                            AnimateDiffLoaderWithContext().load_mm_and_inject_params(latent_upscale_model, motion_model,
                                                                                     beta_schedule, context_options)[0]

                    # Generate Preprocessor images and Apply Control Net
                    if hires_control_net is not None:
                        # Attempt to load previous "cnet_imgs" if previous images were loaded and preprocessor is same
                        if cnet_imgs is True:
                            cnet_imgs = load_ksampler_results("cnet_img", my_unique_id, [preprocessor])
                        # If cnet_imgs is None, generate new ones
                        if cnet_imgs is None:
                            if images is None:
                                images = vae_decode_latent(vae, samples, vae_decode)
                                store_ksampler_results("image", my_unique_id, images)
                            cnet_imgs = AIO_Preprocessor().execute(preprocessor, images)[0]
                            store_ksampler_results("cnet_img", my_unique_id, cnet_imgs, [preprocessor])
                        positive = ControlNetApply().apply_controlnet(positive, hires_control_net, cnet_imgs, hires_cnet_strength)[0]
                        
                    # Iterate for the given number of iterations
                    if upscale_type == "latent":
                        for _ in range(iterations):
                            upscaled_latent_image = latent_upscale_function().upscale(samples, latent_upscaler, upscale_by)[0]
                            samples = KSampler().sample(latent_upscale_model, hires_seed, hires_steps, cfg, sampler_name, scheduler,
                                                            positive, negative, upscaled_latent_image, denoise=hires_denoise)[0]
                            images = None # set to None when samples is updated
                    elif upscale_type == "pixel":
                        if images is None:
                            images = vae_decode_latent(vae, samples, vae_decode)
                            store_ksampler_results("image", my_unique_id, images)
                        images = ImageUpscaleWithModel().upscale(pixel_upscale_model, images)[0]
                        images = ImageScaleBy().upscale(images, "nearest-exact", upscale_by/4)[0]
                    elif upscale_type == "both":
                        for _ in range(iterations):
                            if images is None:
                                images = vae_decode_latent(vae, samples, vae_decode)
                                store_ksampler_results("image", my_unique_id, images)
                            images = ImageUpscaleWithModel().upscale(pixel_upscale_model, images)[0]
                            images = ImageScaleBy().upscale(images, "nearest-exact", upscale_by/4)[0]

                            samples = vae_encode_image(vae, images, vae_decode)
                            upscaled_latent_image = latent_upscale_function().upscale(samples, latent_upscaler, 1)[0]
                            samples = KSampler().sample(latent_upscale_model, hires_seed, hires_steps, cfg, sampler_name, scheduler,
                                                                positive, negative, upscaled_latent_image, denoise=hires_denoise)[0]
                            images = None # set to None when samples is updated

                # ------------------------------------------------------------------------------------------------------
                # Check if "tile" exists in the script after main sampling has taken place
                if keys_exist_in_script("tile"):
                    # Unpack the tuple from the script's "tile" key
                    upscale_by, tile_size, tiling_strategy, tiling_steps, tile_seed, tiled_denoise,\
                        tile_controlnet, strength = script["tile"]

                    # Decode image, store if first decode
                    if images is None:
                        images = vae_decode_latent(vae, samples, vae_decode)
                        if not any(keys_exist_in_script(key) for key in ["xyplot", "hiresfix"]):
                            store_ksampler_results("image", my_unique_id, images)

                    # Upscale image
                    upscaled_image = ImageScaleBy().upscale(images, "nearest-exact", upscale_by)[0]
                    upscaled_latent = vae_encode_image(vae, upscaled_image, vae_decode)

                    # If using Control Net, Apply Control Net using upscaled_image and loaded control_net
                    if tile_controlnet is not None:
                        positive = ControlNetApply().apply_controlnet(positive, tile_controlnet, upscaled_image, 1)[0]

                    # Sample latent
                    TSampler = bnk_tiled_samplers.TiledKSampler
                    samples = TSampler().sample(model, tile_seed, tile_size, tile_size, tiling_strategy, tiling_steps, cfg,
                                                sampler_name, scheduler, positive, negative, upscaled_latent,
                                                denoise=tiled_denoise)[0]
                    images = None  # set to None when samples is updated

                # ------------------------------------------------------------------------------------------------------
                # Check if "anim" exists in the script after the main sampling has taken place
                if keys_exist_in_script("anim"):
                    if images is None:
                        images = vae_decode_latent(vae, samples, vae_decode)
                        if not any(keys_exist_in_script(key) for key in ["xyplot", "hiresfix", "tile"]):
                            store_ksampler_results("image", my_unique_id, images)
                    gifs = AnimateDiffCombine().generate_gif(images, frame_rate, loop_count, format=format,
                    pingpong=pingpong, save_image=save_image, prompt=prompt, extra_pnginfo=extra_pnginfo)["ui"]["gifs"]

                # ------------------------------------------------------------------------------------------------------

                # Decode image if not yet decoded
                if "true" in vae_decode:
                    if images is None:
                        images = vae_decode_latent(vae, samples, vae_decode)
                        # Store decoded image as base image of no script is detected
                        if all(not keys_exist_in_script(key) for key in ["xyplot", "hiresfix", "tile", "anim"]):
                            store_ksampler_results("image", my_unique_id, images)

                # Append Control Net Images (if exist)
                if cnet_imgs is not None and not True:
                    if preprocessor_imgs and upscale_type == "latent":
                        if keys_exist_in_script("xyplot"):
                            print(
                                f"{warning('HighRes-Fix Warning:')} Preprocessor images auto-disabled when XY Plotting.")
                        else:
                            # Resize cnet_imgs if necessary and stack
                            if images.shape[1:3] != cnet_imgs.shape[1:3]:  # comparing height and width
                                cnet_imgs = quick_resize(cnet_imgs, images.shape)
                            images = torch.cat([images, cnet_imgs], dim=0)

                # Define preview images
                if keys_exist_in_script("anim"):
                    preview = {"gifs": gifs, "images": list()}
                elif preview_method == "none" or (preview_method == "vae_decoded_only" and vae_decode == "false"):
                    preview = {"images": list()}
                elif images is not None:
                    preview = PreviewImage().save_images(images, prompt=prompt, extra_pnginfo=extra_pnginfo)["ui"]

                # Define a dummy output image
                if images is None and vae_decode == "false":
                    images = TSC_KSampler.empty_image

            finally:
                # Restore global changes
                set_preview_method(previous_preview_method)
                comfy.samplers.KSampler = original_KSampler
                comfy.sample.prepare_noise = original_prepare_noise
                comfy.samplers.calculate_sigmas = original_calculation
                comfy.samplers.KSampler.SCHEDULERS = original_KSampler_SCHEDULERS

            return samples, images, gifs, preview

        # ---------------------------------------------------------------------------------------------------------------
        # Clean globally stored objects of non-existant nodes
        globals_cleanup(prompt)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # If not XY Plotting
        if not keys_exist_in_script("xyplot"):

            # Process latent image
            samples, images, gifs, preview = process_latent_image(model, seed, steps, cfg, sampler_name, scheduler,
                                            positive, negative, latent_image, denoise, sampler_type, add_noise,
                                            start_at_step, end_at_step, return_with_leftover_noise, refiner_model,
                                            refiner_positive, refiner_negative, vae, vae_decode, preview_method)

            if sampler_type == "sdxl":
                result = (sdxl_tuple, samples, vae, images,)
            else:
                result = (model, positive, negative, samples, vae, images,)
                
            if preview is None:
                return {"result": result}
            else:
                return {"ui": preview, "result": result}

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # If XY Plot
        elif keys_exist_in_script("xyplot"):

            # If no vae connected, throw errors
            if vae == (None,):
                print(f"{error('KSampler(Efficient) Error:')} VAE input must be connected in order to use the XY Plot script.")

                return {"ui": {"images": list()},
                        "result": (model, positive, negative, latent_image, vae, TSC_KSampler.empty_image,)}

            # If vae_decode is not set to true, print message that changing it to true
            if "true" not in vae_decode:
                print(f"{warning('KSampler(Efficient) Warning:')} VAE decoding must be set to \'true\'"
                    " for the XY Plot script, proceeding as if \'true\'.\n")

            #___________________________________________________________________________________________________________
            # Initialize, unpack, and clean variables for the XY Plot script
            vae_name = None
            ckpt_name = None
            clip = None
            clip_skip = None
            refiner_name = None
            refiner_clip = None
            refiner_clip_skip = None
            positive_prompt = None
            negative_prompt = None
            ascore = None
            empty_latent_width = None
            empty_latent_height = None
            lora_stack = None
            cnet_stack = None

            # Split the 'samples' tensor
            samples_tensors = torch.split(latent_image['samples'], 1, dim=0)

            # Check if 'noise_mask' exists and split if it does
            if 'noise_mask' in latent_image:
                noise_mask_tensors = torch.split(latent_image['noise_mask'], 1, dim=0)
                latent_tensors = [{'samples': img, 'noise_mask': mask} for img, mask in
                                  zip(samples_tensors, noise_mask_tensors)]
            else:
                latent_tensors = [{'samples': img} for img in samples_tensors]

            # Set latent only to the first of the batch
            latent_image = latent_tensors[0]

            # Unpack script Tuple (X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, dependencies)
            X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, cache_models, xyplot_as_output_image,\
                xyplot_id, dependencies = script["xyplot"]

            #_______________________________________________________________________________________________________
            # The below section is used to check wether the XY_type is allowed for the Ksampler instance being used.
            # If not the correct type, this section will abort the xy plot script.

            samplers = {
                "regular": {
                    "disallowed": ["AddNoise", "ReturnNoise", "StartStep", "EndStep", "RefineStep",
                                   "Refiner", "Refiner On/Off", "AScore+", "AScore-"],
                    "name": "KSampler (Efficient)"
                },
                "advanced": {
                    "disallowed": ["RefineStep", "Denoise", "RefineStep", "Refiner", "Refiner On/Off",
                                   "AScore+", "AScore-"],
                    "name": "KSampler Adv. (Efficient)"
                },
                "sdxl": {
                    "disallowed": ["AddNoise", "EndStep", "Denoise"],
                    "name": "KSampler SDXL (Eff.)"
                }
            }

            # Define disallowed XY_types for each ksampler type
            def get_ksampler_details(sampler_type):
                return samplers.get(sampler_type, {"disallowed": [], "name": ""})

            def suggest_ksampler(X_type, Y_type, current_sampler):
                for sampler, details in samplers.items():
                    if sampler != current_sampler and X_type not in details["disallowed"] and Y_type not in details["disallowed"]:
                        return details["name"]
                return "a different KSampler"

            # In your main function or code segment:
            details = get_ksampler_details(sampler_type)
            disallowed_XY_types = details["disallowed"]
            ksampler_name = details["name"]

            if X_type in disallowed_XY_types or Y_type in disallowed_XY_types:
                error_prefix = f"{error(f'{ksampler_name} Error:')}"

                failed_type = []
                if X_type in disallowed_XY_types:
                    failed_type.append(f"X_type: '{X_type}'")
                if Y_type in disallowed_XY_types:
                    failed_type.append(f"Y_type: '{Y_type}'")

                suggested_ksampler = suggest_ksampler(X_type, Y_type, sampler_type)

                print(f"{error_prefix} Invalid value for {' and '.join(failed_type)}. "
                    f"Use {suggested_ksampler} for this XY Plot type."
                    f"\nDisallowed XY_types for this KSampler are: {', '.join(disallowed_XY_types)}.")

                return {"ui": {"images": list()},
                    "result": (model, positive, negative, latent_image, vae, TSC_KSampler.empty_image,)}

            #_______________________________________________________________________________________________________
            # Unpack Effficient Loader dependencies
            if dependencies is not None:
                vae_name, ckpt_name, clip, clip_skip, refiner_name, refiner_clip, refiner_clip_skip,\
                    positive_prompt, negative_prompt, token_normalization, weight_interpretation, ascore,\
                    empty_latent_width, empty_latent_height, lora_stack, cnet_stack = dependencies

            #_______________________________________________________________________________________________________
            # Printout XY Plot values to be processed
            def process_xy_for_print(value, replacement, type_):

                if type_ == "Seeds++ Batch" and isinstance(value, list):
                    return [v + seed for v in value]  # Add seed to every entry in the list

                elif type_ == "Scheduler" and isinstance(value, tuple):
                    return value[0]  # Return only the first entry of the tuple

                elif type_ == "VAE" and isinstance(value, list):
                    # For each string in the list, extract the filename from the path
                    return [os.path.basename(v) for v in value]

                elif (type_ == "Checkpoint" or type_ == "Refiner") and isinstance(value, list):
                    # For each tuple in the list, return only the first value if the second or third value is None
                    return [(os.path.basename(v[0]),) + v[1:] if v[1] is None or v[2] is None
                            else (os.path.basename(v[0]), v[1]) if v[2] is None
                            else (os.path.basename(v[0]),) + v[1:] for v in value]

                elif (type_ == "LoRA" or type_ == "LoRA Stacks") and isinstance(value, list):
                # Return only the first Tuple of each inner array
                    return [[(os.path.basename(v[0][0]),) + v[0][1:], "..."] if len(v) > 1
                            else [(os.path.basename(v[0][0]),) + v[0][1:]] for v in value]

                elif type_ == "LoRA Batch" and isinstance(value, list):
                    # Extract the basename of the first value of the first tuple from each sublist
                    return [os.path.basename(v[0][0]) for v in value if v and isinstance(v[0], tuple) and v[0][0]]

                elif (type_ == "LoRA Wt" or type_ == "LoRA MStr") and isinstance(value, list):
                    # Extract the first value of the first tuple from each sublist
                    return [v[0][1] for v in value if v and isinstance(v[0], tuple)]

                elif type_ == "LoRA CStr" and isinstance(value, list):
                    # Extract the first value of the first tuple from each sublist
                    return [v[0][2] for v in value if v and isinstance(v[0], tuple)]

                elif type_ == "ControlNetStrength" and isinstance(value, list):
                    # Extract the third entry of the first tuple from each inner list
                    return [round(inner_list[0][2], 3) for inner_list in value]

                elif type_ == "ControlNetStart%" and isinstance(value, list):
                    # Extract the third entry of the first tuple from each inner list
                    return [round(inner_list[0][3], 3) for inner_list in value]

                elif type_ == "ControlNetEnd%" and isinstance(value, list):
                    # Extract the third entry of the first tuple from each inner list
                    return [round(inner_list[0][4], 3) for inner_list in value]

                elif isinstance(value, tuple):
                    return tuple(replacement if v is None else v for v in value)

                else:
                    return replacement if value is None else value

            # Determine the replacements based on X_type and Y_type
            replacement_X = scheduler if X_type == 'Sampler' else clip_skip if X_type == 'Checkpoint' else None
            replacement_Y = scheduler if Y_type == 'Sampler' else clip_skip if Y_type == 'Checkpoint' else None

            # Process X_value and Y_value
            X_value_processed = process_xy_for_print(X_value, replacement_X, X_type)
            Y_value_processed = process_xy_for_print(Y_value, replacement_Y, Y_type)

            print(info("-" * 40))
            print(info('XY Plot Script Inputs:'))
            print(info(f"(X) {X_type}:"))
            for item in X_value_processed:
                print(info(f"    {item}"))
            print(info(f"(Y) {Y_type}:"))
            for item in Y_value_processed:
                print(info(f"    {item}"))
            print(info("-" * 40))

            #_______________________________________________________________________________________________________
            # Perform various initializations in this section

            # If not caching models, set to 1.
            if cache_models == "False":
                vae_cache = ckpt_cache = lora_cache = refn_cache = 1
            else:
                # Retrieve cache numbers
                vae_cache, ckpt_cache, lora_cache, refn_cache = get_cache_numbers("XY Plot")
            # Pack cache numbers in a tuple
            cache = (vae_cache, ckpt_cache, lora_cache, refn_cache)

            # Add seed to every entry in the list
            X_value = [v + seed for v in X_value] if "Seeds++ Batch" == X_type else X_value
            Y_value = [v + seed for v in Y_value] if "Seeds++ Batch" == Y_type else Y_value

            # Embedd original prompts into prompt variables
            positive_prompt = (positive_prompt, positive_prompt)
            negative_prompt = (negative_prompt, negative_prompt)

            # Set lora_stack to None if one of types are LoRA
            if "LoRA" in X_type or "LoRA" in Y_type:
                lora_stack = None

            # Define the manipulated and static Control Net Variables with a tuple with shape (cn_1, cn_2, cn_3).
            # The information in this tuple will be used by the plotter to properly plot Control Net XY input types.
            cn_1, cn_2, cn_3 = None, None, None
            # If X_type has "ControlNet" or both X_type and Y_type have "ControlNet"
            if "ControlNet" in X_type:
                cn_1, cn_2, cn_3 = X_value[0][0][2], X_value[0][0][3], X_value[0][0][4]
            # If only Y_type has "ControlNet" and not X_type
            elif "ControlNet" in Y_type:
                cn_1, cn_2, cn_3 = Y_value[0][0][2], Y_value[0][0][3], Y_value[0][0][4]
            # Additional checks for other substrings
            if "ControlNetStrength" in X_type or "ControlNetStrength" in Y_type:
                cn_1 = None
            if "ControlNetStart%" in X_type or "ControlNetStart%" in Y_type:
                cn_2 = None
            if "ControlNetEnd%" in X_type or "ControlNetEnd%" in Y_type:
                cn_3 = None
            # Embed the information in cnet_stack
            cnet_stack = (cnet_stack, (cn_1, cn_2, cn_3))

            # Optimize image generation by prioritization:
            priority = [
                "Checkpoint",
                "Refiner",
                "LoRA",
                "LoRA Stacks",
                "VAE",
            ]
            conditioners = {
                "Positive Prompt S/R",
                "Negative Prompt S/R",
                "AScore+",
                "AScore-",
                "Clip Skip",
                "Clip Skip (Refiner)",
                "ControlNetStrength",
                "ControlNetStart%",
                "ControlNetEnd%"
            }
            # Get priority values; return a high number if the type is not in priority list
            x_priority = priority.index(X_type) if X_type in priority else 999
            y_priority = priority.index(Y_type) if Y_type in priority else 999

            # Check if both are conditioners
            are_both_conditioners = X_type in conditioners and Y_type in conditioners

            # Special cases
            is_special_case = (
                    (X_type == "Refiner On/Off" and Y_type in ["RefineStep", "Steps"]) or
                    (X_type == "Nothing" and Y_type != "Nothing") or
                    (Y_type == "LoRA Batch" and (X_type == "LoRA Wt" or X_type == "LoRA MStr" or X_type == "LoRA CStr"))
            )

            # Determine whether to flip
            flip_xy = (y_priority < x_priority and not are_both_conditioners) or is_special_case

            # Perform the flip if necessary
            if flip_xy:
                X_type, Y_type = Y_type, X_type
                X_value, Y_value = Y_value, X_value

            #_______________________________________________________________________________________________________
            # The below code will clean from the cache any ckpt/vae/lora models it will not be reusing.
            # Note: Special LoRA types will not trigger cache: "LoRA Batch", "LoRA Wt", "LoRA MStr", "LoRA CStr"

            # Map the type names to the dictionaries
            dict_map = {"VAE": [], "Checkpoint": [], "LoRA": [], "Refiner": []}

            # Create a list of tuples with types and values
            type_value_pairs = [(X_type, X_value.copy()), (Y_type, Y_value.copy())]
            
            # Replace "LoRA Stacks" with "LoRA"
            type_value_pairs = [('LoRA' if t == 'LoRA Stacks' else t, v) for t, v in type_value_pairs]
            
            # Iterate over type-value pairs
            for t, v in type_value_pairs:
                if t in dict_map:
                    # Flatten the list of lists of tuples if the type is "LoRA"
                    if t == "LoRA":
                        dict_map[t] = [item for sublist in v for item in sublist]
                    else:
                        dict_map[t] = v

            vae_dict = dict_map.get("VAE", [])

            # Construct ckpt_dict and also update vae_dict based on the third entry of the tuples in dict_map["Checkpoint"]
            if dict_map.get("Checkpoint", []):
                ckpt_dict = [t[0] for t in dict_map["Checkpoint"]]
                for t in dict_map["Checkpoint"]:
                    if t[2] is not None and t[2] != "Baked VAE":
                        vae_dict.append(t[2])
            else:
                ckpt_dict = []

            lora_dict = [[t,] for t in dict_map.get("LoRA", [])] if dict_map.get("LoRA", []) else []

            # Construct refn_dict
            if dict_map.get("Refiner", []):
                refn_dict = [t[0] for t in dict_map["Refiner"]]
            else:
                refn_dict = []

            # If both ckpt_dict and lora_dict are not empty, manipulate lora_dict as described
            if ckpt_dict and lora_dict:
                lora_dict = [(lora_stack, ckpt) for ckpt in ckpt_dict for lora_stack in lora_dict]
            # If lora_dict is not empty and ckpt_dict is empty, insert ckpt_name into each tuple in lora_dict
            elif lora_dict:
                lora_dict = [(lora_stack, ckpt_name) for lora_stack in lora_dict]

            # Avoid caching models accross both X and Y
            if X_type == "Checkpoint":
                lora_dict = []
                refn_dict = []
            elif X_type == "Refiner":
                ckpt_dict = []
                lora_dict = []
            elif X_type == "LoRA":
                ckpt_dict = []
                refn_dict = []

            ### Print dict_arrays for debugging
            ###print(f"vae_dict={vae_dict}\nckpt_dict={ckpt_dict}\nlora_dict={lora_dict}\nrefn_dict={refn_dict}")

            # Clean values that won't be reused
            clear_cache_by_exception(xyplot_id, vae_dict=vae_dict, ckpt_dict=ckpt_dict, lora_dict=lora_dict, refn_dict=refn_dict)

            ### Print loaded_objects for debugging
            ###print_loaded_objects_entries()

            #_______________________________________________________________________________________________________
            # Function that changes appropiate variables for next processed generations (also generates XY_labels)
            def define_variable(var_type, var, add_noise, seed, steps, start_at_step, end_at_step,
                                return_with_leftover_noise, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name,
                                clip_skip, refiner_name, refiner_clip_skip, positive_prompt, negative_prompt, ascore,
                                lora_stack, cnet_stack, var_label, num_label):

                # Define default max label size limit
                max_label_len = 42

                # If var_type is "AddNoise", update 'add_noise' with 'var', and generate text label
                if var_type == "AddNoise":
                    add_noise = var
                    text = f"AddNoise: {add_noise}"

                # If var_type is "Seeds++ Batch", generate text label
                elif var_type == "Seeds++ Batch":
                    seed = var
                    text = f"Seed: {seed}"

                # If var_type is "Steps", update 'steps' with 'var' and generate text label
                elif var_type == "Steps":
                    steps = var
                    text = f"Steps: {steps}"

                # If var_type is "StartStep", update 'start_at_step' with 'var' and generate text label
                elif var_type == "StartStep":
                    start_at_step = var
                    text = f"StartStep: {start_at_step}"

                # If var_type is "EndStep", update 'end_at_step' with 'var' and generate text label
                elif var_type == "EndStep":
                    end_at_step = var
                    text = f"EndStep: {end_at_step}"

                # If var_type is "RefineStep", update 'end_at_step' with 'var' and generate text label
                elif var_type == "RefineStep":
                    end_at_step = var
                    text = f"RefineStep: {end_at_step}"

                # If var_type is "ReturnNoise", update 'return_with_leftover_noise' with 'var', and generate text label
                elif var_type == "ReturnNoise":
                    return_with_leftover_noise = var
                    text = f"ReturnNoise: {return_with_leftover_noise}"

                # If var_type is "CFG Scale", update cfg with var and generate text label
                elif var_type == "CFG Scale":
                    cfg = var
                    text = f"CFG: {round(cfg,2)}"

                # If var_type is "Sampler", update sampler_name and scheduler with var, and generate text label
                elif var_type == "Sampler":
                    sampler_name = var[0]
                    if var[1] == "":
                        text = f"{sampler_name}"
                    else:
                        if var[1] != None:
                            scheduler = (var[1], scheduler[1])
                        else:
                            scheduler = (scheduler[1], scheduler[1])
                        text = f"{sampler_name} ({scheduler[0]})"
                    text = text.replace("ancestral", "a").replace("uniform", "u").replace("exponential","exp")

                # If var_type is "Scheduler", update scheduler and generate labels
                elif var_type == "Scheduler":
                    if len(var) == 2:
                        scheduler = (var[0], scheduler[1])
                        text = f"{sampler_name} ({scheduler[0]})"
                    else:
                        scheduler = (var, scheduler[1])
                        text = f"{scheduler[0]}"
                    text = text.replace("ancestral", "a").replace("uniform", "u").replace("exponential","exp")

                # If var_type is "Denoise", update denoise and generate labels
                elif var_type == "Denoise":
                    denoise = var
                    text = f"Denoise: {round(denoise, 2)}"

                # If var_type is "VAE", update vae_name and generate labels
                elif var_type == "VAE":
                    vae_name = var
                    vae_filename = os.path.splitext(os.path.basename(vae_name))[0]
                    text = f"VAE: {vae_filename}"

                # If var_type is "Positive Prompt S/R", update positive_prompt and generate labels
                elif var_type == "Positive Prompt S/R":
                    search_txt, replace_txt = var
                    if replace_txt != None:
                        # check if we are in the Y loop after the X loop
                        if positive_prompt[2] is not None:
                            positive_prompt = (positive_prompt[2].replace(search_txt, replace_txt, 1), positive_prompt[1], positive_prompt[2])
                        else:
                            positive_prompt = (positive_prompt[1].replace(search_txt, replace_txt, 1), positive_prompt[1], positive_prompt[1].replace(search_txt, replace_txt, 1))
                    else:
                        if positive_prompt[2] is not None:
                            positive_prompt = (positive_prompt[2], positive_prompt[1], positive_prompt[2])
                        else:
                            positive_prompt = (positive_prompt[1], positive_prompt[1], positive_prompt[1])
                        replace_txt = search_txt
                    text = f"{replace_txt}"

                # If var_type is "Negative Prompt S/R", update negative_prompt and generate labels
                elif var_type == "Negative Prompt S/R":
                    search_txt, replace_txt = var
                    if replace_txt != None:
                        # check if we are in the Y loop after the X loop
                        if negative_prompt[2] is not None:
                            negative_prompt = (negative_prompt[2].replace(search_txt, replace_txt, 1), negative_prompt[1], negative_prompt[2])
                        else:
                            negative_prompt = (negative_prompt[1].replace(search_txt, replace_txt, 1), negative_prompt[1], negative_prompt[1].replace(search_txt, replace_txt, 1))
                    else:
                        if negative_prompt[2] is not None:
                            negative_prompt = (negative_prompt[2], negative_prompt[1], negative_prompt[2])
                        else:
                            negative_prompt = (negative_prompt[1], negative_prompt[1], negative_prompt[1])
                        replace_txt = search_txt
                    text = f"(-) {replace_txt}"

                # If var_type is "AScore+", update positive ascore and generate labels
                elif var_type == "AScore+":
                    ascore = (var,ascore[1])
                    text = f"+AScore: {ascore[0]}"

                # If var_type is "AScore-", update negative ascore and generate labels
                elif var_type == "AScore-":
                    ascore = (ascore[0],var)
                    text = f"-AScore: {ascore[1]}"

                # If var_type is "Checkpoint", update model and clip (if needed) and generate labels
                elif var_type == "Checkpoint":
                    ckpt_name = var[0]
                    if var[1] == None:
                        clip_skip = (clip_skip[1],clip_skip[1])
                    else:
                        clip_skip = (var[1],clip_skip[1])
                    if var[2] != None:
                        vae_name = var[2]
                    ckpt_filename = os.path.splitext(os.path.basename(ckpt_name))[0]
                    text = f"{ckpt_filename}"

                # If var_type is "Refiner", update model and clip (if needed) and generate labels
                elif var_type == "Refiner":
                    refiner_name = var[0]
                    if var[1] == None:
                        refiner_clip_skip = (refiner_clip_skip[1],refiner_clip_skip[1])
                    else:
                        refiner_clip_skip = (var[1],refiner_clip_skip[1])
                    ckpt_filename = os.path.splitext(os.path.basename(refiner_name))[0]
                    text = f"{ckpt_filename}"

                # If var_type is "Refiner On/Off", set end_at_step = max steps and generate labels
                elif var_type == "Refiner On/Off":
                    end_at_step = int(var * steps)
                    text = f"Refiner: {'On' if var < 1 else 'Off'}"

                elif var_type == "Clip Skip":
                    clip_skip = (var, clip_skip[1])
                    text = f"ClipSkip ({clip_skip[0]})"

                elif var_type == "Clip Skip (Refiner)":
                    refiner_clip_skip = (var, refiner_clip_skip[1])
                    text = f"RefClipSkip ({refiner_clip_skip[0]})"

                elif "LoRA" in var_type:
                    if not lora_stack or var_type == "LoRA Stacks":
                        lora_stack = var.copy()
                    else:
                        # Updating the first tuple of lora_stack
                        lora_stack[0] = tuple(v if v is not None else lora_stack[0][i] for i, v in enumerate(var[0]))

                    max_label_len = 50 + (12 * (len(lora_stack) - 1))
                    lora_name, lora_model_wt, lora_clip_wt = lora_stack[0]
                    lora_filename = os.path.splitext(os.path.basename(lora_name))[0]

                    if var_type == "LoRA" or var_type == "LoRA Stacks":
                        if len(lora_stack) == 1:
                            lora_model_wt = format(float(lora_model_wt), ".2f").rstrip('0').rstrip('.')
                            lora_clip_wt = format(float(lora_clip_wt), ".2f").rstrip('0').rstrip('.')
                            lora_filename = lora_filename[:max_label_len - len(f"LoRA: ({lora_model_wt})")]
                            if lora_model_wt == lora_clip_wt:
                                text = f"LoRA: {lora_filename}({lora_model_wt})"
                            else:
                                text = f"LoRA: {lora_filename}({lora_model_wt},{lora_clip_wt})"
                        elif len(lora_stack) > 1:
                            lora_filenames = [os.path.splitext(os.path.basename(lora_name))[0] for lora_name, _, _ in
                                              lora_stack]
                            lora_details = [(format(float(lora_model_wt), ".2f").rstrip('0').rstrip('.'),
                                             format(float(lora_clip_wt), ".2f").rstrip('0').rstrip('.')) for
                                            _, lora_model_wt, lora_clip_wt in lora_stack]
                            non_name_length = sum(
                                len(f"({lora_details[i][0]},{lora_details[i][1]})") + 2 for i in range(len(lora_stack)))
                            available_space = max_label_len - non_name_length
                            max_name_length = available_space // len(lora_stack)
                            lora_filenames = [filename[:max_name_length] for filename in lora_filenames]
                            text_elements = [
                                f"{lora_filename}({lora_details[i][0]})" if lora_details[i][0] == lora_details[i][1]
                                else f"{lora_filename}({lora_details[i][0]},{lora_details[i][1]})" for i, lora_filename in
                                enumerate(lora_filenames)]
                            text = " ".join(text_elements)

                    elif var_type == "LoRA Batch":
                        text = f"LoRA: {lora_filename}"

                    elif var_type == "LoRA Wt":
                        lora_model_wt = format(float(lora_model_wt), ".2f").rstrip('0').rstrip('.')
                        text = f"LoRA Wt: {lora_model_wt}"

                    elif var_type == "LoRA MStr":
                        lora_model_wt = format(float(lora_model_wt), ".2f").rstrip('0').rstrip('.')
                        text = f"LoRA Mstr: {lora_model_wt}"

                    elif var_type == "LoRA CStr":
                        lora_clip_wt = format(float(lora_clip_wt), ".2f").rstrip('0').rstrip('.')
                        text = f"LoRA Cstr: {lora_clip_wt}"

                elif var_type in ["ControlNetStrength", "ControlNetStart%", "ControlNetEnd%"]:
                    if "Strength" in var_type:
                        entry_index = 2
                    elif "Start%" in var_type:
                        entry_index = 3
                    elif "End%" in var_type:
                        entry_index = 4

                    # If the first entry of cnet_stack is None, set it to var
                    if cnet_stack[0] is None:
                        cnet_stack = (var, cnet_stack[1])
                    else:
                        # Extract the desired entry from var's first tuple
                        entry_from_var = var[0][entry_index]

                        # Extract the first tuple from cnet_stack[0][0] and make it mutable
                        first_cn_entry = list(cnet_stack[0][0])

                        # Replace the appropriate entry
                        first_cn_entry[entry_index] = entry_from_var

                        # Further update first_cn_entry based on cnet_stack[1]
                        for i, value in enumerate(cnet_stack[1][-3:]):  # Considering last 3 entries
                            if value is not None:
                                first_cn_entry[i + 2] = value  # "+2" to offset for the first 2 entries of the tuple

                        # Convert back to tuple for the updated values
                        updated_first_entry = tuple(first_cn_entry)

                        # Construct the updated cnet_stack[0] using the updated_first_entry and the rest of the values from cnet_stack[0]
                        updated_cnet_stack_0 = [updated_first_entry] + list(cnet_stack[0][1:])

                        # Update cnet_stack
                        cnet_stack = (updated_cnet_stack_0, cnet_stack[1])

                    # Print the desired value
                    text = f'{var_type}: {round(cnet_stack[0][0][entry_index], 3)}'

                elif var_type == "XY_Capsule":
                    text = var.getLabel()

                else: # No matching type found
                    text=""

                def truncate_texts(texts, num_label, max_label_len):
                    truncate_length = max(min(max(len(text) for text in texts), max_label_len), 24)

                    return [text if len(text) <= truncate_length else text[:truncate_length] + "..." for text in
                            texts]

                # Add the generated text to var_label if it's not full
                if len(var_label) < num_label:
                    var_label.append(text)

                # If var_type VAE , truncate entries in the var_label list when it's full
                if len(var_label) == num_label and (var_type == "VAE" or var_type == "Checkpoint"
                                                    or var_type == "Refiner" or "LoRA" in var_type):
                    var_label = truncate_texts(var_label, num_label, max_label_len)

                # Return the modified variables
                return add_noise, seed, steps, start_at_step, end_at_step, return_with_leftover_noise, cfg,\
                    sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip, \
                    refiner_name, refiner_clip_skip, positive_prompt, negative_prompt, ascore,\
                    lora_stack, cnet_stack, var_label

            #_______________________________________________________________________________________________________
            # The function below is used to optimally load Checkpoint/LoRA/VAE models between generations.
            def define_model(model, clip, clip_skip, refiner_model, refiner_clip, refiner_clip_skip,
                             ckpt_name, refiner_name, positive, negative, refiner_positive, refiner_negative,
                             positive_prompt, negative_prompt, ascore, vae, vae_name, lora_stack, cnet_stack, index,
                             types, xyplot_id, cache, sampler_type, empty_latent_width, empty_latent_height):

                # Variable to track wether to encode prompt or not
                encode = False
                encode_refiner = False

                # Unpack types tuple
                X_type, Y_type = types

                # Note: Index is held at 0 when Y_type == "Nothing"

                # Load Checkpoint if required. If Y_type is LoRA, required models will be loaded by load_lora func.
                if (X_type == "Checkpoint" and index == 0 and Y_type not in ("LoRA", "LoRA Stacks")):
                    if lora_stack is None:
                        model, clip, _ = load_checkpoint(ckpt_name, xyplot_id, cache=cache[1])
                    else: # Load Efficient Loader LoRA
                        model, clip = load_lora(lora_stack, ckpt_name, xyplot_id,
                                                cache=None, ckpt_cache=cache[1])
                    encode = True

                # Load LoRA if required
                elif (X_type in ("LoRA", "LoRA Stacks")):
                    # Don't cache Checkpoints
                    model, clip = load_lora(lora_stack, ckpt_name, xyplot_id, cache=cache[2])
                    encode = True
                elif Y_type in ("LoRA", "LoRA Stacks"):  # X_type must be Checkpoint, so cache those as defined
                    model, clip = load_lora(lora_stack, ckpt_name, xyplot_id,
                                            cache=None, ckpt_cache=cache[1])
                    encode = True
                elif X_type == "LoRA Batch" or X_type == "LoRA Wt" or X_type == "LoRA MStr" or X_type == "LoRA CStr":
                    # Don't cache Checkpoints or LoRAs
                    model, clip = load_lora(lora_stack, ckpt_name, xyplot_id, cache=0)
                    encode = True

                if (X_type == "Refiner" and index == 0) or Y_type == "Refiner":
                    refiner_model, refiner_clip, _ = \
                        load_checkpoint(refiner_name, xyplot_id, output_vae=False, cache=cache[3], ckpt_type="refn")
                    encode_refiner = True

                # Encode base prompt if required
                encode_types = ["Positive Prompt S/R", "Negative Prompt S/R", "Clip Skip", "ControlNetStrength",
                                "ControlNetStart%",  "ControlNetEnd%", "XY_Capsule"]
                if (X_type in encode_types and index == 0) or Y_type in encode_types:
                    encode = True

                # Encode refiner prompt if required
                encode_refiner_types = ["Positive Prompt S/R", "Negative Prompt S/R", "AScore+", "AScore-",
                                        "Clip Skip (Refiner)", "XY_Capsule"]
                if (X_type in encode_refiner_types and index == 0) or Y_type in encode_refiner_types:
                    encode_refiner = True

                # Encode base prompt
                if encode == True:
                    positive, negative, clip = \
                        encode_prompts(positive_prompt, negative_prompt, token_normalization, weight_interpretation,
                                       clip, clip_skip, refiner_clip, refiner_clip_skip, ascore, sampler_type == "sdxl",
                                       empty_latent_width, empty_latent_height, return_type="base")
                    # Apply ControlNet Stack if given
                    if cnet_stack:
                        controlnet_conditioning = TSC_Apply_ControlNet_Stack().apply_cnet_stack(positive, negative, cnet_stack)
                        positive, negative = controlnet_conditioning[0], controlnet_conditioning[1]

                if encode_refiner == True:
                    refiner_positive, refiner_negative, refiner_clip = \
                        encode_prompts(positive_prompt, negative_prompt, token_normalization, weight_interpretation,
                                       clip, clip_skip, refiner_clip, refiner_clip_skip, ascore, sampler_type == "sdxl",
                                       empty_latent_width, empty_latent_height, return_type="refiner")

                # Load VAE if required
                if (X_type == "VAE" and index == 0) or Y_type == "VAE":
                    #vae = load_vae(vae_name, xyplot_id, cache=cache[0])
                    vae = get_bvae_by_ckpt_name(ckpt_name) if vae_name == "Baked VAE" \
                        else load_vae(vae_name, xyplot_id, cache=cache[0])
                elif X_type == "Checkpoint" and index == 0 and vae_name:
                    vae = get_bvae_by_ckpt_name(ckpt_name) if vae_name == "Baked VAE" \
                        else load_vae(vae_name, xyplot_id, cache=cache[0])

                return model, positive, negative, refiner_model, refiner_positive, refiner_negative, vae

            # ______________________________________________________________________________________________________
            # The below function is used to generate the results based on all the processed variables
            def process_values(model, refiner_model, add_noise, seed, steps, start_at_step, end_at_step,
                               return_with_leftover_noise, cfg, sampler_name, scheduler, positive, negative,
                               refiner_positive, refiner_negative, latent_image, denoise, vae, vae_decode,
                               sampler_type, latent_list=[], image_tensor_list=[], image_pil_list=[], xy_capsule=None):

                capsule_result = None
                if xy_capsule is not None:
                    capsule_result = xy_capsule.get_result(model, clip, vae)
                    if capsule_result is not None:
                        image, latent = capsule_result
                        latent_list.append(latent)

                if capsule_result is None:

                    samples, images, _, _ = process_latent_image(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                                  latent_image, denoise, sampler_type, add_noise, start_at_step,
                                                  end_at_step, return_with_leftover_noise, refiner_model,
                                                  refiner_positive, refiner_negative, vae, vae_decode, preview_method)

                    # Add the latent tensor to the tensors list
                    latent_list.append(samples)

                    # Decode the latent tensor if required
                    image = images if images is not None else vae_decode_latent(vae, samples, vae_decode)

                    if xy_capsule is not None:
                        xy_capsule.set_result(image, samples)

                # Add the resulting image tensor to image_tensor_list
                image_tensor_list.append(image)

                # Convert the image from tensor to PIL Image and add it to the image_pil_list
                image_pil_list.append(tensor2pil(image))

                # Return the touched variables
                return latent_list, image_tensor_list, image_pil_list

            # ______________________________________________________________________________________________________
            # The below section is the heart of the XY Plot image generation

             # Initiate Plot label text variables X/Y_label
            X_label = []
            Y_label = []

            # Store the KSamplers original scheduler inside the same scheduler variable
            scheduler = (scheduler, scheduler)

            # Store the Eff Loaders original clip_skips inside the same clip_skip variables
            clip_skip = (clip_skip, clip_skip)
            refiner_clip_skip = (refiner_clip_skip, refiner_clip_skip)

            # Store types in a Tuple for easy function passing
            types = (X_type, Y_type)

            # Clone original model parameters
            def clone_or_none(*originals):
                cloned_items = []
                for original in originals:
                    try:
                        cloned_items.append(original.clone())
                    except (AttributeError, TypeError):
                        # If not clonable, just append the original item
                        cloned_items.append(original)
                return cloned_items
            original_model, original_clip, original_positive, original_negative,\
                original_refiner_model, original_refiner_clip, original_refiner_positive, original_refiner_negative =\
                clone_or_none(model, clip, positive, negative, refiner_model, refiner_clip, refiner_positive, refiner_negative)

            # Fill Plot Rows (X)
            for X_index, X in enumerate(X_value):
                # add a none value in the positive prompt memory.
                # the tuple is composed of (actual prompt, original prompte before S/R, prompt after X S/R)
                positive_prompt = (positive_prompt[0], positive_prompt[1], None)
                negative_prompt = (negative_prompt[0], negative_prompt[1], None)

                # Define X parameters and generate labels
                add_noise, seed, steps, start_at_step, end_at_step, return_with_leftover_noise, cfg,\
                    sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip,\
                    refiner_name, refiner_clip_skip, positive_prompt, negative_prompt, ascore,\
                    lora_stack, cnet_stack, X_label = \
                    define_variable(X_type, X, add_noise, seed, steps, start_at_step, end_at_step,
                                    return_with_leftover_noise, cfg, sampler_name, scheduler, denoise, vae_name,
                                    ckpt_name, clip_skip, refiner_name, refiner_clip_skip, positive_prompt,
                                    negative_prompt, ascore, lora_stack, cnet_stack, X_label, len(X_value))

                if X_type != "Nothing" and Y_type == "Nothing":
                    if X_type == "XY_Capsule":
                        model, clip, refiner_model, refiner_clip = \
                            clone_or_none(original_model, original_clip, original_refiner_model, original_refiner_clip)
                        model, clip, vae = X.pre_define_model(model, clip, vae)

                    # Models & Conditionings
                    model, positive, negative, refiner_model, refiner_positive, refiner_negative, vae = \
                        define_model(model, clip, clip_skip[0], refiner_model, refiner_clip, refiner_clip_skip[0],
                                     ckpt_name, refiner_name, positive, negative, refiner_positive, refiner_negative,
                                     positive_prompt[0], negative_prompt[0], ascore, vae, vae_name, lora_stack, cnet_stack[0],
                                     0, types, xyplot_id, cache, sampler_type, empty_latent_width, empty_latent_height)

                    xy_capsule = None
                    if X_type == "XY_Capsule":
                        xy_capsule = X

                    # Generate Results
                    latent_list, image_tensor_list, image_pil_list = \
                        process_values(model, refiner_model, add_noise, seed, steps, start_at_step, end_at_step,
                                       return_with_leftover_noise, cfg, sampler_name, scheduler[0], positive, negative,
                                       refiner_positive, refiner_negative, latent_image, denoise, vae, vae_decode, sampler_type, xy_capsule=xy_capsule)

                elif X_type != "Nothing" and Y_type != "Nothing":
                    for Y_index, Y in enumerate(Y_value):

                        if Y_type == "XY_Capsule" or X_type == "XY_Capsule":
                            model, clip, refiner_model, refiner_clip = \
                                clone_or_none(original_model, original_clip, original_refiner_model, original_refiner_clip)

                        if Y_type == "XY_Capsule" and X_type == "XY_Capsule":
                            Y.set_x_capsule(X)

                        # Define Y parameters and generate labels
                        add_noise, seed, steps, start_at_step, end_at_step, return_with_leftover_noise, cfg,\
                            sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip,\
                            refiner_name, refiner_clip_skip, positive_prompt, negative_prompt, ascore,\
                            lora_stack, cnet_stack, Y_label = \
                            define_variable(Y_type, Y, add_noise, seed, steps, start_at_step, end_at_step,
                                            return_with_leftover_noise, cfg, sampler_name, scheduler, denoise, vae_name,
                                            ckpt_name, clip_skip, refiner_name, refiner_clip_skip, positive_prompt,
                                            negative_prompt, ascore, lora_stack, cnet_stack, Y_label, len(Y_value))

                        if Y_type == "XY_Capsule":
                            model, clip, vae = Y.pre_define_model(model, clip, vae)
                        elif X_type == "XY_Capsule":
                            model, clip, vae = X.pre_define_model(model, clip, vae)

                        # Models & Conditionings
                        model, positive, negative, refiner_model, refiner_positive, refiner_negative, vae = \
                            define_model(model, clip, clip_skip[0], refiner_model, refiner_clip, refiner_clip_skip[0],
                                         ckpt_name, refiner_name, positive, negative, refiner_positive, refiner_negative,
                                         positive_prompt[0], negative_prompt[0], ascore, vae, vae_name, lora_stack, cnet_stack[0],
                                         Y_index, types, xyplot_id, cache, sampler_type, empty_latent_width,
                                         empty_latent_height)

                        # Generate Results
                        xy_capsule = None
                        if Y_type == "XY_Capsule":
                            xy_capsule = Y

                        latent_list, image_tensor_list, image_pil_list = \
                            process_values(model, refiner_model, add_noise, seed, steps, start_at_step, end_at_step,
                                           return_with_leftover_noise, cfg, sampler_name, scheduler[0],
                                           positive, negative, refiner_positive, refiner_negative, latent_image,
                                           denoise, vae, vae_decode, sampler_type, xy_capsule=xy_capsule)

            # Clean up cache
            if cache_models == "False":
                clear_cache_by_exception(xyplot_id, vae_dict=[], ckpt_dict=[], lora_dict=[], refn_dict=[])
            else:
                # Avoid caching models accross both X and Y
                if X_type == "Checkpoint":
                    clear_cache_by_exception(xyplot_id, lora_dict=[], refn_dict=[])
                elif X_type == "Refiner":
                    clear_cache_by_exception(xyplot_id, ckpt_dict=[], lora_dict=[])
                elif X_type in ("LoRA", "LoRA Stacks"):
                    clear_cache_by_exception(xyplot_id, ckpt_dict=[], refn_dict=[])

            # __________________________________________________________________________________________________________
            # Function for printing all plot variables (WARNING: This function is an absolute mess)
            def print_plot_variables(X_type, Y_type, X_value, Y_value, add_noise, seed, steps, start_at_step, end_at_step,
                                     return_with_leftover_noise, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name,
                                     clip_skip, refiner_name, refiner_clip_skip, ascore, lora_stack, cnet_stack, sampler_type,
                                     num_rows, num_cols, i_height, i_width):

                print("-" * 40)  # Print an empty line followed by a separator line
                print(f"{xyplot_message('XY Plot Results:')}")

                def get_vae_name(X_type, Y_type, X_value, Y_value, vae_name):
                    if X_type == "VAE":
                        vae_name = "\n      ".join(map(lambda x: os.path.splitext(os.path.basename(str(x)))[0], X_value))
                    elif Y_type == "VAE":
                        vae_name = "\n      ".join(map(lambda y: os.path.splitext(os.path.basename(str(y)))[0], Y_value))
                    elif vae_name:
                        vae_name = os.path.splitext(os.path.basename(str(vae_name)))[0]
                    else:
                        vae_name = ""
                    return vae_name

                def get_clip_skip(X_type, Y_type, X_value, Y_value, cskip, mode):
                    clip_type = "Clip Skip" if mode == "ckpt" else "Clip Skip (Refiner)"
                    if X_type == clip_type:
                        cskip = ", ".join(map(str, X_value))
                    elif Y_type == clip_type:
                        cskip = ", ".join(map(str, Y_value))
                    elif cskip[1] != None:
                        cskip = cskip[1]
                    else:
                        cskip = ""
                    return cskip

                def get_checkpoint_name(X_type, Y_type, X_value, Y_value, ckpt_name, clip_skip, mode, vae_name=None):

                    # If ckpt_name is None, return it as is
                    if ckpt_name is not None:
                        ckpt_name = os.path.basename(ckpt_name)

                    # Define types based on mode
                    primary_type = "Checkpoint" if mode == "ckpt" else "Refiner"
                    clip_type = "Clip Skip" if mode == "ckpt" else "Clip Skip (Refiner)"

                    # Determine ckpt and othr based on primary type
                    if X_type == primary_type:
                        ckpt_type, ckpt_value = X_type, X_value.copy()
                        othr_type, othr_value = Y_type, Y_value.copy()
                    elif Y_type == primary_type:
                        ckpt_type, ckpt_value = Y_type, Y_value.copy()
                        othr_type, othr_value = X_type, X_value.copy()
                    else:
                        # Process as per original function if mode is "ckpt"
                        clip_skip = get_clip_skip(X_type, Y_type, X_value, Y_value, clip_skip, mode)
                        if mode == "ckpt":
                            if vae_name:
                                vae_name = get_vae_name(X_type, Y_type, X_value, Y_value, vae_name)
                            return ckpt_name, clip_skip, vae_name
                        else:
                            # For refn mode
                            return ckpt_name, clip_skip

                    # Process clip skip based on mode
                    if othr_type == clip_type:
                        clip_skip = ", ".join(map(str, othr_value))
                    elif ckpt_value[0][1] != None:
                        clip_skip = None

                    # Process vae_name based on mode
                    if mode == "ckpt":
                        if othr_type == "VAE":
                            vae_name = get_vae_name(X_type, Y_type, X_value, Y_value, vae_name)
                        elif ckpt_value[0][2] != None:
                            vae_name = None

                    def format_name(v, _type):
                        base = os.path.basename(v[0])
                        if _type == clip_type and v[1] is not None:
                            return base
                        elif _type == "VAE" and v[1] is not None and v[2] is not None:
                            return f"{base}({v[1]})"
                        elif v[1] is not None and v[2] is not None:
                            return f"{base}({v[1]}) + vae:{v[2]}"
                        elif v[1] is not None:
                            return f"{base}({v[1]})"
                        else:
                            return base

                    ckpt_name = "\n      ".join([format_name(v, othr_type) for v in ckpt_value])
                    if mode == "ckpt":
                        return ckpt_name, clip_skip, vae_name
                    else:
                        return ckpt_name, clip_skip

                def get_lora_name(X_type, Y_type, X_value, Y_value, lora_stack=None):
                    lora_name = lora_wt = lora_model_str = lora_clip_str = None

                    # Check for all possible LoRA types
                    lora_types = ["LoRA", "LoRA Stacks", "LoRA Batch", "LoRA Wt", "LoRA MStr", "LoRA CStr"]

                    def get_lora_name_for_weights(lora_path, lora_name):
                        if lora_path:
                            return os.path.basename(X_value[0][0][0]) if lora_name is None else lora_name
                        return None

                    if X_type not in lora_types and Y_type not in lora_types:
                        if lora_stack:
                            names_list = []
                            for name, model_wt, clip_wt in lora_stack:
                                base_name = os.path.splitext(os.path.basename(name))[0]
                                formatted_str = f"{base_name}({round(model_wt, 3)},{round(clip_wt, 3)})"
                                names_list.append(formatted_str)
                            lora_name = f"[{', '.join(names_list)}]"
                    else:
                        if X_type in lora_types:
                            value = get_lora_sublist_name(X_type, X_value)
                            if  X_type in ("LoRA", "LoRA Stacks"):
                                lora_name = value
                                lora_model_str = None
                                lora_clip_str = None
                            if X_type == "LoRA Batch":
                                lora_name = value
                                lora_model_str = X_value[0][0][1] if lora_model_str is None else lora_model_str
                                lora_clip_str = X_value[0][0][2] if lora_clip_str is None else lora_clip_str
                            elif X_type == "LoRA MStr":
                                lora_name = get_lora_name_for_weights(X_value[0][0][0], lora_name)
                                lora_model_str = value
                                lora_clip_str = X_value[0][0][2] if lora_clip_str is None else lora_clip_str
                            elif X_type == "LoRA CStr":
                                lora_name = get_lora_name_for_weights(X_value[0][0][0], lora_name)
                                lora_model_str = X_value[0][0][1] if lora_model_str is None else lora_model_str
                                lora_clip_str = value
                            elif X_type == "LoRA Wt":
                                lora_name = get_lora_name_for_weights(X_value[0][0][0], lora_name)
                                lora_wt = value

                        if Y_type in lora_types:
                            value = get_lora_sublist_name(Y_type, Y_value)
                            if  Y_type in ("LoRA", "LoRA Stacks"):
                                lora_name = value
                                lora_model_str = None
                                lora_clip_str = None
                            if Y_type == "LoRA Batch":
                                lora_name = value
                                lora_model_str = Y_value[0][0][1] if lora_model_str is None else lora_model_str
                                lora_clip_str = Y_value[0][0][2] if lora_clip_str is None else lora_clip_str
                            elif Y_type == "LoRA MStr":
                                lora_name = get_lora_name_for_weights(Y_value[0][0][0], lora_name)
                                lora_model_str = value
                                lora_clip_str = Y_value[0][0][2] if lora_clip_str is None else lora_clip_str
                            elif Y_type == "LoRA CStr":
                                lora_name = get_lora_name_for_weights(Y_value[0][0][0], lora_name)
                                lora_model_str = Y_value[0][0][1] if lora_model_str is None else lora_model_str
                                lora_clip_str = value
                            elif Y_type == "LoRA Wt":
                                lora_name = get_lora_name_for_weights(Y_value[0][0][0], lora_name)
                                lora_wt = value

                    return lora_name, lora_wt, lora_model_str, lora_clip_str

                def get_lora_sublist_name(lora_type, lora_value):
                    if lora_type in ("LoRA", "LoRA Batch", "LoRA Stacks"):
                        formatted_sublists = []
                        for sublist in lora_value:
                            formatted_entries = []
                            for x in sublist:
                                base_name = os.path.splitext(os.path.basename(str(x[0])))[0]
                                formatted_str = f"{base_name}({round(x[1], 3)},{round(x[2], 3)})" if lora_type in ("LoRA", "LoRA Stacks") else f"{base_name}"
                                formatted_entries.append(formatted_str)
                            formatted_sublists.append(f"{', '.join(formatted_entries)}")
                        return "\n      ".join(formatted_sublists)
                    elif lora_type == "LoRA MStr":
                        return ", ".join([str(round(x[0][1], 3)) for x in lora_value])
                    elif lora_type == "LoRA CStr":
                        return ", ".join([str(round(x[0][2], 3)) for x in lora_value])
                    elif lora_type == "LoRA Wt":
                        return ", ".join([str(round(x[0][1], 3)) for x in lora_value])  # assuming LoRA Wt uses the second value
                    else:
                        return ""

                # VAE, Checkpoint, Clip Skip, LoRA
                ckpt_name, clip_skip, vae_name = get_checkpoint_name(X_type, Y_type, X_value, Y_value, ckpt_name, clip_skip, "ckpt", vae_name)
                lora_name, lora_wt, lora_model_str, lora_clip_str = get_lora_name(X_type, Y_type, X_value, Y_value, lora_stack)
                refiner_name, refiner_clip_skip = get_checkpoint_name(X_type, Y_type, X_value, Y_value, refiner_name, refiner_clip_skip, "refn")

                # AddNoise
                add_noise = ", ".join(map(str, X_value)) if X_type == "AddNoise" else ", ".join(
                    map(str, Y_value)) if Y_type == "AddNoise" else add_noise

                # Seeds++ Batch
                seed = "\n      ".join(map(str, X_value)) if X_type == "Seeds++ Batch" else "\n      ".join(
                    map(str, Y_value)) if Y_type == "Seeds++ Batch" else seed

                # Steps
                steps = ", ".join(map(str, X_value)) if X_type == "Steps" else ", ".join(
                    map(str, Y_value)) if Y_type == "Steps" else steps

                # StartStep
                start_at_step = ", ".join(map(str, X_value)) if X_type == "StartStep" else ", ".join(
                    map(str, Y_value)) if Y_type == "StartStep" else start_at_step

                # EndStep/RefineStep
                end_at_step = ", ".join(map(str, X_value)) if X_type in ["EndStep", "RefineStep"] else ", ".join(
                    map(str, Y_value)) if Y_type in ["EndStep", "RefineStep"] else end_at_step

                # ReturnNoise
                return_with_leftover_noise = ", ".join(map(str, X_value)) if X_type == "ReturnNoise" else ", ".join(
                    map(str, Y_value)) if Y_type == "ReturnNoise" else return_with_leftover_noise

                # CFG
                cfg = ", ".join(map(str, X_value)) if X_type == "CFG Scale" else ", ".join(
                    map(str, Y_value)) if Y_type == "CFG Scale" else round(cfg,3)

                # Sampler/Scheduler
                if X_type == "Sampler":
                    if Y_type == "Scheduler":
                        sampler_name = ", ".join([f"{x[0]}" for x in X_value])
                        scheduler = ", ".join([f"{y}" for y in Y_value])
                    else:
                        sampler_name = ", ".join([f"{x[0]}({x[1] if x[1] != '' and x[1] is not None else scheduler[1]})" for x in X_value])
                        scheduler = "_"
                elif Y_type == "Sampler":
                    if X_type == "Scheduler":
                        sampler_name = ", ".join([f"{y[0]}" for y in Y_value])
                        scheduler = ", ".join([f"{x}" for x in X_value])
                    else:
                        sampler_name = ", ".join([f"{y[0]}({y[1] if y[1] != '' and y[1] is not None else scheduler[1]})" for y in Y_value])
                        scheduler = "_"
                else:
                    scheduler = ", ".join([str(x[0]) if isinstance(x, tuple) else str(x) for x in X_value]) if X_type == "Scheduler" else \
                        ", ".join([str(y[0]) if isinstance(y, tuple) else str(y) for y in Y_value]) if Y_type == "Scheduler" else scheduler[0]

                # Denoise
                denoise = ", ".join(map(str, X_value)) if X_type == "Denoise" else ", ".join(
                    map(str, Y_value)) if Y_type == "Denoise" else round(denoise,3)

                # Check if ascore is None
                if ascore is None:
                    pos_ascore = neg_ascore = None
                else:
                    # Ascore+
                    pos_ascore = (", ".join(map(str, X_value)) if X_type == "Ascore+"
                                  else ", ".join(map(str, Y_value)) if Y_type == "Ascore+" else round(ascore[0],3))
                    # Ascore-
                    neg_ascore = (", ".join(map(str, X_value)) if X_type == "Ascore-"
                                  else ", ".join(map(str, Y_value)) if Y_type == "Ascore-" else round(ascore[1],3))

                #..........................................PRINTOUTS....................................................
                print(f"(X) {X_type}")
                print(f"(Y) {Y_type}")
                print(f"img_count: {len(X_value)*len(Y_value)}")
                print(f"img_dims: {i_height} x {i_width}")
                print(f"plot_dim: {num_cols} x {num_rows}")
                print(f"ckpt: {ckpt_name if ckpt_name is not None else ''}")
                if clip_skip:
                    print(f"clip_skip: {clip_skip}")
                if sampler_type == "sdxl":
                    if refiner_clip_skip == "_":
                        print(f"refiner(clipskip): {refiner_name if refiner_name is not None else ''}")
                    else:
                        print(f"refiner: {refiner_name if refiner_name is not None else ''}")
                        print(f"refiner_clip_skip: {refiner_clip_skip if refiner_clip_skip is not None else ''}")
                        print(f"+ascore: {pos_ascore if pos_ascore is not None else ''}")
                        print(f"-ascore: {neg_ascore if neg_ascore is not None else ''}")
                if lora_name:
                    print(f"lora: {lora_name}")
                if lora_wt:
                    print(f"lora_wt: {lora_wt}")
                if lora_model_str:
                    print(f"lora_mstr: {lora_model_str}")
                if lora_clip_str:
                    print(f"lora_cstr: {lora_clip_str}")
                if vae_name:
                    print(f"vae:  {vae_name}")
                if sampler_type == "advanced":
                    print(f"add_noise: {add_noise}")
                print(f"seed: {seed}")
                print(f"steps: {steps}")
                if sampler_type == "advanced":
                    print(f"start_at_step: {start_at_step}")
                    print(f"end_at_step: {end_at_step}")
                    print(f"return_noise: {return_with_leftover_noise}")
                if sampler_type == "sdxl":
                    print(f"start_at_step: {start_at_step}")
                    if X_type == "Refiner On/Off":
                        print(f"refine_at_percent: {X_value[0]}")
                    elif Y_type == "Refiner On/Off":
                        print(f"refine_at_percent: {Y_value[0]}")
                    else:
                        print(f"refine_at_step: {end_at_step}")
                print(f"cfg: {cfg}")
                if scheduler == "_":
                    print(f"sampler(scheduler): {sampler_name}")
                else:
                    print(f"sampler: {sampler_name}")
                    print(f"scheduler: {scheduler}")
                if sampler_type == "regular":
                    print(f"denoise: {denoise}")

                if X_type == "Positive Prompt S/R" or Y_type == "Positive Prompt S/R":
                    positive_prompt = ", ".join([str(x[0]) if i == 0 else str(x[1]) for i, x in enumerate(
                        X_value)]) if X_type == "Positive Prompt S/R" else ", ".join(
                        [str(y[0]) if i == 0 else str(y[1]) for i, y in
                         enumerate(Y_value)]) if Y_type == "Positive Prompt S/R" else positive_prompt
                    print(f"+prompt_s/r: {positive_prompt}")

                if X_type == "Negative Prompt S/R" or Y_type == "Negative Prompt S/R":
                    negative_prompt = ", ".join([str(x[0]) if i == 0 else str(x[1]) for i, x in enumerate(
                        X_value)]) if X_type == "Negative Prompt S/R" else ", ".join(
                        [str(y[0]) if i == 0 else str(y[1]) for i, y in
                         enumerate(Y_value)]) if Y_type == "Negative Prompt S/R" else negative_prompt
                    print(f"-prompt_s/r: {negative_prompt}")

                if "ControlNet" in X_type or "ControlNet" in Y_type:
                    cnet_strength,  cnet_start_pct, cnet_end_pct = cnet_stack[1]

                if "ControlNet" in X_type:
                    if "Strength" in X_type:
                        cnet_strength = [str(round(inner_list[0][2], 3)) for inner_list in X_value if
                                           isinstance(inner_list, list) and
                                           inner_list and isinstance(inner_list[0], tuple) and len(inner_list[0]) >= 3]
                    if "Start%" in X_type:
                        cnet_start_pct = [str(round(inner_list[0][3], 3)) for inner_list in X_value if
                                           isinstance(inner_list, list) and
                                           inner_list and isinstance(inner_list[0], tuple) and len(inner_list[0]) >= 3]
                    if "End%" in X_type:
                        cnet_end_pct = [str(round(inner_list[0][4], 3)) for inner_list in X_value if
                                           isinstance(inner_list, list) and
                                           inner_list and isinstance(inner_list[0], tuple) and len(inner_list[0]) >= 3]
                if "ControlNet" in Y_type:
                    if "Strength" in Y_type:
                        cnet_strength = [str(round(inner_list[0][2], 3)) for inner_list in Y_value if
                                         isinstance(inner_list, list) and
                                         inner_list and isinstance(inner_list[0], tuple) and len(
                                             inner_list[0]) >= 3]
                    if "Start%" in Y_type:
                        cnet_start_pct = [str(round(inner_list[0][3], 3)) for inner_list in Y_value if
                                          isinstance(inner_list, list) and
                                          inner_list and isinstance(inner_list[0], tuple) and len(
                                              inner_list[0]) >= 3]
                    if "End%" in Y_type:
                        cnet_end_pct = [str(round(inner_list[0][4], 3)) for inner_list in Y_value if
                                         isinstance(inner_list, list) and
                                         inner_list and isinstance(inner_list[0], tuple) and len(
                                             inner_list[0]) >= 3]

                if "ControlNet" in X_type or "ControlNet" in Y_type:
                    print(f"cnet_strength: {', '.join(cnet_strength) if isinstance(cnet_strength, list) else cnet_strength}")
                    print(f"cnet_start%: {', '.join(cnet_start_pct) if isinstance(cnet_start_pct, list) else cnet_start_pct}")
                    print(f"cnet_end%: {', '.join(cnet_end_pct) if isinstance(cnet_end_pct, list) else cnet_end_pct}")

            # ______________________________________________________________________________________________________
            def adjusted_font_size(text, initial_font_size, i_width):
                font = ImageFont.truetype(str(Path(font_path)), initial_font_size)
                text_width = font.getlength(text)

                if text_width > (i_width * 0.9):
                    scaling_factor = 0.9  # A value less than 1 to shrink the font size more aggressively
                    new_font_size = int(initial_font_size * (i_width / text_width) * scaling_factor)
                else:
                    new_font_size = initial_font_size

                return new_font_size

            # ______________________________________________________________________________________________________

            def rearrange_list_A(arr, num_cols, num_rows):
                new_list = []
                for i in range(num_rows):
                    for j in range(num_cols):
                        index = j * num_rows + i
                        new_list.append(arr[index])
                return new_list

            def rearrange_list_B(arr, num_rows, num_cols):
                new_list = []
                for i in range(num_rows):
                    for j in range(num_cols):
                        index = i * num_cols + j
                        new_list.append(arr[index])
                return new_list

            # Extract plot dimensions
            num_rows = max(len(Y_value) if Y_value is not None else 0, 1)
            num_cols = max(len(X_value) if X_value is not None else 0, 1)

            # Flip X & Y results back if flipped earlier (for Checkpoint/LoRA For loop optimizations)
            if flip_xy == True:
                X_type, Y_type = Y_type, X_type
                X_value, Y_value = Y_value, X_value
                X_label, Y_label = Y_label, X_label
                num_rows, num_cols = num_cols, num_rows
                image_pil_list = rearrange_list_A(image_pil_list, num_rows, num_cols)
            else:
                image_pil_list = rearrange_list_B(image_pil_list, num_rows, num_cols)
                image_tensor_list = rearrange_list_A(image_tensor_list, num_cols, num_rows)
                latent_list = rearrange_list_A(latent_list, num_cols, num_rows)

            # Extract final image dimensions
            i_height, i_width = image_tensor_list[0].shape[1], image_tensor_list[0].shape[2]

            # Print XY Plot Results
            print_plot_variables(X_type, Y_type, X_value, Y_value, add_noise, seed,  steps, start_at_step, end_at_step,
                                 return_with_leftover_noise, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name,
                                 clip_skip, refiner_name, refiner_clip_skip, ascore, lora_stack, cnet_stack,
                                 sampler_type, num_rows, num_cols, i_height, i_width)

            # Concatenate the 'samples' and 'noise_mask' tensors along the first dimension (dim=0)
            keys = latent_list[0].keys()
            result = {}
            for key in keys:
                tensors = [d[key] for d in latent_list]
                result[key] = torch.cat(tensors, dim=0)
            latent_list = result

            # Store latent_list as last latent
            ###update_value_by_id("latent", my_unique_id, latent_list)

            # Calculate the dimensions of the white background image
            border_size_top = i_width // 15

            # Longest Y-label length
            if len(Y_label) > 0:
                Y_label_longest = max(len(s) for s in Y_label)
            else:
                # Handle the case when the sequence is empty
                Y_label_longest = 0  # or any other appropriate value

            Y_label_scale = min(Y_label_longest + 4,24) / 24

            if Y_label_orientation == "Vertical":
                border_size_left = border_size_top
            else:  # Assuming Y_label_orientation is "Horizontal"
                # border_size_left is now min(i_width, i_height) plus 20% of the difference between the two
                border_size_left = min(i_width, i_height) + int(0.2 * abs(i_width - i_height))
                border_size_left = int(border_size_left * Y_label_scale)

            # Modify the border size, background width and x_offset initialization based on Y_type and Y_label_orientation
            if Y_type == "Nothing":
                bg_width = num_cols * i_width + (num_cols - 1) * grid_spacing
                x_offset_initial = 0
            else:
                if Y_label_orientation == "Vertical":
                    bg_width = num_cols * i_width + (num_cols - 1) * grid_spacing + 3 * border_size_left
                    x_offset_initial = border_size_left * 3
                else:  # Assuming Y_label_orientation is "Horizontal"
                    bg_width = num_cols * i_width + (num_cols - 1) * grid_spacing + border_size_left
                    x_offset_initial = border_size_left

            # Modify the background height based on X_type
            if X_type == "Nothing":
                bg_height = num_rows * i_height + (num_rows - 1) * grid_spacing
                y_offset = 0
            else:
                bg_height = num_rows * i_height + (num_rows - 1) * grid_spacing + 3 * border_size_top
                y_offset = border_size_top * 3

            # Create the white background image
            background = Image.new('RGBA', (int(bg_width), int(bg_height)), color=(255, 255, 255, 255))

            for row in range(num_rows):

                # Initialize the X_offset
                x_offset = x_offset_initial

                for col in range(num_cols):
                    # Calculate the index for image_pil_list
                    index = col * num_rows + row
                    img = image_pil_list[index]

                    # Paste the image
                    background.paste(img, (x_offset, y_offset))

                    if row == 0 and X_type != "Nothing":
                        # Assign text
                        text = X_label[col]

                        # Add the corresponding X_value as a label above the image
                        initial_font_size = int(48 * img.width / 512)
                        font_size = adjusted_font_size(text, initial_font_size, img.width)
                        label_height = int(font_size*1.5)

                        # Create a white background label image
                        label_bg = Image.new('RGBA', (img.width, label_height), color=(255, 255, 255, 0))
                        d = ImageDraw.Draw(label_bg)

                        # Create the font object
                        font = ImageFont.truetype(str(Path(font_path)), font_size)

                        # Calculate the text size and the starting position
                        _, _, text_width, text_height = d.textbbox([0,0], text, font=font)
                        text_x = (img.width - text_width) // 2
                        text_y = (label_height - text_height) // 2

                        # Add the text to the label image
                        d.text((text_x, text_y), text, fill='black', font=font)

                        # Calculate the available space between the top of the background and the top of the image
                        available_space = y_offset - label_height

                        # Calculate the new Y position for the label image
                        label_y = available_space // 2

                        # Paste the label image above the image on the background using alpha_composite()
                        background.alpha_composite(label_bg, (x_offset, label_y))

                    if col == 0 and Y_type != "Nothing":
                        # Assign text
                        text = Y_label[row]

                        # Add the corresponding Y_value as a label to the left of the image
                        if Y_label_orientation == "Vertical":
                            initial_font_size = int(48 * i_width / 512)  # Adjusting this to be same as X_label size
                            font_size = adjusted_font_size(text, initial_font_size, i_width)
                        else:  # Assuming Y_label_orientation is "Horizontal"
                            initial_font_size = int(48 *  (border_size_left/Y_label_scale) / 512)  # Adjusting this to be same as X_label size
                            font_size = adjusted_font_size(text, initial_font_size,  int(border_size_left/Y_label_scale))

                        # Create a white background label image
                        label_bg = Image.new('RGBA', (img.height, int(font_size*1.2)), color=(255, 255, 255, 0))
                        d = ImageDraw.Draw(label_bg)

                        # Create the font object
                        font = ImageFont.truetype(str(Path(font_path)), font_size)

                        # Calculate the text size and the starting position
                        _, _, text_width, text_height = d.textbbox([0,0], text, font=font)
                        text_x = (img.height - text_width) // 2
                        text_y = (font_size - text_height) // 2

                        # Add the text to the label image
                        d.text((text_x, text_y), text, fill='black', font=font)

                        # Rotate the label_bg 90 degrees counter-clockwise only if Y_label_orientation is "Vertical"
                        if Y_label_orientation == "Vertical":
                            label_bg = label_bg.rotate(90, expand=True)

                        # Calculate the available space between the left of the background and the left of the image
                        available_space = x_offset - label_bg.width

                        # Calculate the new X position for the label image
                        label_x = available_space // 2

                        # Calculate the Y position for the label image based on its orientation
                        if Y_label_orientation == "Vertical":
                            label_y = y_offset + (img.height - label_bg.height) // 2
                        else:  # Assuming Y_label_orientation is "Horizontal"
                            label_y = y_offset + img.height - (img.height - label_bg.height) // 2

                        # Paste the label image to the left of the image on the background using alpha_composite()
                        background.alpha_composite(label_bg, (label_x, label_y))

                    # Update the x_offset
                    x_offset += img.width + grid_spacing

                # Update the y_offset
                y_offset += img.height + grid_spacing

            xy_plot_image = pil2tensor(background)

         # Generate the preview_images
        preview_images = PreviewImage().save_images(xy_plot_image)["ui"]["images"]

        # Generate output_images
        output_images = torch.stack([tensor.squeeze() for tensor in image_tensor_list])

        # Set the output_image the same as plot image defined by 'xyplot_as_output_image'
        if xyplot_as_output_image == True:
            output_images = xy_plot_image

        # Print cache if set to true
        if cache_models == "True":
            print_loaded_objects_entries(xyplot_id, prompt)

        print("-" * 40)  # Print an empty line followed by a separator line

        if sampler_type == "sdxl":
            sdxl_tuple = original_model, original_clip, original_positive, original_negative,\
                original_refiner_model, original_refiner_clip, original_refiner_positive, original_refiner_negative
            result = (sdxl_tuple, latent_list, optional_vae, output_images,)
        else:
            result = (original_model, original_positive, original_negative, latent_list, optional_vae, output_images,)
        return {"ui": {"images": preview_images}, "result": result}

#=======================================================================================================================
# TSC KSampler Adv (Efficient)
class TSC_KSamplerAdvanced(TSC_KSampler):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"model": ("MODEL",),
                     "add_noise": (["enable", "disable"],),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "return_with_leftover_noise": (["disable", "enable"],),
                     "preview_method": (["auto", "latent2rgb", "taesd", "none"],),
                     "vae_decode": (["true", "true (tiled)", "false", "output only", "output only (tiled)"],),
                     },
                "optional": {"optional_vae": ("VAE",),
                             "script": ("SCRIPT",), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID", },
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE",)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "sample_adv"
    CATEGORY = "Efficiency Nodes/Sampling"

    def sample_adv(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, start_at_step, end_at_step, return_with_leftover_noise, preview_method, vae_decode,
               prompt=None, extra_pnginfo=None, my_unique_id=None, optional_vae=(None,), script=None):

        return super().sample(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, preview_method, vae_decode, denoise=1.0, prompt=prompt, extra_pnginfo=extra_pnginfo, my_unique_id=my_unique_id,
               optional_vae=optional_vae, script=script, add_noise=add_noise, start_at_step=start_at_step,end_at_step=end_at_step,
                       return_with_leftover_noise=return_with_leftover_noise,sampler_type="advanced")

#=======================================================================================================================
# TSC KSampler SDXL (Efficient)
class TSC_KSamplerSDXL(TSC_KSampler):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"sdxl_tuple": ("SDXL_TUPLE",),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (SCHEDULERS,),
                     "latent_image": ("LATENT",),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "refine_at_step": ("INT", {"default": -1, "min": -1, "max": 10000}),
                     "preview_method": (["auto", "latent2rgb", "taesd", "none"],),
                     "vae_decode": (["true", "true (tiled)", "false", "output only", "output only (tiled)"],),
                     },
                "optional": {"optional_vae": ("VAE",),
                             "script": ("SCRIPT",),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("SDXL_TUPLE", "LATENT", "VAE", "IMAGE",)
    RETURN_NAMES = ("SDXL_TUPLE", "LATENT", "VAE", "IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "sample_sdxl"
    CATEGORY = "Efficiency Nodes/Sampling"

    def sample_sdxl(self, sdxl_tuple, noise_seed, steps, cfg, sampler_name, scheduler, latent_image,
               start_at_step, refine_at_step, preview_method, vae_decode, prompt=None, extra_pnginfo=None,
               my_unique_id=None, optional_vae=(None,), refiner_extras=None, script=None):
        # sdxl_tuple sent through the 'model' channel
        negative = None
        return super().sample(sdxl_tuple, noise_seed, steps, cfg, sampler_name, scheduler,
               refiner_extras, negative, latent_image, preview_method, vae_decode, denoise=1.0,
               prompt=prompt, extra_pnginfo=extra_pnginfo, my_unique_id=my_unique_id, optional_vae=optional_vae,
               script=script, add_noise=None, start_at_step=start_at_step, end_at_step=refine_at_step,
               return_with_leftover_noise=None,sampler_type="sdxl")

########################################################################################################################
# Common XY Plot Functions/Variables
XYPLOT_LIM = 50 #XY Plot default axis size limit
XYPLOT_DEF = 3  #XY Plot default batch count
CKPT_EXTENSIONS = LORA_EXTENSIONS = ['.safetensors', '.ckpt']
VAE_EXTENSIONS = ['.safetensors', '.ckpt', '.pt']
try:
    xy_batch_default_path = os.path.abspath(os.sep) + "example_folder"
except Exception:
    xy_batch_default_path = ""

def generate_floats(batch_count, first_float, last_float):
    if batch_count > 1:
        interval = (last_float - first_float) / (batch_count - 1)
        return [round(first_float + i * interval, 3) for i in range(batch_count)]
    else:
        return [first_float] if batch_count == 1 else []

def generate_ints(batch_count, first_int, last_int):
    if batch_count > 1:
        interval = (last_int - first_int) / (batch_count - 1)
        values = [int(first_int + i * interval) for i in range(batch_count)]
    else:
        values = [first_int] if batch_count == 1 else []
    values = list(set(values))  # Remove duplicates
    values.sort()  # Sort in ascending order
    return values

def get_batch_files(directory_path, valid_extensions, include_subdirs=False):
    batch_files = []

    try:
        if include_subdirs:
            # Using os.walk to get files from subdirectories
            for dirpath, dirnames, filenames in os.walk(directory_path):
                for file in filenames:
                    if any(file.endswith(ext) for ext in valid_extensions):
                        batch_files.append(os.path.join(dirpath, file))
        else:
            # Previous code for just the given directory
            batch_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if
                           os.path.isfile(os.path.join(directory_path, f)) and any(
                               f.endswith(ext) for ext in valid_extensions)]
    except Exception as e:
        print(f"Error while listing files in {directory_path}: {e}")

    return batch_files

def print_xy_values(xy_type, xy_value, xy_name):
    print("===== XY Value Returns =====")
    print(f"{xy_name} Values:")
    print("- Type:", xy_type)
    print("- Entries:", xy_value)
    print("============================")

#=======================================================================================================================
# TSC XY Plot
class TSC_XYplot:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "grid_spacing": ("INT", {"default": 0, "min": 0, "max": 500, "step": 5}),
                    "XY_flip": (["False","True"],),
                    "Y_label_orientation": (["Horizontal", "Vertical"],),
                    "cache_models": (["True", "False"],),
                    "ksampler_output_image": (["Images","Plot"],),},
                "optional": {
                    "dependencies": ("DEPENDENCIES", ),
                    "X": ("XY", ),
                    "Y": ("XY", ),},
                "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("SCRIPT",)
    RETURN_NAMES = ("SCRIPT",)
    FUNCTION = "XYplot"
    CATEGORY = "Efficiency Nodes/Scripts"

    def XYplot(self, grid_spacing, XY_flip, Y_label_orientation, cache_models, ksampler_output_image, my_unique_id,
               dependencies=None, X=None, Y=None):

        # Unpack X & Y Tuples if connected
        if X != None:
            X_type, X_value  = X
        else:
            X_type = "Nothing"
            X_value = [""]
        if Y != None:
            Y_type, Y_value = Y
        else:
            Y_type = "Nothing"
            Y_value = [""]

        # If types are the same exit. If one isn't "Nothing", print error
        if X_type != "XY_Capsule" and (X_type == Y_type) and X_type not in ["Positive Prompt S/R", "Negative Prompt S/R"]:
            if X_type != "Nothing":
                print(f"{error('XY Plot Error:')} X and Y input types must be different.")
            return (None,)

        # Check that dependencies are connected for specific plot types
        encode_types = {
            "Checkpoint", "Refiner",
            "LoRA", "LoRA Stacks", "LoRA Batch", "LoRA Wt", "LoRA MStr", "LoRA CStr",
            "Positive Prompt S/R", "Negative Prompt S/R",
            "AScore+", "AScore-",
            "Clip Skip", "Clip Skip (Refiner)",
            "ControlNetStrength", "ControlNetStart%", "ControlNetEnd%"
        }

        if X_type in encode_types or Y_type in encode_types:
            if dependencies is None:  # Not connected
                print(f"{error('XY Plot Error:')} The dependencies input must be connected for certain plot types.")
                # Return None
                return (None,)

        # Check if both X_type and Y_type are special lora_types
        lora_types = {"LoRA Batch", "LoRA Wt", "LoRA MStr", "LoRA CStr"}
        if (X_type in lora_types and Y_type not in lora_types) or (Y_type in lora_types and X_type not in lora_types):
            print(f"{error('XY Plot Error:')} Both X and Y must be connected to use the 'LoRA Plot' node.")
            return (None,)

        # Do not allow LoRA and LoRA Stacks
        lora_types = {"LoRA", "LoRA Stacks"}
        if (X_type in lora_types and Y_type in lora_types):
            print(f"{error('XY Plot Error:')} X and Y input types must be different.")
            return (None,)

        # Clean Schedulers from Sampler data (if other type is Scheduler)
        if X_type == "Sampler" and Y_type == "Scheduler":
            # Clear X_value Scheduler's
            X_value = [(x[0], "") for x in X_value]
        elif Y_type == "Sampler" and X_type == "Scheduler":
            # Clear Y_value Scheduler's
            Y_value = [(y[0], "") for y in Y_value]

        # Embed information into "Scheduler" X/Y_values for text label
        if X_type == "Scheduler" and Y_type != "Sampler":
            # X_value second tuple value of each array entry = None
            X_value = [(x, None) for x in X_value]

        if Y_type == "Scheduler" and X_type != "Sampler":
            # Y_value second tuple value of each array entry = None
            Y_value = [(y, None) for y in Y_value]

        # Clean VAEs from Checkpoint data if other type is VAE
        if X_type == "Checkpoint" and Y_type == "VAE":
            # Clear X_value VAE's
            X_value = [(t[0], t[1], None) for t in X_value]
        elif Y_type == "VAE" and X_type == "Checkpoint":
            # Clear Y_value VAE's
            Y_value = [(t[0], t[1], None) for t in Y_value]

        # Flip X and Y
        if XY_flip == "True":
            X_type, Y_type = Y_type, X_type
            X_value, Y_value = Y_value, X_value
            
        # Define Ksampler output image behavior
        xyplot_as_output_image = ksampler_output_image == "Plot"

        # Pack xyplot tuple into its dictionary item under script
        script = {"xyplot": (X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, cache_models,
                        xyplot_as_output_image, my_unique_id, dependencies)}

        return (script,)

#=======================================================================================================================
# TSC XY Plot: Seeds Values
class TSC_XYplot_SeedsBatch:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, batch_count):
        if batch_count == 0:
            return (None,)
        xy_type = "Seeds++ Batch"
        xy_value = list(range(batch_count))
        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: Add/Return Noise
class TSC_XYplot_AddReturnNoise:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "XY_type": (["add_noise", "return_with_leftover_noise"],)}
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, XY_type):
        type_mapping = {
            "add_noise": "AddNoise",
            "return_with_leftover_noise": "ReturnNoise"
        }
        xy_type = type_mapping[XY_type]
        xy_value = ["enable", "disable"]
        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: Step Values
class TSC_XYplot_Steps:
    parameters = ["steps","start_at_step", "end_at_step", "refine_at_step"]
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_parameter": (cls.parameters,),
                "batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "first_step": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "last_step": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "first_start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "last_start_step": ("INT", {"default": 10, "min": 0, "max": 10000}),
                "first_end_step": ("INT", {"default": 10, "min": 0, "max": 10000}),
                "last_end_step": ("INT", {"default": 20, "min": 0, "max": 10000}),
                "first_refine_step": ("INT", {"default": 10, "min": 0, "max": 10000}),
                "last_refine_step": ("INT", {"default": 20, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, target_parameter, batch_count, first_step, last_step, first_start_step, last_start_step,
                 first_end_step, last_end_step, first_refine_step, last_refine_step):

        if target_parameter == "steps":
            xy_type = "Steps"
            xy_first = first_step
            xy_last = last_step
        elif target_parameter == "start_at_step":
            xy_type = "StartStep"
            xy_first = first_start_step
            xy_last = last_start_step
        elif target_parameter == "end_at_step":
            xy_type = "EndStep"
            xy_first = first_end_step
            xy_last = last_end_step
        elif target_parameter == "refine_at_step":
            xy_type = "RefineStep"
            xy_first = first_refine_step
            xy_last = last_refine_step

        xy_value = generate_ints(batch_count, xy_first, xy_last)
        return ((xy_type, xy_value),) if xy_value else (None,)

#=======================================================================================================================
# TSC XY Plot: CFG Values
class TSC_XYplot_CFG:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "first_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "last_cfg": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, batch_count, first_cfg, last_cfg):
        xy_type = "CFG Scale"
        xy_value = generate_floats(batch_count, first_cfg, last_cfg)
        return ((xy_type, xy_value),) if xy_value else (None,)

#=======================================================================================================================
# TSC XY Plot: Sampler & Scheduler Values
class TSC_XYplot_Sampler_Scheduler:
    parameters = ["sampler", "scheduler", "sampler & scheduler"]

    @classmethod
    def INPUT_TYPES(cls):
        samplers = ["None"] + comfy.samplers.KSampler.SAMPLERS
        schedulers = ["None"] + SCHEDULERS
        inputs = {
            "required": {
                "target_parameter": (cls.parameters,),
                "input_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM, "step": 1})
            }
        }
        for i in range(1, XYPLOT_LIM+1):
            inputs["required"][f"sampler_{i}"] = (samplers,)
            inputs["required"][f"scheduler_{i}"] = (schedulers,)

        return inputs

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, target_parameter, input_count, **kwargs):
        if target_parameter == "scheduler":
            xy_type = "Scheduler"
            schedulers = [kwargs.get(f"scheduler_{i}") for i in range(1, input_count + 1)]
            xy_value = [scheduler for scheduler in schedulers if scheduler != "None"]
        elif target_parameter == "sampler":
            xy_type = "Sampler"
            samplers = [kwargs.get(f"sampler_{i}") for i in range(1, input_count + 1)]
            xy_value = [(sampler, None) for sampler in samplers if sampler != "None"]
        else:
            xy_type = "Sampler"
            samplers = [kwargs.get(f"sampler_{i}") for i in range(1, input_count + 1)]
            schedulers = [kwargs.get(f"scheduler_{i}") for i in range(1, input_count + 1)]
            xy_value = [(sampler, scheduler if scheduler != "None" else None) for sampler,
            scheduler in zip(samplers, schedulers) if sampler != "None"]

        return ((xy_type, xy_value),) if xy_value else (None,)

#=======================================================================================================================
# TSC XY Plot: Denoise Values
class TSC_XYplot_Denoise:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "first_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "last_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, batch_count, first_denoise, last_denoise):
        xy_type = "Denoise"
        xy_value = generate_floats(batch_count, first_denoise, last_denoise)
        return ((xy_type, xy_value),) if xy_value else (None,)

#=======================================================================================================================
# TSC XY Plot: VAE Values
class TSC_XYplot_VAE:

    modes = ["VAE Names", "VAE Batch"]

    @classmethod
    def INPUT_TYPES(cls):

        vaes = ["None", "Baked VAE"] + folder_paths.get_filename_list("vae")

        inputs = {
            "required": {
                        "input_mode": (cls.modes,),
                        "batch_path": ("STRING", {"default": xy_batch_default_path, "multiline": False}),
                        "subdirectories": ("BOOLEAN", {"default": False}),
                        "batch_sort": (["ascending", "descending"],),
                        "batch_max": ("INT", {"default": -1, "min": -1, "max": XYPLOT_LIM, "step": 1}),
                        "vae_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM, "step": 1})
            }
        }

        for i in range(1, XYPLOT_LIM+1):
            inputs["required"][f"vae_name_{i}"] = (vaes,)

        return inputs

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, input_mode, batch_path, subdirectories, batch_sort, batch_max, vae_count, **kwargs):

        xy_type = "VAE"

        if "Batch" not in input_mode:
            # Extract values from kwargs
            vaes = [kwargs.get(f"vae_name_{i}") for i in range(1, vae_count + 1)]
            xy_value = [vae for vae in vaes if vae != "None"]
        else:
            if batch_max == 0:
                return (None,)

            try:
                vaes = get_batch_files(batch_path, VAE_EXTENSIONS, include_subdirs=subdirectories)

                if not vaes:
                    print(f"{error('XY Plot Error:')} No VAE files found.")
                    return (None,)

                if batch_sort == "ascending":
                    vaes.sort()
                elif batch_sort == "descending":
                    vaes.sort(reverse=True)

                # Construct the xy_value using the obtained vaes
                xy_value = [vae for vae in vaes]

                if batch_max != -1:  # If there's a limit
                    xy_value = xy_value[:batch_max]

            except Exception as e:
                print(f"{error('XY Plot Error:')} {e}")
                return (None,)

        return ((xy_type, xy_value),) if xy_value else (None,)

#=======================================================================================================================
# TSC XY Plot: Prompt S/R
class TSC_XYplot_PromptSR:

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "target_prompt": (["positive", "negative"],),
                "search_txt": ("STRING", {"default": "", "multiline": False}),
                "replace_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM-1}),
            }
        }

        # Dynamically add replace_X inputs
        for i in range(1, XYPLOT_LIM):
            replace_key = f"replace_{i}"
            inputs["required"][replace_key] = ("STRING", {"default": "", "multiline": False})

        return inputs

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, target_prompt, search_txt, replace_count, **kwargs):
        if search_txt == "":
            return (None,)

        if target_prompt == "positive":
            xy_type = "Positive Prompt S/R"
        elif target_prompt == "negative":
            xy_type = "Negative Prompt S/R"

        # Create base entry
        xy_values = [(search_txt, None)]

        if replace_count > 0:
            # Append additional entries based on replace_count
            xy_values.extend([(search_txt, kwargs.get(f"replace_{i+1}")) for i in range(replace_count)])

        return ((xy_type, xy_values),)

#=======================================================================================================================
# TSC XY Plot: Aesthetic Score
class TSC_XYplot_AScore:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_ascore": (["positive", "negative"],),
                "batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "first_ascore": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "last_ascore": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, target_ascore, batch_count, first_ascore, last_ascore):
        if target_ascore == "positive":
            xy_type = "AScore+"
        else:
            xy_type = "AScore-"
        xy_value = generate_floats(batch_count, first_ascore, last_ascore)
        return ((xy_type, xy_value),) if xy_value else (None,)

#=======================================================================================================================
# TSC XY Plot: Refiner On/Off
class TSC_XYplot_Refiner_OnOff:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "refine_at_percent": ("FLOAT",{"default": 0.80, "min": 0.00, "max": 1.00, "step": 0.01})},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, refine_at_percent):
        xy_type = "Refiner On/Off"
        xy_value = [refine_at_percent, 1]
        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: Clip Skip
class TSC_XYplot_ClipSkip:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_ckpt": (["Base","Refiner"],),
                "batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "first_clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                "last_clip_skip": ("INT", {"default": -3, "min": -24, "max": -1, "step": 1}),
            },
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, target_ckpt, batch_count, first_clip_skip, last_clip_skip):
        if target_ckpt == "Base":
            xy_type = "Clip Skip"
        else:
            xy_type = "Clip Skip (Refiner)"
        xy_value = generate_ints(batch_count, first_clip_skip, last_clip_skip)
        return ((xy_type, xy_value),) if xy_value else (None,)

#=======================================================================================================================
# TSC XY Plot: Checkpoint Values
class TSC_XYplot_Checkpoint:
    modes = ["Ckpt Names", "Ckpt Names+ClipSkip", "Ckpt Names+ClipSkip+VAE", "Checkpoint Batch"]
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = ["None"] + folder_paths.get_filename_list("checkpoints")
        vaes = ["Baked VAE"] + folder_paths.get_filename_list("vae")

        inputs = {
            "required": {
                        "target_ckpt": (["Base", "Refiner"],),
                        "input_mode": (cls.modes,),
                        "batch_path": ("STRING", {"default": xy_batch_default_path, "multiline": False}),
                        "subdirectories": ("BOOLEAN", {"default": False}),
                        "batch_sort": (["ascending", "descending"],),
                        "batch_max": ("INT", {"default": -1, "min": -1, "max": 50, "step": 1}),
                        "ckpt_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM, "step": 1})
            }
        }

        for i in range(1, XYPLOT_LIM+1):
            inputs["required"][f"ckpt_name_{i}"] = (checkpoints,)
            inputs["required"][f"clip_skip_{i}"] = ("INT", {"default": -1, "min": -24, "max": -1, "step": 1})
            inputs["required"][f"vae_name_{i}"] = (vaes,)

        return inputs

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, target_ckpt, input_mode, batch_path, subdirectories, batch_sort, batch_max, ckpt_count, **kwargs):

        # Define XY type
        xy_type = "Checkpoint" if target_ckpt == "Base" else "Refiner"

        if "Batch" not in input_mode:
            # Extract values from kwargs
            checkpoints = [kwargs.get(f"ckpt_name_{i}") for i in range(1, ckpt_count + 1)]
            clip_skips = [kwargs.get(f"clip_skip_{i}") for i in range(1, ckpt_count + 1)]
            vaes = [kwargs.get(f"vae_name_{i}") for i in range(1, ckpt_count + 1)]

            # Set None for Clip Skip and/or VAE if not correct modes
            for i in range(ckpt_count):
                if "ClipSkip" not in input_mode:
                    clip_skips[i] = None
                if "VAE" not in input_mode:
                    vaes[i] = None

            xy_value = [(checkpoint, clip_skip, vae) for checkpoint, clip_skip, vae in zip(checkpoints, clip_skips, vaes) if
                        checkpoint != "None"]
        else:
            if batch_max == 0:
                return (None,)

            try:
                ckpts = get_batch_files(batch_path, CKPT_EXTENSIONS, include_subdirs=subdirectories)

                if not ckpts:
                    print(f"{error('XY Plot Error:')} No Checkpoint files found.")
                    return (None,)

                if batch_sort == "ascending":
                    ckpts.sort()
                elif batch_sort == "descending":
                    ckpts.sort(reverse=True)

                # Construct the xy_value using the obtained ckpts
                xy_value = [(ckpt, None, None) for ckpt in ckpts]

                if batch_max != -1:  # If there's a limit
                    xy_value = xy_value[:batch_max]

            except Exception as e:
                print(f"{error('XY Plot Error:')} {e}")
                return (None,)

        return ((xy_type, xy_value),) if xy_value else (None,)

#=======================================================================================================================
# TSC XY Plot: LoRA Batch (DISABLED)
class TSC_XYplot_LoRA_Batch:

    @classmethod
    def INPUT_TYPES(cls):

        return {"required": {
                "batch_path": ("STRING", {"default": xy_batch_default_path, "multiline": False}),
                "subdirectories": ("BOOLEAN", {"default": False}),
                "batch_sort": (["ascending", "descending"],),
                "batch_max": ("INT",{"default": -1, "min": -1, "max": XYPLOT_LIM, "step": 1}),
                "model_strength": ("FLOAT", {"default": 1.0, "min": -10.00, "max": 10.0, "step": 0.01}),
                "clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})},
                "optional": {"lora_stack": ("LORA_STACK",)}
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, batch_path, subdirectories, batch_sort, model_strength, clip_strength, batch_max, lora_stack=None):
        if batch_max == 0:
            return (None,)

        xy_type = "LoRA Stacks"

        loras = get_batch_files(batch_path, LORA_EXTENSIONS, include_subdirs=subdirectories)

        if not loras:
            print(f"{error('XY Plot Error:')} No LoRA files found.")
            return (None,)

        if batch_sort == "ascending":
            loras.sort()
        elif batch_sort == "descending":
            loras.sort(reverse=True)

        # Construct the xy_value using the obtained loras
        xy_value = [[(lora, model_strength, clip_strength)] + (lora_stack if lora_stack else []) for lora in loras]

        if batch_max != -1:  # If there's a limit
            xy_value = xy_value[:batch_max]

        return ((xy_type, xy_value),) if xy_value else (None,)

#=======================================================================================================================
# TSC XY Plot: LoRA Values
class TSC_XYplot_LoRA:
    modes = ["LoRA Names", "LoRA Names+Weights", "LoRA Batch"]

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")

        inputs = {
            "required": {
                "input_mode": (cls.modes,),
                "batch_path": ("STRING", {"default": xy_batch_default_path, "multiline": False}),
                "subdirectories": ("BOOLEAN", {"default": False}),
                "batch_sort": (["ascending", "descending"],),
                "batch_max": ("INT", {"default": -1, "min": -1, "max": XYPLOT_LIM, "step": 1}),
                "lora_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM, "step": 1}),
                "model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

        for i in range(1, XYPLOT_LIM+1):
            inputs["required"][f"lora_name_{i}"] = (loras,)
            inputs["required"][f"model_str_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"clip_str_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        inputs["optional"] = {
            "lora_stack": ("LORA_STACK",)
        }
        return inputs

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def __init__(self):
        self.lora_batch = TSC_XYplot_LoRA_Batch()

    def xy_value(self, input_mode, batch_path, subdirectories, batch_sort, batch_max, lora_count, model_strength,
                 clip_strength, lora_stack=None, **kwargs):

        xy_type = "LoRA"
        result = (None,)
        lora_stack = lora_stack if lora_stack else []

        if "Batch" not in input_mode:
            # Extract values from kwargs
            loras = [kwargs.get(f"lora_name_{i}") for i in range(1, lora_count + 1)]
            model_strs = [kwargs.get(f"model_str_{i}", model_strength) for i in range(1, lora_count + 1)]
            clip_strs = [kwargs.get(f"clip_str_{i}", clip_strength) for i in range(1, lora_count + 1)]

            # Use model_strength and clip_strength for the loras where values are not provided
            if "Weights" not in input_mode:
                for i in range(lora_count):
                    model_strs[i] = model_strength
                    clip_strs[i] = clip_strength

            # Extend each sub-array with lora_stack if it's not None
            xy_value = [[(lora, model_str, clip_str)] + lora_stack for lora, model_str, clip_str
                        in zip(loras, model_strs, clip_strs) if lora != "None"]

            result = ((xy_type, xy_value),)
        else:
            try:
                result = self.lora_batch.xy_value(batch_path, subdirectories, batch_sort, model_strength,
                                                  clip_strength, batch_max, lora_stack)
            except Exception as e:
                print(f"{error('XY Plot Error:')} {e}")

        return result

#=======================================================================================================================
# TSC XY Plot: LoRA Plot
class TSC_XYplot_LoRA_Plot:

    modes = ["X: LoRA Batch, Y: LoRA Weight",
             "X: LoRA Batch, Y: Model Strength",
             "X: LoRA Batch, Y: Clip Strength",
             "X: Model Strength, Y: Clip Strength",
            ]

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {"required": {
                "input_mode": (cls.modes,),
                "lora_name": (loras,),
                "model_strength": ("FLOAT", {"default": 1.0, "min": -10.00, "max": 10.0, "step": 0.01}),
                "clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "X_batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "X_batch_path": ("STRING", {"default": xy_batch_default_path, "multiline": False}),
                "X_subdirectories": ("BOOLEAN", {"default": False}),
                "X_batch_sort": (["ascending", "descending"],),
                "X_first_value": ("FLOAT", {"default": 0.0, "min": -10.00, "max": 10.0, "step": 0.01}),
                "X_last_value": ("FLOAT", {"default": 1.0, "min": -10.00, "max": 10.0, "step": 0.01}),
                "Y_batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "Y_first_value": ("FLOAT", {"default": 0.0, "min": -10.00, "max": 10.0, "step": 0.01}),
                "Y_last_value": ("FLOAT", {"default": 1.0, "min": -10.00, "max": 10.0, "step": 0.01}),},
            "optional": {"lora_stack": ("LORA_STACK",)}
        }

    RETURN_TYPES = ("XY","XY",)
    RETURN_NAMES = ("X","Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def __init__(self):
        self.lora_batch = TSC_XYplot_LoRA_Batch()

    def generate_values(self, mode, X_or_Y, *args, **kwargs):
        result = self.lora_batch.xy_value(*args, **kwargs)

        if result and result[0]:
            xy_type, xy_value_list = result[0]

            # Adjust type based on the mode
            if "LoRA Weight" in mode:
                xy_type = "LoRA Wt"
            elif "Model Strength" in mode:
                xy_type = "LoRA MStr"
            elif "Clip Strength" in mode:
                xy_type = "LoRA CStr"

            # Check whether the value is for X or Y
            if "LoRA Batch" in mode: # Changed condition
                return self.generate_batch_values(*args, **kwargs)
            else:
                return ((xy_type, xy_value_list),)

        return (None,)

    def xy_value(self, input_mode, lora_name, model_strength, clip_strength, X_batch_count, X_batch_path, X_subdirectories,
                 X_batch_sort, X_first_value, X_last_value, Y_batch_count, Y_first_value, Y_last_value, lora_stack=None):

        x_value, y_value = [], []
        lora_stack = lora_stack if lora_stack else []

        if "Model Strength" in input_mode and "Clip Strength" in input_mode:
            if lora_name == 'None':
                return (None,None,)
        if "LoRA Batch" in input_mode:
            lora_name = None
        if "LoRA Weight" in input_mode:
            model_strength = None
            clip_strength = None
        if "Model Strength" in input_mode:
            model_strength = None
        if "Clip Strength" in input_mode:
            clip_strength = None

        # Handling X values
        if "X: LoRA Batch" in input_mode:
            try:
                x_value = self.lora_batch.xy_value(X_batch_path, X_subdirectories, X_batch_sort,
                                                   model_strength, clip_strength, X_batch_count, lora_stack)[0][1]
            except Exception as e:
                print(f"{error('XY Plot Error:')} {e}")
                return (None,)
            x_type = "LoRA Batch"
        elif "X: Model Strength" in input_mode:
            x_floats = generate_floats(X_batch_count, X_first_value, X_last_value)
            x_type = "LoRA MStr"
            x_value = [[(lora_name, x, clip_strength)] + lora_stack for x in x_floats]

        # Handling Y values
        y_floats = generate_floats(Y_batch_count, Y_first_value, Y_last_value)
        if "Y: LoRA Weight" in input_mode:
            y_type = "LoRA Wt"
            y_value = [[(lora_name, y, y)] + lora_stack for y in y_floats]
        elif "Y: Model Strength" in input_mode:
            y_type = "LoRA MStr"
            y_value = [[(lora_name, y, clip_strength)] + lora_stack for y in y_floats]
        elif "Y: Clip Strength" in input_mode:
            y_type = "LoRA CStr"
            y_value = [[(lora_name, model_strength, y)] + lora_stack for y in y_floats]

        return ((x_type, x_value), (y_type, y_value))

#=======================================================================================================================
# TSC XY Plot: LoRA Stacks
class TSC_XYplot_LoRA_Stacks:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "node_state": (["Enabled"],)},
                "optional": {
                    "lora_stack_1": ("LORA_STACK",),
                    "lora_stack_2": ("LORA_STACK",),
                    "lora_stack_3": ("LORA_STACK",),
                    "lora_stack_4": ("LORA_STACK",),
                    "lora_stack_5": ("LORA_STACK",),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, node_state, lora_stack_1=None, lora_stack_2=None, lora_stack_3=None, lora_stack_4=None, lora_stack_5=None):
        xy_type = "LoRA Stacks"
        xy_value = [stack for stack in [lora_stack_1, lora_stack_2, lora_stack_3, lora_stack_4, lora_stack_5] if stack is not None]
        if not xy_value or not any(xy_value) or node_state == "Disabled":
            return (None,)
        else:
            return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: Control Net Strength (DISABLED)
class TSC_XYplot_Control_Net_Strength:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "first_strength": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "last_strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            },
            "optional": {"cnet_stack": ("CONTROL_NET_STACK",)},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, control_net, image, batch_count, first_strength, last_strength,
                 start_percent, end_percent, cnet_stack=None):

        if batch_count == 0:
            return (None,)

        xy_type = "ControlNetStrength"
        strength_increment = (last_strength - first_strength) / (batch_count - 1) if batch_count > 1 else 0

        xy_value = []

        # Always add the first strength.
        xy_value.append([(control_net, image, first_strength, start_percent, end_percent)])

        # Add intermediate strengths only if batch_count is more than 2.
        for i in range(1, batch_count - 1):
            xy_value.append([(control_net, image, first_strength + i * strength_increment, start_percent,
                              end_percent)])

        # Always add the last strength if batch_count is more than 1.
        if batch_count > 1:
            xy_value.append([(control_net, image, last_strength, start_percent, end_percent)])

        # If cnet_stack is provided, extend each inner array with its content
        if cnet_stack:
            for inner_list in xy_value:
                inner_list.extend(cnet_stack)

        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: Control Net Start % (DISABLED)
class TSC_XYplot_Control_Net_Start:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "first_start_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "last_start_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            },
            "optional": {"cnet_stack": ("CONTROL_NET_STACK",)},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, control_net, image, batch_count, first_start_percent, last_start_percent,
                 strength, end_percent, cnet_stack=None):

        if batch_count == 0:
            return (None,)

        xy_type = "ControlNetStart%"
        percent_increment = (last_start_percent - first_start_percent) / (batch_count - 1) if batch_count > 1 else 0

        xy_value = []

        # Always add the first start_percent.
        xy_value.append([(control_net, image, strength, first_start_percent, end_percent)])

        # Add intermediate start percents only if batch_count is more than 2.
        for i in range(1, batch_count - 1):
            xy_value.append([(control_net, image, strength, first_start_percent + i * percent_increment,
                              end_percent)])

        # Always add the last start_percent if batch_count is more than 1.
        if batch_count > 1:
            xy_value.append([(control_net, image, strength, last_start_percent, end_percent)])

        # If cnet_stack is provided, extend each inner array with its content
        if cnet_stack:
            for inner_list in xy_value:
                inner_list.extend(cnet_stack)

        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: Control Net End % (DISABLED)
class TSC_XYplot_Control_Net_End:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "first_end_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "last_end_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            },
            "optional": {"cnet_stack": ("CONTROL_NET_STACK",)},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, control_net, image, batch_count, first_end_percent, last_end_percent,
                 strength, start_percent, cnet_stack=None):

        if batch_count == 0:
            return (None,)

        xy_type = "ControlNetEnd%"
        percent_increment = (last_end_percent - first_end_percent) / (batch_count - 1) if batch_count > 1 else 0

        xy_value = []

        # Always add the first end_percent.
        xy_value.append([(control_net, image, strength, start_percent, first_end_percent)])

        # Add intermediate end percents only if batch_count is more than 2.
        for i in range(1, batch_count - 1):
            xy_value.append([(control_net, image, strength, start_percent,
                              first_end_percent + i * percent_increment)])

        # Always add the last end_percent if batch_count is more than 1.
        if batch_count > 1:
            xy_value.append([(control_net, image, strength, start_percent, last_end_percent)])

        # If cnet_stack is provided, extend each inner array with its content
        if cnet_stack:
            for inner_list in xy_value:
                inner_list.extend(cnet_stack)

        return ((xy_type, xy_value),)


# =======================================================================================================================
# TSC XY Plot: Control Net
class TSC_XYplot_Control_Net(TSC_XYplot_Control_Net_Strength, TSC_XYplot_Control_Net_Start, TSC_XYplot_Control_Net_End):
    parameters = ["strength", "start_percent", "end_percent"]
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "target_parameter": (cls.parameters,),
                "batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "first_strength": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "last_strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "first_start_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "last_start_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "first_end_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "last_end_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            },
            "optional": {"cnet_stack": ("CONTROL_NET_STACK",)},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, control_net, image, target_parameter, batch_count, first_strength, last_strength, first_start_percent,
                 last_start_percent, first_end_percent, last_end_percent, strength, start_percent, end_percent, cnet_stack=None):

        if target_parameter == "strength":
            return TSC_XYplot_Control_Net_Strength.xy_value(self, control_net, image, batch_count, first_strength,
                                                            last_strength, start_percent, end_percent, cnet_stack=cnet_stack)
        elif target_parameter == "start_percent":
            return TSC_XYplot_Control_Net_Start.xy_value(self, control_net, image, batch_count, first_start_percent,
                                                         last_start_percent, strength, end_percent, cnet_stack=cnet_stack)
        elif target_parameter == "end_percent":
            return TSC_XYplot_Control_Net_End.xy_value(self, control_net, image, batch_count, first_end_percent,
                                                       last_end_percent, strength, start_percent, cnet_stack=cnet_stack)

#=======================================================================================================================
# TSC XY Plot: Control Net Plot
class TSC_XYplot_Control_Net_Plot:

    plot_types = ["X: Strength, Y: Start%",
             "X: Strength, Y: End%",
             "X: Start%, Y: Strength",
             "X: Start%, Y: End%",
             "X: End%, Y: Strength",
             "X: End%, Y: Start%"]

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "plot_type": (cls.plot_types,),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "X_batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "X_first_value": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "X_last_value": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "Y_batch_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM}),
                "Y_first_value": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 10.0, "step": 0.01}),
                "Y_last_value": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 10.0, "step": 0.01}),},
            "optional": {"cnet_stack": ("CONTROL_NET_STACK",)},
        }

    RETURN_TYPES = ("XY","XY",)
    RETURN_NAMES = ("X","Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def get_value(self, axis, control_net, image, strength, start_percent, end_percent,
                  batch_count, first_value, last_value):

        # Adjust upper bound for Start% and End% type
        if axis in ["Start%", "End%"]:
            first_value = min(1, first_value)
            last_value = min(1, last_value)

        increment = (last_value - first_value) / (batch_count - 1) if batch_count > 1 else 0

        values = []

        # Always add the first value.
        if axis == "Strength":
            values.append([(control_net, image, first_value, start_percent, end_percent)])
        elif axis == "Start%":
            values.append([(control_net, image, strength, first_value, end_percent)])
        elif axis == "End%":
            values.append([(control_net, image, strength, start_percent, first_value)])

        # Add intermediate values only if batch_count is more than 2.
        for i in range(1, batch_count - 1):
            if axis == "Strength":
                values.append(
                    [(control_net, image, first_value + i * increment, start_percent, end_percent)])
            elif axis == "Start%":
                values.append(
                    [(control_net, image, strength, first_value + i * increment, end_percent)])
            elif axis == "End%":
                values.append(
                    [(control_net, image, strength, start_percent, first_value + i * increment)])

        # Always add the last value if batch_count is more than 1.
        if batch_count > 1:
            if axis == "Strength":
                values.append([(control_net, image, last_value, start_percent, end_percent)])
            elif axis == "Start%":
                values.append([(control_net, image, strength, last_value, end_percent)])
            elif axis == "End%":
                values.append([(control_net, image, strength, start_percent, last_value)])

        return values

    def xy_value(self, control_net, image, strength, start_percent, end_percent, plot_type,
                 X_batch_count, X_first_value, X_last_value, Y_batch_count, Y_first_value, Y_last_value,
                 cnet_stack=None):

        x_type, y_type = plot_type.split(", ")

        # Now split each type by ": "
        x_type = x_type.split(": ")[1].strip()
        y_type = y_type.split(": ")[1].strip()

        x_entry = None
        y_entry = None

        if X_batch_count > 0:
            x_value = self.get_value(x_type, control_net, image, strength, start_percent,
                                     end_percent, X_batch_count, X_first_value, X_last_value)
            # If cnet_stack is provided, extend each inner array with its content
            if cnet_stack:
                for inner_list in x_value:
                    inner_list.extend(cnet_stack)

            x_entry = ("ControlNet" + x_type, x_value)

        if Y_batch_count > 0:
            y_value = self.get_value(y_type, control_net, image, strength, start_percent,
                                     end_percent, Y_batch_count, Y_first_value, Y_last_value)
            # If cnet_stack is provided, extend each inner array with its content
            if cnet_stack:
                for inner_list in y_value:
                    inner_list.extend(cnet_stack)

            y_entry = ("ControlNet" + y_type, y_value)

        return (x_entry, y_entry,)

#=======================================================================================================================
# TSC XY Plot: Manual Entry Notes
class TSC_XYplot_Manual_XY_Entry_Info:

    syntax = "(X/Y_types)     (X/Y_values)\n" \
               "Seeds++ Batch   batch_count\n" \
               "Steps           steps_1;steps_2;...\n" \
               "StartStep       start_step_1;start_step_2;...\n" \
               "EndStep         end_step_1;end_step_2;...\n" \
               "CFG Scale       cfg_1;cfg_2;...\n" \
               "Sampler(1)      sampler_1;sampler_2;...\n" \
               "Sampler(2)      sampler_1,scheduler_1;...\n" \
               "Sampler(3)      sampler_1;...;,default_scheduler\n" \
               "Scheduler       scheduler_1;scheduler_2;...\n" \
               "Denoise         denoise_1;denoise_2;...\n" \
               "VAE             vae_1;vae_2;vae_3;...\n" \
               "+Prompt S/R     search_txt;replace_1;replace_2;...\n" \
               "-Prompt S/R     search_txt;replace_1;replace_2;...\n" \
               "Checkpoint(1)   ckpt_1;ckpt_2;ckpt_3;...\n" \
               "Checkpoint(2)   ckpt_1,clip_skip_1;...\n" \
               "Checkpoint(3)   ckpt_1;ckpt_2;...;,default_clip_skip\n" \
               "Clip Skip       clip_skip_1;clip_skip_2;...\n" \
               "LoRA(1)         lora_1;lora_2;lora_3;...\n" \
               "LoRA(2)         lora_1;...;,default_model_str,default_clip_str\n" \
               "LoRA(3)         lora_1,model_str_1,clip_str_1;..."

    @classmethod
    def INPUT_TYPES(cls):
        samplers = ";\n".join(comfy.samplers.KSampler.SAMPLERS)
        schedulers = ";\n".join(SCHEDULERS)
        vaes = ";\n".join(folder_paths.get_filename_list("vae"))
        ckpts = ";\n".join(folder_paths.get_filename_list("checkpoints"))
        loras = ";\n".join(folder_paths.get_filename_list("loras"))
        return {"required": {
            "notes": ("STRING", {"default":
                                    f"_____________SYNTAX_____________\n{cls.syntax}\n\n"
                                    f"____________SAMPLERS____________\n{samplers}\n\n"
                                    f"___________SCHEDULERS___________\n{schedulers}\n\n"
                                    f"_____________VAES_______________\n{vaes}\n\n"
                                    f"___________CHECKPOINTS__________\n{ckpts}\n\n"
                                    f"_____________LORAS______________\n{loras}\n","multiline": True}),},}

    RETURN_TYPES = ()
    CATEGORY = "Efficiency Nodes/XY Inputs"


#=======================================================================================================================
# TSC XY Plot: Manual Entry
class TSC_XYplot_Manual_XY_Entry:

    plot_types = ["Nothing", "Seeds++ Batch", "Steps", "StartStep", "EndStep", "CFG Scale", "Sampler", "Scheduler",
                  "Denoise", "VAE", "Positive Prompt S/R", "Negative Prompt S/R", "Checkpoint", "Clip Skip", "LoRA"]
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plot_type": (cls.plot_types,),
            "plot_value": ("STRING", {"default": "", "multiline": True}),}
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, plot_type, plot_value):

        # Store X values as arrays
        if plot_type not in {"Positive Prompt S/R", "Negative Prompt S/R", "VAE", "Checkpoint", "LoRA", "Scheduler"}:
            plot_value = plot_value.replace(" ", "")  # Remove spaces
        plot_value = plot_value.replace("\n", "")  # Remove newline characters
        plot_value = plot_value.rstrip(";")  # Remove trailing semicolon
        plot_value = plot_value.split(";")  # Turn to array

        # Define the valid bounds for each type
        bounds = {
            "Seeds++ Batch": {"min": 1, "max": 50},
            "Steps": {"min": 1, "max": 10000},
            "StartStep": {"min": 0, "max": 10000},
            "EndStep": {"min": 0, "max": 10000},
            "CFG Scale": {"min": 0, "max": 100},
            "Sampler": {"options": comfy.samplers.KSampler.SAMPLERS},
            "Scheduler": {"options": SCHEDULERS},
            "Denoise": {"min": 0, "max": 1},
            "VAE": {"options": folder_paths.get_filename_list("vae")},
            "Checkpoint": {"options": folder_paths.get_filename_list("checkpoints")},
            "Clip Skip": {"min": -24, "max": -1},
            "LoRA": {"options": folder_paths.get_filename_list("loras"),
                     "model_str": {"min": -10, "max": 10},"clip_str": {"min": -10, "max": 10},},
        }

        # Validates a value based on its corresponding value_type and bounds.
        def validate_value(value, value_type, bounds):
            # ________________________________________________________________________
            # Seeds++ Batch
            if value_type == "Seeds++ Batch":
                try:
                    x = int(float(value))
                    if x < bounds["Seeds++ Batch"]["min"]:
                        x = bounds["Seeds++ Batch"]["min"]
                    elif x > bounds["Seeds++ Batch"]["max"]:
                        x = bounds["Seeds++ Batch"]["max"]
                except ValueError:
                    print(f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid batch count.")
                    return None
                if float(value) != x:
                    print(f"\033[31mmXY Plot Error:\033[0m '{value}' is not a valid batch count.")
                    return None
                return x
            # ________________________________________________________________________
            # Steps
            elif value_type == "Steps":
                try:
                    x = int(value)
                    if x < bounds["Steps"]["min"]:
                        x = bounds["Steps"]["min"]
                    elif x > bounds["Steps"]["max"]:
                        x = bounds["Steps"]["max"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Step count.")
                    return None
            # __________________________________________________________________________________________________________
            # Start at Step
            elif value_type == "StartStep":
                try:
                    x = int(value)
                    if x < bounds["StartStep"]["min"]:
                        x = bounds["StartStep"]["min"]
                    elif x > bounds["StartStep"]["max"]:
                        x = bounds["StartStep"]["max"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Start Step.")
                    return None
            # __________________________________________________________________________________________________________
            # End at Step
            elif value_type == "EndStep":
                try:
                    x = int(value)
                    if x < bounds["EndStep"]["min"]:
                        x = bounds["EndStep"]["min"]
                    elif x > bounds["EndStep"]["max"]:
                        x = bounds["EndStep"]["max"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid End Step.")
                    return None
            # __________________________________________________________________________________________________________
            # CFG Scale
            elif value_type == "CFG Scale":
                try:
                    x = float(value)
                    if x < bounds["CFG Scale"]["min"]:
                        x = bounds["CFG Scale"]["min"]
                    elif x > bounds["CFG Scale"]["max"]:
                        x = bounds["CFG Scale"]["max"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a number between {bounds['CFG Scale']['min']}"
                        f" and {bounds['CFG Scale']['max']} for CFG Scale.")
                    return None
            # __________________________________________________________________________________________________________
            # Sampler
            elif value_type == "Sampler":
                if isinstance(value, str) and ',' in value:
                    value = tuple(map(str.strip, value.split(',')))
                if isinstance(value, tuple):
                    if len(value) >= 2:
                        value = value[:2]  # Slice the value tuple to keep only the first two elements
                        sampler, scheduler = value
                        scheduler = scheduler.lower()  # Convert the scheduler name to lowercase
                        if sampler not in bounds["Sampler"]["options"]:
                            valid_samplers = '\n'.join(bounds["Sampler"]["options"])
                            print(
                                f"\033[31mXY Plot Error:\033[0m '{sampler}' is not a valid sampler. Valid samplers are:\n{valid_samplers}")
                            sampler = None
                        if scheduler not in bounds["Scheduler"]["options"]:
                            valid_schedulers = '\n'.join(bounds["Scheduler"]["options"])
                            print(
                                f"\033[31mXY Plot Error:\033[0m '{scheduler}' is not a valid scheduler. Valid schedulers are:\n{valid_schedulers}")
                            scheduler = None
                        if sampler is None or scheduler is None:
                            return None
                        else:
                            return sampler, scheduler
                    else:
                        print(
                            f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid sampler.'")
                        return None
                else:
                    if value not in bounds["Sampler"]["options"]:
                        valid_samplers = '\n'.join(bounds["Sampler"]["options"])
                        print(
                            f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid sampler. Valid samplers are:\n{valid_samplers}")
                        return None
                    else:
                        return value, None
            # __________________________________________________________________________________________________________
            # Scheduler
            elif value_type == "Scheduler":
                if value not in bounds["Scheduler"]["options"]:
                    valid_schedulers = '\n'.join(bounds["Scheduler"]["options"])
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Scheduler. Valid Schedulers are:\n{valid_schedulers}")
                    return None
                else:
                    return value
            # __________________________________________________________________________________________________________
            # Denoise
            elif value_type == "Denoise":
                try:
                    x = float(value)
                    if x < bounds["Denoise"]["min"]:
                        x = bounds["Denoise"]["min"]
                    elif x > bounds["Denoise"]["max"]:
                        x = bounds["Denoise"]["max"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a number between {bounds['Denoise']['min']} "
                        f"and {bounds['Denoise']['max']} for Denoise.")
                    return None
            # __________________________________________________________________________________________________________
            # VAE
            elif value_type == "VAE":
                if value not in bounds["VAE"]["options"]:
                    valid_vaes = '\n'.join(bounds["VAE"]["options"])
                    print(f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid VAE. Valid VAEs are:\n{valid_vaes}")
                    return None
                else:
                    return value
            # __________________________________________________________________________________________________________
            # Checkpoint
            elif value_type == "Checkpoint":
                if isinstance(value, str) and ',' in value:
                    value = tuple(map(str.strip, value.split(',')))
                if isinstance(value, tuple):
                    if len(value) >= 2:
                        value = value[:2]  # Slice the value tuple to keep only the first two elements
                        checkpoint, clip_skip = value
                        try:
                            clip_skip = int(clip_skip)  # Convert the clip_skip to integer
                        except ValueError:
                            print(f"\033[31mXY Plot Error:\033[0m '{clip_skip}' is not a valid clip_skip. "
                                  f"Valid clip skip values are integers between {bounds['Clip Skip']['min']} and {bounds['Clip Skip']['max']}.")
                            return None
                        if checkpoint not in bounds["Checkpoint"]["options"]:
                            valid_checkpoints = '\n'.join(bounds["Checkpoint"]["options"])
                            print(
                                f"\033[31mXY Plot Error:\033[0m '{checkpoint}' is not a valid checkpoint. Valid checkpoints are:\n{valid_checkpoints}")
                            checkpoint = None
                        if clip_skip < bounds["Clip Skip"]["min"] or clip_skip > bounds["Clip Skip"]["max"]:
                            print(f"\033[31mXY Plot Error:\033[0m '{clip_skip}' is not a valid clip skip. "
                                  f"Valid clip skip values are integers between {bounds['Clip Skip']['min']} and {bounds['Clip Skip']['max']}.")
                            clip_skip = None
                        if checkpoint is None or clip_skip is None:
                            return None
                        else:
                            return checkpoint, clip_skip, None
                    else:
                        print(
                            f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid checkpoint.'")
                        return None
                else:
                    if value not in bounds["Checkpoint"]["options"]:
                        valid_checkpoints = '\n'.join(bounds["Checkpoint"]["options"])
                        print(
                            f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid checkpoint. Valid checkpoints are:\n{valid_checkpoints}")
                        return None
                    else:
                        return value, None, None
            # __________________________________________________________________________________________________________
            # Clip Skip
            elif value_type == "Clip Skip":
                try:
                    x = int(value)
                    if x < bounds["Clip Skip"]["min"]:
                        x = bounds["Clip Skip"]["min"]
                    elif x > bounds["Clip Skip"]["max"]:
                        x = bounds["Clip Skip"]["max"]
                    return x
                except ValueError:
                    print(f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Clip Skip.")
                    return None
            # __________________________________________________________________________________________________________
            # LoRA
            elif value_type == "LoRA":
                if isinstance(value, str) and ',' in value:
                    value = tuple(map(str.strip, value.split(',')))

                if isinstance(value, tuple):
                    lora_name, model_str, clip_str = (value + (1.0, 1.0))[:3]  # Defaults model_str and clip_str to 1 if not provided

                    if lora_name not in bounds["LoRA"]["options"]:
                        valid_loras = '\n'.join(bounds["LoRA"]["options"])
                        print(f"{error('XY Plot Error:')} '{lora_name}' is not a valid LoRA. Valid LoRAs are:\n{valid_loras}")
                        lora_name = None

                    try:
                        model_str = float(model_str)
                        clip_str = float(clip_str)
                    except ValueError:
                        print(f"{error('XY Plot Error:')} The LoRA model strength and clip strength values should be numbers"
                            f" between {bounds['LoRA']['model_str']['min']} and {bounds['LoRA']['model_str']['max']}.")
                        return None

                    if model_str < bounds["LoRA"]["model_str"]["min"] or model_str > bounds["LoRA"]["model_str"]["max"]:
                        print(f"{error('XY Plot Error:')} '{model_str}' is not a valid LoRA model strength value. "
                              f"Valid lora model strength values are between {bounds['LoRA']['model_str']['min']} and {bounds['LoRA']['model_str']['max']}.")
                        model_str = None

                    if clip_str < bounds["LoRA"]["clip_str"]["min"] or clip_str > bounds["LoRA"]["clip_str"]["max"]:
                        print(f"{error('XY Plot Error:')} '{clip_str}' is not a valid LoRA clip strength value. "
                              f"Valid lora clip strength values are between {bounds['LoRA']['clip_str']['min']} and {bounds['LoRA']['clip_str']['max']}.")
                        clip_str = None

                    if lora_name is None or model_str is None or clip_str is None:
                        return None
                    else:
                        return lora_name, model_str, clip_str
                else:
                    if value not in bounds["LoRA"]["options"]:
                        valid_loras = '\n'.join(bounds["LoRA"]["options"])
                        print(f"{error('XY Plot Error:')} '{value}' is not a valid LoRA. Valid LoRAs are:\n{valid_loras}")
                        return None
                    else:
                        return value, 1.0, 1.0

            # __________________________________________________________________________________________________________
            else:
                return None

        # Validate plot_value array length is 1 if doing a "Seeds++ Batch"
        if len(plot_value) != 1 and plot_type == "Seeds++ Batch":
            print(f"{error('XY Plot Error:')} '{';'.join(plot_value)}' is not a valid batch count.")
            return (None,)

        # Apply allowed shortcut syntax to certain input types
        if plot_type in ["Sampler", "Checkpoint", "LoRA"]:
            if plot_value[-1].startswith(','):
                # Remove the leading comma from the last entry and store it as suffixes
                suffixes = plot_value.pop().lstrip(',').split(',')
                # Split all preceding entries into subentries
                plot_value = [entry.split(',') for entry in plot_value]
                # Make all entries the same length as suffixes by appending missing elements
                for entry in plot_value:
                    entry += suffixes[len(entry) - 1:]
                # Join subentries back into strings
                plot_value = [','.join(entry) for entry in plot_value]

        # Prompt S/R X Cleanup
        if plot_type in {"Positive Prompt S/R", "Negative Prompt S/R"}:
            if plot_value[0] == '':
                print(f"{error('XY Plot Error:')} Prompt S/R value can not be empty.")
                return (None,)
            else:
                plot_value = [(plot_value[0], None) if i == 0 else (plot_value[0], x) for i, x in enumerate(plot_value)]

        # Loop over each entry in plot_value and check if it's valid
        if plot_type not in {"Nothing", "Positive Prompt S/R", "Negative Prompt S/R"}:
            for i in range(len(plot_value)):
                plot_value[i] = validate_value(plot_value[i], plot_type, bounds)
                if plot_value[i] == None:
                    return (None,)

        # Set up Seeds++ Batch structure
        if plot_type == "Seeds++ Batch":
            plot_value = list(range(plot_value[0]))

        # Nest LoRA value in another array to reflect LoRA stack changes
        if plot_type == "LoRA":
            plot_value = [[x] for x in plot_value]

        # Clean X/Y_values
        if plot_type == "Nothing":
            plot_value = [""]

        return ((plot_type, plot_value),)

#=======================================================================================================================
# TSC XY Plot: Join Inputs
class TSC_XYplot_JoinInputs:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "XY_1": ("XY",),
            "XY_2": ("XY",),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, XY_1, XY_2):
        xy_type_1, xy_value_1 = XY_1
        xy_type_2, xy_value_2 = XY_2

        if xy_type_1 != xy_type_2:
            print(f"{error('Join XY Inputs Error:')} Input types must match")
            return (None,)
        elif xy_type_1 == "Seeds++ Batch":
            xy_type = xy_type_1
            combined_length = len(xy_value_1) + len(xy_value_2)
            xy_value = list(range(combined_length))
        elif xy_type_1 == "Positive Prompt S/R" or xy_type_1 == "Negative Prompt S/R":
            xy_type = xy_type_1
            xy_value = xy_value_1 + [(xy_value_1[0][0], t[1]) for t in xy_value_2[1:]]
        else:
            xy_type = xy_type_1
            xy_value = xy_value_1 + xy_value_2
        return ((xy_type, xy_value),)

########################################################################################################################
# TSC Image Overlay
class TSC_ImageOverlay:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "overlay_resize": (["None", "Fit", "Resize by rescale_factor", "Resize to width & heigth"],),
                "resize_method": (["nearest-exact", "bilinear", "area"],),
                "rescale_factor": ("FLOAT", {"default": 1, "min": 0.01, "max": 16.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "x_offset": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                "rotation": ("INT", {"default": 0, "min": -180, "max": 180, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 0.1}),
            },
            "optional": {"optional_mask": ("MASK",),}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay_image"
    CATEGORY = "Efficiency Nodes/Image"

    def apply_overlay_image(self, base_image, overlay_image, overlay_resize, resize_method, rescale_factor,
                            width, height, x_offset, y_offset, rotation, opacity, optional_mask=None):

        # Pack tuples and assign variables
        size = width, height
        location = x_offset, y_offset
        mask = optional_mask

        # Check for different sizing options
        if overlay_resize != "None":
            #Extract overlay_image size and store in Tuple "overlay_image_size" (WxH)
            overlay_image_size = overlay_image.size()
            overlay_image_size = (overlay_image_size[2], overlay_image_size[1])
            if overlay_resize == "Fit":
                h_ratio = base_image.size()[1] / overlay_image_size[1]
                w_ratio = base_image.size()[2] / overlay_image_size[0]
                ratio = min(h_ratio, w_ratio)
                overlay_image_size = tuple(round(dimension * ratio) for dimension in overlay_image_size)
            elif overlay_resize == "Resize by rescale_factor":
                overlay_image_size = tuple(int(dimension * rescale_factor) for dimension in overlay_image_size)
            elif overlay_resize == "Resize to width & heigth":
                overlay_image_size = (size[0], size[1])

            samples = overlay_image.movedim(-1, 1)
            overlay_image = comfy.utils.common_upscale(samples, overlay_image_size[0], overlay_image_size[1], resize_method, False)
            overlay_image = overlay_image.movedim(1, -1)
            
        overlay_image = tensor2pil(overlay_image)

         # Add Alpha channel to overlay
        overlay_image = overlay_image.convert('RGBA')
        overlay_image.putalpha(Image.new("L", overlay_image.size, 255))

        # If mask connected, check if the overlay_image image has an alpha channel
        if mask is not None:
            # Convert mask to pil and resize
            mask = tensor2pil(mask)
            mask = mask.resize(overlay_image.size)
            # Apply mask as overlay's alpha
            overlay_image.putalpha(ImageOps.invert(mask))

        # Rotate the overlay image with antialiasing
        overlay_image = overlay_image.rotate(rotation, resample=Image.BICUBIC, expand=True)

        # Apply opacity on overlay image
        r, g, b, a = overlay_image.split()
        a = a.point(lambda x: max(0, int(x * (1 - opacity / 100))))
        overlay_image.putalpha(a)

        # Split the base_image tensor along the first dimension to get a list of tensors
        base_image_list = torch.unbind(base_image, dim=0)

        # Convert each tensor to a PIL image, apply the overlay, and then convert it back to a tensor
        processed_base_image_list = []
        for tensor in base_image_list:
            # Convert tensor to PIL Image
            image = tensor2pil(tensor)

            # Create a new blank image with an alpha channel
            combined = Image.new('RGBA', image.size, (0, 0, 0, 0))

            # Paste the base image onto the new image
            combined.paste(image, (0, 0))

            # Paste the overlay image onto the combined image using alpha compositing
            combined.alpha_composite(overlay_image, location)

            # Convert the combined image back to RGB mode
            result = combined.convert('RGB')

            # Convert PIL Image back to tensor
            processed_tensor = pil2tensor(result)

            # Append to list
            processed_base_image_list.append(processed_tensor)

        # Combine the processed images back into a single tensor
        base_image = torch.stack([tensor.squeeze() for tensor in processed_base_image_list])

        # Return the edited base image
        return (base_image,)

########################################################################################################################
# Noise Sources & Seed Variations
# https://github.com/shiimizu/ComfyUI_smZNodes
# https://github.com/chrisgoringe/cg-noise

# TSC Noise Sources & Variations Script
class TSC_Noise_Control_Script:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rng_source": (["cpu", "gpu", "nv"],),
                "cfg_denoiser": ("BOOLEAN", {"default": False}),
                "add_seed_noise": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "weight": ("FLOAT", {"default": 0.015, "min": 0, "max": 1, "step": 0.001})},
            "optional": {"script": ("SCRIPT",)}
        }

    RETURN_TYPES = ("SCRIPT",)
    RETURN_NAMES = ("SCRIPT",)
    FUNCTION = "noise_control"
    CATEGORY = "Efficiency Nodes/Scripts"

    def noise_control(self, rng_source, cfg_denoiser, add_seed_noise, seed, weight, script=None):
        script = script or {}
        script["noise"] = (rng_source, cfg_denoiser, add_seed_noise, seed, weight)
        return (script,)

########################################################################################################################
# Add controlnet options if have controlnet_aux installed (https://github.com/Fannovel16/comfyui_controlnet_aux)
use_controlnet_widget = preprocessor_widget = (["_"],)
if os.path.exists(os.path.join(custom_nodes_dir, "comfyui_controlnet_aux")):
    printout = "Attempting to add Control Net options to the 'HiRes-Fix Script' Node (comfyui_controlnet_aux add-on)..."
    #print(f"{message('Efficiency Nodes:')} {printout}", end="", flush=True)

    try:
        with suppress_output():
            AIO_Preprocessor = getattr(import_module("comfyui_controlnet_aux.__init__"), 'AIO_Preprocessor')
        use_controlnet_widget = ("BOOLEAN", {"default": False})
        preprocessor_widget = AIO_Preprocessor.INPUT_TYPES()["optional"]["preprocessor"]
        print(f"\r{message('Efficiency Nodes:')} {printout}{success('Success!')}")
    except Exception:
        print(f"\r{message('Efficiency Nodes:')} {printout}{error('Failed!')}")

# TSC HighRes-Fix with model latent upscalers (https://github.com/city96/SD-Latent-Upscaler)
class TSC_HighRes_Fix:

    default_latent_upscalers = LatentUpscaleBy.INPUT_TYPES()["required"]["upscale_method"][0]

    city96_upscale_methods =\
        ["city96." + ver for ver in city96_latent_upscaler.LatentUpscaler.INPUT_TYPES()["required"]["latent_ver"][0]]
    city96_scalings_raw = city96_latent_upscaler.LatentUpscaler.INPUT_TYPES()["required"]["scale_factor"][0]
    city96_scalings_float = [float(scale) for scale in city96_scalings_raw]

    ttl_nn_upscale_methods = \
        ["ttl_nn." + ver for ver in
         ttl_nn_latent_upscaler.NNLatentUpscale.INPUT_TYPES()["required"]["version"][0]]

    latent_upscalers = default_latent_upscalers + city96_upscale_methods + ttl_nn_upscale_methods
    pixel_upscalers = folder_paths.get_filename_list("upscale_models")

    @classmethod
    def INPUT_TYPES(cls):

        return {"required": {"upscale_type": (["latent","pixel","both"],),
                             "hires_ckpt_name": (["(use same)"] + folder_paths.get_filename_list("checkpoints"),),
                             "latent_upscaler": (cls.latent_upscalers,),
                             "pixel_upscaler": (cls.pixel_upscalers,),
                             "upscale_by": ("FLOAT", {"default": 1.25, "min": 0.01, "max": 8.0, "step": 0.05}),
                             "use_same_seed": ("BOOLEAN", {"default": True}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "hires_steps": ("INT", {"default": 12, "min": 1, "max": 10000}),
                             "denoise": ("FLOAT", {"default": .56, "min": 0.00, "max": 1.00, "step": 0.01}),
                             "iterations": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
                             "use_controlnet": use_controlnet_widget,
                             "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "preprocessor": preprocessor_widget,
                             "preprocessor_imgs": ("BOOLEAN", {"default": False})
                             },
                "optional": {"script": ("SCRIPT",)},
                "hidden": {"my_unique_id": "UNIQUE_ID"}
                }

    RETURN_TYPES = ("SCRIPT",)
    FUNCTION = "hires_fix_script"
    CATEGORY = "Efficiency Nodes/Scripts"

    def hires_fix_script(self, upscale_type, hires_ckpt_name, latent_upscaler, pixel_upscaler, upscale_by,
                         use_same_seed, seed, hires_steps, denoise, iterations, use_controlnet, control_net_name,
                         strength, preprocessor, preprocessor_imgs, script=None, my_unique_id=None):
        latent_upscale_function = None
        latent_upscale_model = None
        pixel_upscale_model = None

        def float_to_string(num):
            if num == int(num):
                return "{:.1f}".format(num)
            else:
                return str(num)

        if iterations > 0 and upscale_by > 0:
            if upscale_type == "latent":
                # For latent methods from city96
                if latent_upscaler in self.city96_upscale_methods:
                    # Remove extra characters added
                    latent_upscaler = latent_upscaler.replace("city96.", "")

                    # Set function to city96_latent_upscaler.LatentUpscaler
                    latent_upscale_function = city96_latent_upscaler.LatentUpscaler

                    # Find the nearest valid scaling in city96_scalings_float
                    nearest_scaling = min(self.city96_scalings_float, key=lambda x: abs(x - upscale_by))

                    # Retrieve the index of the nearest scaling
                    nearest_scaling_index = self.city96_scalings_float.index(nearest_scaling)

                    # Use the index to get the raw string representation
                    nearest_scaling_raw = self.city96_scalings_raw[nearest_scaling_index]

                    upscale_by = float_to_string(upscale_by)

                    # Check if the input upscale_by value was different from the nearest valid value
                    if upscale_by != nearest_scaling_raw:
                        print(f"{warning('HighRes-Fix Warning:')} "
                              f"When using 'city96.{latent_upscaler}', 'upscale_by' must be one of {self.city96_scalings_raw}.\n"
                              f"Rounding to the nearest valid value ({nearest_scaling_raw}).\033[0m")
                        upscale_by = nearest_scaling_raw

                # For ttl upscale methods
                elif latent_upscaler in self.ttl_nn_upscale_methods:
                    # Remove extra characters added
                    latent_upscaler = latent_upscaler.replace("ttl_nn.", "")

                    # Bound to min/max limits
                    upscale_by_clamped = min(max(upscale_by, 1), 2)
                    if upscale_by != upscale_by_clamped:
                        print(f"{warning('HighRes-Fix Warning:')} "
                              f"When using 'ttl_nn.{latent_upscaler}', 'upscale_by' must be between 1 and 2.\n"
                              f"Rounding to the nearest valid value ({upscale_by_clamped}).\033[0m")
                    upscale_by = upscale_by_clamped

                    latent_upscale_function = ttl_nn_latent_upscaler.NNLatentUpscale

                # For default upscale methods
                elif latent_upscaler in self.default_latent_upscalers:
                    latent_upscale_function = LatentUpscaleBy

                else: # Default
                    latent_upscale_function = LatentUpscaleBy
                    latent_upscaler = self.default_latent_upscalers[0]
                    print(f"{warning('HiResFix Script Warning:')} Chosen latent upscale method not found! "
                          f"defaulting to '{latent_upscaler}'.\n")

                # Load Checkpoint if defined
                if hires_ckpt_name == "(use same)":
                    clear_cache(my_unique_id, 0, "ckpt")
                else:
                    latent_upscale_model, _, _ = \
                        load_checkpoint(hires_ckpt_name, my_unique_id, output_vae=False, cache=1, cache_overwrite=True)

            elif upscale_type == "pixel":
                pixel_upscale_model = UpscaleModelLoader().load_model(pixel_upscaler)[0]

            elif upscale_type == "both":
                latent_upscale_function = LatentUpscaleBy
                latent_upscaler = self.default_latent_upscalers[0]
                pixel_upscale_model = UpscaleModelLoader().load_model(pixel_upscaler)[0]

                if hires_ckpt_name == "(use same)":
                    clear_cache(my_unique_id, 0, "ckpt")
                else:
                    latent_upscale_model, _, _ = \
                        load_checkpoint(hires_ckpt_name, my_unique_id, output_vae=False, cache=1, cache_overwrite=True)

        control_net = ControlNetLoader().load_controlnet(control_net_name)[0] if use_controlnet is True else None

        # Construct the script output
        script = script or {}
        script["hiresfix"] = (upscale_type, latent_upscaler, upscale_by, use_same_seed, seed, hires_steps,
                              denoise, iterations, control_net, strength, preprocessor, preprocessor_imgs,
                              latent_upscale_function, latent_upscale_model, pixel_upscale_model)

        return (script,)

########################################################################################################################
# TSC Tiled Upscaler (https://github.com/BlenderNeko/ComfyUI_TiledKSampler)
class TSC_Tiled_Upscaler:
    @classmethod
    def INPUT_TYPES(cls):
        # Split the list based on the keyword "tile"
        cnet_filenames = [name for name in folder_paths.get_filename_list("controlnet")]

        return {"required": {"upscale_by": ("FLOAT", {"default": 1.25, "min": 0.01, "max": 8.0, "step": 0.05}),
                             "tile_size": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                             "tiling_strategy": (["random", "random strict", "padded", 'simple', 'none'],),
                             "tiling_steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "denoise": ("FLOAT", {"default": .4, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "use_controlnet": ("BOOLEAN", {"default": False}),
                             "tile_controlnet": (cnet_filenames,),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             },
                "optional": {"script": ("SCRIPT",)}}

    RETURN_TYPES = ("SCRIPT",)
    FUNCTION = "tiled_sampling"
    CATEGORY = "Efficiency Nodes/Scripts"

    def tiled_sampling(self, upscale_by, tile_size, tiling_strategy, tiling_steps, seed, denoise,
                       use_controlnet, tile_controlnet, strength, script=None):
        if tiling_strategy != 'none':
            script = script or {}
            tile_controlnet = ControlNetLoader().load_controlnet(tile_controlnet)[0] if use_controlnet else None

            script["tile"] = (upscale_by, tile_size, tiling_strategy, tiling_steps, seed, denoise, tile_controlnet, strength)
        return (script,)

########################################################################################################################
# TSC LoRA Stack to String converter
class TSC_LoRA_Stack2String:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"lora_stack": ("LORA_STACK",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LoRA string",)
    FUNCTION = "convert"
    CATEGORY = "Efficiency Nodes/Misc"

    def convert(self, lora_stack):
        """
        Converts a list of tuples into a single space-separated string.
        Each tuple contains (STR, FLOAT1, FLOAT2) and is converted to the format "<lora:STR:FLOAT1:FLOAT2>".
        """
        output = ' '.join(f"<lora:{tup[0]}:{tup[1]}:{tup[2]}>" for tup in lora_stack)
        return (output,)

########################################################################################################################
# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "KSampler (Efficient)": TSC_KSampler,
    "KSampler Adv. (Efficient)":TSC_KSamplerAdvanced,
    "KSampler SDXL (Eff.)": TSC_KSamplerSDXL,
    "Efficient Loader": TSC_EfficientLoader,
    "Eff. Loader SDXL": TSC_EfficientLoaderSDXL,
    "LoRA Stacker": TSC_LoRA_Stacker,
    "Control Net Stacker": TSC_Control_Net_Stacker,
    "Apply ControlNet Stack": TSC_Apply_ControlNet_Stack,
    "Unpack SDXL Tuple": TSC_Unpack_SDXL_Tuple,
    "Pack SDXL Tuple": TSC_Pack_SDXL_Tuple,
    "XY Plot": TSC_XYplot,
    "XY Input: Seeds++ Batch": TSC_XYplot_SeedsBatch,
    "XY Input: Add/Return Noise": TSC_XYplot_AddReturnNoise,
    "XY Input: Steps": TSC_XYplot_Steps,
    "XY Input: CFG Scale": TSC_XYplot_CFG,
    "XY Input: Sampler/Scheduler": TSC_XYplot_Sampler_Scheduler,
    "XY Input: Denoise": TSC_XYplot_Denoise,
    "XY Input: VAE": TSC_XYplot_VAE,
    "XY Input: Prompt S/R": TSC_XYplot_PromptSR,
    "XY Input: Aesthetic Score": TSC_XYplot_AScore,
    "XY Input: Refiner On/Off": TSC_XYplot_Refiner_OnOff,
    "XY Input: Checkpoint": TSC_XYplot_Checkpoint,
    "XY Input: Clip Skip": TSC_XYplot_ClipSkip,
    "XY Input: LoRA": TSC_XYplot_LoRA,
    "XY Input: LoRA Plot": TSC_XYplot_LoRA_Plot,
    "XY Input: LoRA Stacks": TSC_XYplot_LoRA_Stacks,
    "XY Input: Control Net": TSC_XYplot_Control_Net,
    "XY Input: Control Net Plot": TSC_XYplot_Control_Net_Plot,
    "XY Input: Manual XY Entry": TSC_XYplot_Manual_XY_Entry,
    "Manual XY Entry Info": TSC_XYplot_Manual_XY_Entry_Info,
    "Join XY Inputs of Same Type": TSC_XYplot_JoinInputs,
    "Image Overlay": TSC_ImageOverlay,
    "Noise Control Script": TSC_Noise_Control_Script,
    "HighRes-Fix Script": TSC_HighRes_Fix,
    "Tiled Upscaler Script": TSC_Tiled_Upscaler,
    "LoRA Stack to String converter": TSC_LoRA_Stack2String
}

########################################################################################################################
# Add AnimateDiff Script based off Kosinkadink's Nodes (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) deprecated
"""
if os.path.exists(os.path.join(custom_nodes_dir, "ComfyUI-AnimateDiff-Evolved")):
    printout = "Attempting to add 'AnimatedDiff Script' Node (ComfyUI-AnimateDiff-Evolved add-on)..."
    print(f"{message('Efficiency Nodes:')} {printout}", end="")
    try:
        module = import_module("ComfyUI-AnimateDiff-Evolved.animatediff.nodes")
        AnimateDiffLoaderWithContext = getattr(module, 'AnimateDiffLoaderWithContext')
        AnimateDiffCombine = getattr(module, 'AnimateDiffCombine_Deprecated')
        print(f"\r{message('Efficiency Nodes:')} {printout}{success('Success!')}")

        # TSC AnimatedDiff Script (https://github.com/BlenderNeko/ComfyUI_TiledKSampler)
        class TSC_AnimateDiff_Script:
            @classmethod
            def INPUT_TYPES(cls):

                return {"required": {
                            "motion_model": AnimateDiffLoaderWithContext.INPUT_TYPES()["required"]["model_name"],
                            "beta_schedule": AnimateDiffLoaderWithContext.INPUT_TYPES()["required"]["beta_schedule"],
                            "frame_rate": AnimateDiffCombine.INPUT_TYPES()["required"]["frame_rate"],
                            "loop_count": AnimateDiffCombine.INPUT_TYPES()["required"]["loop_count"],
                            "format": AnimateDiffCombine.INPUT_TYPES()["required"]["format"],
                            "pingpong": AnimateDiffCombine.INPUT_TYPES()["required"]["pingpong"],
                            "save_image": AnimateDiffCombine.INPUT_TYPES()["required"]["save_image"]},
                        "optional": {"context_options": ("CONTEXT_OPTIONS",)}
                }

            RETURN_TYPES = ("SCRIPT",)
            FUNCTION = "animatediff"
            CATEGORY = "Efficiency Nodes/Scripts"

            def animatediff(self, motion_model, beta_schedule, frame_rate, loop_count, format, pingpong, save_image,
                            script=None, context_options=None):
                script = script or {}
                script["anim"] = (motion_model, beta_schedule, context_options, frame_rate, loop_count, format, pingpong, save_image)
                return (script,)

        NODE_CLASS_MAPPINGS.update({"AnimateDiff Script": TSC_AnimateDiff_Script})

    except Exception:
        print(f"\r{message('Efficiency Nodes:')} {printout}{error('Failed!')}")
        """

########################################################################################################################
# Simpleeval Nodes (https://github.com/danthedeckie/simpleeval)
try:
    import simpleeval

    # TSC Evaluate Integers
    class TSC_EvaluateInts:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "python_expression": ("STRING", {"default": "((a + b) - c) / 2", "multiline": False}),
                "print_to_console": (["False", "True"],), },
                "optional": {
                    "a": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                    "b": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                    "c": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}), },
            }

        RETURN_TYPES = ("INT", "FLOAT", "STRING",)
        OUTPUT_NODE = True
        FUNCTION = "evaluate"
        CATEGORY = "Efficiency Nodes/Simple Eval"

        def evaluate(self, python_expression, print_to_console, a=0, b=0, c=0):
            # simple_eval doesn't require the result to be converted to a string
            result = simpleeval.simple_eval(python_expression, names={'a': a, 'b': b, 'c': c})
            int_result = int(result)
            float_result = float(result)
            string_result = str(result)
            if print_to_console == "True":
                print(f"\n{error('Evaluate Integers:')}")
                print(f"\033[90m{{a = {a} , b = {b} , c = {c}}} \033[0m")
                print(f"{python_expression} = \033[92m INT: " + str(int_result) + " , FLOAT: " + str(
                    float_result) + ", STRING: " + string_result + "\033[0m")
            return (int_result, float_result, string_result,)


    # ==================================================================================================================
    # TSC Evaluate Floats
    class TSC_EvaluateFloats:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "python_expression": ("STRING", {"default": "((a + b) - c) / 2", "multiline": False}),
                "print_to_console": (["False", "True"],), },
                "optional": {
                    "a": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 1}),
                    "b": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 1}),
                    "c": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 1}), },
            }

        RETURN_TYPES = ("INT", "FLOAT", "STRING",)
        OUTPUT_NODE = True
        FUNCTION = "evaluate"
        CATEGORY = "Efficiency Nodes/Simple Eval"

        def evaluate(self, python_expression, print_to_console, a=0, b=0, c=0):
            # simple_eval doesn't require the result to be converted to a string
            result = simpleeval.simple_eval(python_expression, names={'a': a, 'b': b, 'c': c})
            int_result = int(result)
            float_result = float(result)
            string_result = str(result)
            if print_to_console == "True":
                print(f"\n{error('Evaluate Floats:')}")
                print(f"\033[90m{{a = {a} , b = {b} , c = {c}}} \033[0m")
                print(f"{python_expression} = \033[92m INT: " + str(int_result) + " , FLOAT: " + str(
                    float_result) + ", STRING: " + string_result + "\033[0m")
            return (int_result, float_result, string_result,)


    # ==================================================================================================================
    # TSC Evaluate Strings
    class TSC_EvaluateStrs:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "python_expression": ("STRING", {"default": "a + b + c", "multiline": False}),
                "print_to_console": (["False", "True"],)},
                "optional": {
                    "a": ("STRING", {"default": "Hello", "multiline": False}),
                    "b": ("STRING", {"default": " World", "multiline": False}),
                    "c": ("STRING", {"default": "!", "multiline": False}), }
            }

        RETURN_TYPES = ("STRING",)
        OUTPUT_NODE = True
        FUNCTION = "evaluate"
        CATEGORY = "Efficiency Nodes/Simple Eval"

        def evaluate(self, python_expression, print_to_console, a="", b="", c=""):
            variables = {'a': a, 'b': b, 'c': c}  # Define the variables for the expression

            functions = simpleeval.DEFAULT_FUNCTIONS.copy()
            functions.update({"len": len})  # Add the functions for the expression

            result = simpleeval.simple_eval(python_expression, names=variables, functions=functions)
            if print_to_console == "True":
                print(f"\n{error('Evaluate Strings:')}")
                print(f"\033[90ma = {a} \nb = {b} \nc = {c}\033[0m")
                print(f"{python_expression} = \033[92m" + str(result) + "\033[0m")
            return (str(result),)  # Convert result to a string before returning


    # ==================================================================================================================
    # TSC Simple Eval Examples
    class TSC_EvalExamples:
        @classmethod
        def INPUT_TYPES(cls):
            filepath = os.path.join(my_dir, 'workflows', 'SimpleEval_Node_Examples.txt')
            with open(filepath, 'r') as file:
                examples = file.read()
            return {"required": {"models_text": ("STRING", {"default": examples, "multiline": True}), }, }

        RETURN_TYPES = ()
        CATEGORY = "Efficiency Nodes/Simple Eval"

    # ==================================================================================================================
    NODE_CLASS_MAPPINGS.update({"Evaluate Integers": TSC_EvaluateInts})
    NODE_CLASS_MAPPINGS.update({"Evaluate Floats": TSC_EvaluateFloats})
    NODE_CLASS_MAPPINGS.update({"Evaluate Strings": TSC_EvaluateStrs})
    NODE_CLASS_MAPPINGS.update({"Simple Eval Examples": TSC_EvalExamples})

except ImportError:
    print(f"{warning('Efficiency Nodes Warning:')} Failed to import python package 'simpleeval'; related nodes disabled.\n")
