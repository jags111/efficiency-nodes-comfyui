# https://github.com/shiimizu/ComfyUI_smZNodes
import comfy
import torch
from typing import List
import comfy.sample
from comfy import model_base, model_management
from comfy.samplers import KSampler, KSamplerX0Inpaint, wrap_model
#from comfy.k_diffusion.external import CompVisDenoiser
import nodes
import inspect
import functools
import importlib
import os
import re
import itertools

import torch
from comfy import model_management

def catenate_conds(conds):
    if not isinstance(conds[0], dict):
        return torch.cat(conds)

    return {key: torch.cat([x[key] for x in conds]) for key in conds[0].keys()}


def subscript_cond(cond, a, b):
    if not isinstance(cond, dict):
        return cond[a:b]

    return {key: vec[a:b] for key, vec in cond.items()}


def pad_cond(tensor, repeats, empty):
    if not isinstance(tensor, dict):
        return torch.cat([tensor, empty.repeat((tensor.shape[0], repeats, 1)).to(device=tensor.device)], axis=1)

    tensor['crossattn'] = pad_cond(tensor['crossattn'], repeats, empty)
    return tensor


class CFGDenoiser(torch.nn.Module):
    """
    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)
    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)
    instead of one. Originally, the second prompt is just an empty string, but we use non-empty
    negative prompt.
    """

    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.model_wrap = None
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.steps = None
        """number of steps as specified by user in UI"""

        self.total_steps = None
        """expected number of calls to denoiser calculated from self.steps and specifics of the selected sampler"""

        self.step = 0
        self.image_cfg_scale = None
        self.padded_cond_uncond = False
        self.sampler = None
        self.model_wrap = None
        self.p = None
        self.mask_before_denoising = False


    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        denoised = out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)

        return denoised

    def get_pred_x0(self, x_in, x_out, sigma):
        return x_out

    def update_inner_model(self):
        self.model_wrap = None

        c, uc = self.p.get_conds()
        self.sampler.sampler_extra_args['cond'] = c
        self.sampler.sampler_extra_args['uncond'] = uc

    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        model_management.throw_exception_if_processing_interrupted()

        is_edit_model = False

        conds_list, tensor = cond
        assert not is_edit_model or all(len(conds) == 1 for conds in conds_list), "AND is not supported for InstructPix2Pix checkpoint (unless using Image CFG scale = 1.0)"

        if self.mask_before_denoising and self.mask is not None:
            x = self.init_latent * self.mask + self.nmask * x

        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]

        if False:
            image_uncond = torch.zeros_like(image_cond)
            make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": c_crossattn, "c_adm": c_adm, 'transformer_options': {'from_smZ': True}} # pylint: disable=C3001
        else:
            image_uncond = image_cond
            if isinstance(uncond, dict):
                make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": None, "c_adm": x.c_adm, 'transformer_options': {'from_smZ': True}}
            else:
                make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": c_crossattn, "c_concat": None, "c_adm": x.c_adm, 'transformer_options': {'from_smZ': True}}

        if not is_edit_model:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond])
        else:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma] + [sigma])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond] + [torch.zeros_like(self.init_latent)])

        skip_uncond = False

        # alternating uncond allows for higher thresholds without the quality loss normally expected from raising it
        if self.step % 2 and s_min_uncond > 0 and sigma[0] < s_min_uncond and not is_edit_model:
            skip_uncond = True
            x_in = x_in[:-batch_size]
            sigma_in = sigma_in[:-batch_size]

        if tensor.shape[1] == uncond.shape[1] or skip_uncond:
            if is_edit_model:
                cond_in = catenate_conds([tensor, uncond, uncond])
            elif skip_uncond:
                cond_in = tensor
            else:
                cond_in = catenate_conds([tensor, uncond])

            x_out = torch.zeros_like(x_in)
            for batch_offset in range(0, x_out.shape[0], batch_size):
                a = batch_offset
                b = a + batch_size
                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], **make_condition_dict(subscript_cond(cond_in, a, b), image_cond_in[a:b]))
        else:
            x_out = torch.zeros_like(x_in)
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])

                if not is_edit_model:
                    c_crossattn = subscript_cond(tensor, a, b)
                else:
                    c_crossattn = torch.cat([tensor[a:b]], uncond)

                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], **make_condition_dict(c_crossattn, image_cond_in[a:b]))

            if not skip_uncond:
                x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], **make_condition_dict(uncond, image_cond_in[-uncond.shape[0]:]))

        denoised_image_indexes = [x[0][0] for x in conds_list]
        if skip_uncond:
            fake_uncond = torch.cat([x_out[i:i+1] for i in denoised_image_indexes])
            x_out = torch.cat([x_out, fake_uncond])  # we skipped uncond denoising, so we put cond-denoised image to where the uncond-denoised image should be

        if is_edit_model:
            denoised = self.combine_denoised_for_edit_model(x_out, cond_scale)
        elif skip_uncond:
            denoised = self.combine_denoised(x_out, conds_list, uncond, 1.0)
        else:
            denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale)

        if not self.mask_before_denoising and self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        self.step += 1
        del x_out
        return denoised

# ========================================================================

def expand(tensor1, tensor2):
    def adjust_tensor_shape(tensor_small, tensor_big):
        # Calculate replication factor
        # -(-a // b) is ceiling of division without importing math.ceil
        replication_factor = -(-tensor_big.size(1) // tensor_small.size(1))
        
        # Use repeat to extend tensor_small
        tensor_small_extended = tensor_small.repeat(1, replication_factor, 1)
        
        # Take the rows of the extended tensor_small to match tensor_big
        tensor_small_matched = tensor_small_extended[:, :tensor_big.size(1), :]
        
        return tensor_small_matched

    # Check if their second dimensions are different
    if tensor1.size(1) != tensor2.size(1):
        # Check which tensor has the smaller second dimension and adjust its shape
        if tensor1.size(1) < tensor2.size(1):
            tensor1 = adjust_tensor_shape(tensor1, tensor2)
        else:
            tensor2 = adjust_tensor_shape(tensor2, tensor1)
    return (tensor1, tensor2)

def _find_outer_instance(target, target_type):
    import inspect
    frame = inspect.currentframe()
    while frame:
        if target in frame.f_locals:
            found = frame.f_locals[target]
            if isinstance(found, target_type) and found != 1: # steps == 1
                return found
        frame = frame.f_back
    return None

# ========================================================================
def bounded_modulo(number, modulo_value):
    return number if number < modulo_value else modulo_value

def calc_cond(c, current_step):
    """Group by smZ conds that may do prompt-editing / regular conds / comfy conds."""
    _cond = []
    # Group by conds from smZ
    fn=lambda x : x[1].get("from_smZ", None) is not None
    an_iterator = itertools.groupby(c, fn )
    for key, group in an_iterator:
        ls=list(group)
        # Group by prompt-editing conds
        fn2=lambda x : x[1].get("smZid", None)
        an_iterator2 = itertools.groupby(ls, fn2)
        for key2, group2 in an_iterator2:
            ls2=list(group2)
            if key2 is not None:
                orig_len = ls2[0][1].get('orig_len', 1)
                i = bounded_modulo(current_step, orig_len - 1)
                _cond = _cond + [ls2[i]]
            else:
                _cond = _cond + ls2
    return _cond

class CFGNoisePredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ksampler = _find_outer_instance('self', comfy.samplers.KSampler)
        self.step = 0
        self.orig = comfy.samplers.CFGNoisePredictor(model) #CFGNoisePredictorOrig(model)
        self.inner_model = model
        self.inner_model2 = CFGDenoiser(model.apply_model)
        self.inner_model2.num_timesteps = model.num_timesteps
        self.inner_model2.device = self.ksampler.device if hasattr(self.ksampler, "device") else None
        self.s_min_uncond = 0.0
        self.alphas_cumprod = model.alphas_cumprod
        self.c_adm = None
        self.init_cond = None
        self.init_uncond = None
        self.is_prompt_editing_u = False
        self.is_prompt_editing_c = False

    def apply_model(self, x, timestep, cond, uncond, cond_scale, cond_concat=None, model_options={}, seed=None):
        
        cc=calc_cond(cond, self.step)
        uu=calc_cond(uncond, self.step)
        self.step += 1

        if (any([p[1].get('from_smZ', False) for p in cc]) or 
            any([p[1].get('from_smZ', False) for p in uu])):
            if model_options.get('transformer_options',None) is None:
                model_options['transformer_options'] = {}
            model_options['transformer_options']['from_smZ'] = True

        # Only supports one cond
        for ix in range(len(cc)):
            if cc[ix][1].get('from_smZ', False):
                cc = [cc[ix]]
                break
        for ix in range(len(uu)):
            if uu[ix][1].get('from_smZ', False):
                uu = [uu[ix]]
                break
        c=cc[0][1]
        u=uu[0][1]
        _cc = cc[0][0]
        _uu = uu[0][0]
        if c.get("adm_encoded", None) is not None:
            self.c_adm = torch.cat([c['adm_encoded'], u['adm_encoded']])
            # SDXL. Need to pad with repeats
            _cc, _uu = expand(_cc, _uu)
            _uu, _cc = expand(_uu, _cc)
        x.c_adm = self.c_adm
        conds_list = c.get('conds_list', [[(0, 1.0)]])
        image_cond = txt2img_image_conditioning(None, x)
        out = self.inner_model2(x, timestep, cond=(conds_list, _cc), uncond=_uu, cond_scale=cond_scale, s_min_uncond=self.s_min_uncond, image_cond=image_cond)
        return out

def txt2img_image_conditioning(sd_model, x, width=None, height=None):
    return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)

# =======================================================================================

def set_model_k(self: KSampler):
    self.model_denoise = CFGNoisePredictor(self.model) # main change
    if ((getattr(self.model, "parameterization", "") == "v") or
        (getattr(self.model, "model_type", -1) == model_base.ModelType.V_PREDICTION)):
        self.model_wrap = wrap_model(self.model_denoise, quantize=True)
        self.model_wrap.parameterization = getattr(self.model, "parameterization", "v")
    else:
        self.model_wrap = wrap_model(self.model_denoise, quantize=True)
        self.model_wrap.parameterization = getattr(self.model, "parameterization", "eps")
    self.model_k = KSamplerX0Inpaint(self.model_wrap)

class SDKSampler(comfy.samplers.KSampler):
    def __init__(self, *args, **kwargs):
        super(SDKSampler, self).__init__(*args, **kwargs)
        set_model_k(self)
