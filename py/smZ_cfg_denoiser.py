# https://github.com/shiimizu/ComfyUI_smZNodes
import comfy
import torch
from typing import List
import comfy.sample
from comfy import model_base, model_management
from comfy.samplers import KSampler, KSamplerX0Inpaint
#from comfy.k_diffusion.external import CompVisDenoiser
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy import samplers
from comfy_extras import nodes_custom_sampler
from comfy.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from comfy.sample import np
from comfy import model_management
import comfy.samplers
import inspect
import nodes
import inspect
import functools
import importlib
import os
import re
import itertools
import comfy.sample
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

# ========================================================================
def _find_outer_instance(target:str, target_type=None, callback=None):
    import inspect
    frame = inspect.currentframe()
    i = 0
    while frame and i < 10:
        if target in frame.f_locals:
            if callback is not None:
                return callback(frame)
            else:
                found = frame.f_locals[target]
                if isinstance(found, target_type):
                    return found
        frame = frame.f_back
        i += 1
    return None

if hasattr(comfy.model_patcher, 'ModelPatcher'):
    from comfy.model_patcher import ModelPatcher
else:
    ModelPatcher = object()

# ===========================================================
def prepare_noise(latent_image, seed, noise_inds=None, device='cpu'):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    model = _find_outer_instance('model', ModelPatcher)
    if model is not None and (opts:=model.model_options.get('smZ_opts', None)) is None:
        import comfy.sample
        return comfy.sample.prepare_noise_orig(latent_image, seed, noise_inds)

    if opts.randn_source == 'gpu':
        device = model_management.get_torch_device()

    def get_generator(seed):
        nonlocal device
        nonlocal opts
        _generator = torch.Generator(device=device)
        generator = _generator.manual_seed(seed)
        if opts.randn_source == 'nv':
            generator = rng_philox.Generator(seed)
        return generator
    generator = generator_eta = get_generator(seed)

    if opts.eta_noise_seed_delta > 0:
        seed = min(int(seed + opts.eta_noise_seed_delta), int(0xffffffffffffffff))
        generator_eta = get_generator(seed)


    # hijack randn_like
    import comfy.k_diffusion.sampling
    comfy.k_diffusion.sampling.torch = TorchHijack(generator_eta, opts.randn_source)

    if noise_inds is None:
        shape = latent_image.size()
        if opts.randn_source == 'nv':
            return torch.asarray(generator.randn(shape), device=devices.cpu)
        else:
            return torch.randn(shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device, generator=generator)
    
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        shape = [1] + list(latent_image.size())[1:]
        if opts.randn_source == 'nv':
            noise = torch.asarray(generator.randn(shape), device=devices.cpu)
        else:
            noise = torch.randn(shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device, generator=generator)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

# ===========================================================

# ========================================================================
def bounded_modulo(number, modulo_value):
    return number if number < modulo_value else modulo_value

def get_adm(c):
    for y in ["adm_encoded", "c_adm", "y"]:
        if y in c:
            c_c_adm = c[y]
            if y == "adm_encoded": y="c_adm"
            if type(c_c_adm) is not torch.Tensor: c_c_adm = c_c_adm.cond 
            return {y: c_c_adm, 'key': y}
    return None

getp=lambda x: x[1] if type(x) is list else x
def get_cond(c, current_step, reverse=False):
    """Group by smZ conds that may do prompt-editing / regular conds / comfy conds."""
    if not reverse: _cond = []
    else: _all = []
    fn2=lambda x : getp(x).get("smZid", None)
    prompt_editing = False
    for key, group in itertools.groupby(c, fn2):
        lsg=list(group)
        if key is not None:
            lsg_len = len(lsg)
            i = current_step if current_step < lsg_len else -1
            if lsg_len != 1: prompt_editing = True
            if not reverse: _cond.append(lsg[i])
            else: _all.append(lsg)
        else:
            if not reverse: _cond.extend(lsg)
            else:
                lsg.reverse()
                _all.append(lsg)
    
    if reverse:
        ls=_all
        ls.reverse()
        result=[]
        for d in ls:
            if isinstance(d, list):
                result.extend(d)
            else:
                result.append(d)
        del ls,_all
        return (result, prompt_editing)
    return (_cond, prompt_editing)

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

# ===========================================================
class CFGNoisePredictor:
    def __init__(self, model):
        super().__init__(model)
        self.step = 0
        self.inner_model2 = CFGDenoiser(self.inner_model.apply_model)
        self.c_adm = None
        self.init_cond = None
        self.init_uncond = None
        self.is_prompt_editing_c = True
        self.is_prompt_editing_u = True
        self.use_CFGDenoiser = None
        self.opts = None
        self.sampler = None
        self.steps_multiplier = 1


    def apply_model(self, *args, **kwargs):
        x=kwargs['x'] if 'x' in kwargs else args[0]
        timestep=kwargs['timestep'] if 'timestep' in kwargs else args[1]
        cond=kwargs['cond'] if 'cond' in kwargs else args[2]
        uncond=kwargs['uncond'] if 'uncond' in kwargs else args[3]
        cond_scale=kwargs['cond_scale'] if 'cond_scale' in kwargs else args[4]
        model_options=kwargs['model_options'] if 'model_options' in kwargs else {}

        # reverse doesn't work for some reason???
        # if self.init_cond is None:
        #     if len(cond) != 1 and any(['smZid' in ic for ic in cond]):
        #         self.init_cond = get_cond(cond, self.step, reverse=True)[0]
        #     else:
        #         self.init_cond = cond
        # cond = self.init_cond

        # if self.init_uncond is None:
        #     if len(uncond) != 1 and any(['smZid' in ic for ic in uncond]):
        #         self.init_uncond = get_cond(uncond, self.step, reverse=True)[0]
        #     else:
        #         self.init_uncond = uncond
        # uncond = self.init_uncond

        if self.is_prompt_editing_c:
            cc, ccp=get_cond(cond, self.step // self.steps_multiplier)
            self.is_prompt_editing_c=ccp
        else: cc = cond

        if self.is_prompt_editing_u:
            uu, uup=get_cond(uncond, self.step // self.steps_multiplier)
            self.is_prompt_editing_u=uup
        else: uu = uncond

        if 'transformer_options' not in model_options:
            model_options['transformer_options'] = {}

        if (any([getp(p).get('from_smZ', False) for p in cc]) or
            any([getp(p).get('from_smZ', False) for p in uu])):
            model_options['transformer_options']['from_smZ'] = True

        if not model_options['transformer_options'].get('from_smZ', False):
            out = super().apply_model(*args, **kwargs)
            return out

        if self.is_prompt_editing_c:
            if 'cond' in kwargs: kwargs['cond'] = cc
            else: args[2]=cc
        if self.is_prompt_editing_u:
            if 'uncond' in kwargs: kwargs['uncond'] = uu
            else: args[3]=uu

        if (self.is_prompt_editing_c or self.is_prompt_editing_u) and not self.sampler:
            def get_sampler(frame):
                return frame.f_code.co_name
            self.sampler = _find_outer_instance('extra_args', callback=get_sampler) or 'unknown'
            second_order_samplers = ["dpmpp_2s", "dpmpp_sde", "dpm_2", "heun"]
            # heunpp2 can be first, second, or third order
            third_order_samplers = ["heunpp2"]
            self.steps_multiplier = 2 if any(map(self.sampler.__contains__, second_order_samplers)) else self.steps_multiplier
            self.steps_multiplier = 3 if any(map(self.sampler.__contains__, third_order_samplers)) else self.steps_multiplier

        if self.use_CFGDenoiser is None:
            multi_cc = (any([getp(p)['smZ_opts'].multi_conditioning if 'smZ_opts' in getp(p) else False for p in cc]) and len(cc) > 1)
            multi_uu = (any([getp(p)['smZ_opts'].multi_conditioning if 'smZ_opts' in getp(p) else False for p in uu]) and len(uu) > 1)
            _opts = model_options.get('smZ_opts', None)
            if _opts is not None:
                self.inner_model2.opts = _opts
            self.use_CFGDenoiser = getattr(_opts, 'use_CFGDenoiser', multi_cc or multi_uu)

        # extends a conds_list to the number of latent images
        if self.use_CFGDenoiser and not hasattr(self.inner_model2, 'conds_list'):
            conds_list = []
            for ccp in cc:
                cpl = ccp['conds_list'] if 'conds_list' in ccp else [[(0, 1.0)]]
                conds_list.extend(cpl[0])
            conds_list=[conds_list]
            ix=-1
            cl = conds_list * len(x)
            conds_list=[list(((ix:=ix+1), zl[1]) for zl in cll) for cll in cl]
            self.inner_model2.conds_list = conds_list

        # to_comfy = not opts.debug
        to_comfy = True
        if self.use_CFGDenoiser and not to_comfy:
            _cc = torch.cat([c['model_conds']['c_crossattn'].cond for c in cc])
            _uu = torch.cat([c['model_conds']['c_crossattn'].cond for c in uu])

        # reverse conds here because comfyui reverses them later
        if len(cc) != 1 and any(['smZid' in ic for ic in cond]):
            cc = list(reversed(cc))
            if 'cond' in kwargs: kwargs['cond'] = cc
            else: args[2]=cc
        if len(uu) != 1 and any(['smZid' in ic for ic in uncond]):
            uu = list(reversed(uu))
            if 'uncond' in kwargs: kwargs['uncond'] = uu
            else: args[3]=uu
        
        if not self.use_CFGDenoiser:
            kwargs['model_options'] = model_options
            out = super().apply_model(*args, **kwargs)
        else:
            self.inner_model2.x_in = x
            self.inner_model2.sigma = timestep
            self.inner_model2.cond_scale = cond_scale
            self.inner_model2.image_cond = image_cond = None
            if 'x' in kwargs: kwargs['x'].conds_list = self.inner_model2.conds_list
            else: args[0].conds_list = self.inner_model2.conds_list
            if not hasattr(self.inner_model2, 's_min_uncond'):
                self.inner_model2.s_min_uncond = getattr(model_options.get('smZ_opts', None), 's_min_uncond', 0)
            if 'model_function_wrapper' in model_options:
                model_options['model_function_wrapper_orig'] = model_options.pop('model_function_wrapper')
            if to_comfy:
                model_options["model_function_wrapper"] = self.inner_model2.forward_
            else:
                if 'sigmas' not in model_options['transformer_options']:
                    model_options['transformer_options']['sigmas'] = timestep
            self.inner_model2.model_options = kwargs['model_options'] = model_options
            if not hasattr(self.inner_model2, 'skip_uncond'):
                self.inner_model2.skip_uncond = math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False
            if to_comfy:
                out = sampling_function(self.inner_model, *args, **kwargs)
            else:
                out = self.inner_model2(x, timestep, cond=_cc, uncond=_uu, cond_scale=cond_scale, s_min_uncond=self.inner_model2.s_min_uncond, image_cond=image_cond)
        self.step += 1
        return out


def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
            uncond_ = None
        else:
            uncond_ = uncond

        cfg_result = None
        cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options, cond_scale)
        if hasattr(x, 'conds_list'): cfg_result = cond_pred

        if "sampler_cfg_function" in model_options:
            args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                    "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
            cfg_result = x - model_options["sampler_cfg_function"](args)
        else:
            if cfg_result is None:
                cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                    "sigma": timestep, "model_options": model_options, "input": x}
            cfg_result = fn(args)

        return cfg_result

if hasattr(comfy.samplers, 'get_area_and_mult'):
    from comfy.samplers import get_area_and_mult, can_concat_cond, cond_cat
def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options, cond_scale_in):
    conds = []
    a1111 = hasattr(x_in, 'conds_list')

    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1

    to_run = []
    for x in cond:
        p = get_area_and_mult(x, x_in, timestep)
        if p is None:
            continue

        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = model_management.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp)//i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        if control is not None:
            c['control'] = control if 'tiled_diffusion' in model_options else control.get_control(input_x, timestep_, c, len(cond_or_uncond))

        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        c['transformer_options'] = transformer_options

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
        del input_x

        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                if a1111:
                    out_cond_ = torch.zeros_like(x_in)
                    out_cond_[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                    conds.append(out_cond_)
                else:
                    out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
        del mult
    if not a1111:
        out_cond /= out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    if a1111:
        conds_len = len(conds)
        if conds_len != 0:
            lenc = max(conds_len,1.0)
            cond_scale = 1.0/lenc * (1.0 if "sampler_cfg_function" in model_options else cond_scale_in)
            conds_list = x_in.conds_list
            if (inner_conds_list_len:=len(conds_list[0])) < conds_len:
                conds_list = [[(ix, 1.0 if ix > inner_conds_list_len-1 else conds_list[0][ix][1]) for ix in range(conds_len)]]
            out_cond = out_uncond.clone()
            for cond, (_, weight) in zip(conds, conds_list[0]):
                out_cond += (cond / (out_count / lenc) - out_uncond) * weight * cond_scale

    del out_count
    return out_cond, out_uncond

# =======================================================================================

def inject_code(original_func, data):
    # Get the source code of the original function
    original_source = inspect.getsource(original_func)

    # Split the source code into lines
    lines = original_source.split("\n")

    for item in data:
        # Find the line number of the target line
        target_line_number = None
        for i, line in enumerate(lines):
            if item['target_line'] in line:
                target_line_number = i + 1

                # Find the indentation of the line where the new code will be inserted
                indentation = ''
                for char in line:
                    if char == ' ':
                        indentation += char
                    else:
                        break
                
                # Indent the new code to match the original
                code_to_insert = dedent(item['code_to_insert'])
                code_to_insert = indent(code_to_insert, indentation)
                break

        if target_line_number is None:
            raise FileNotFoundError
            # Target line not found, return the original function
            # return original_func

        # Insert the code to be injected after the target line
        lines.insert(target_line_number, code_to_insert)

    # Recreate the modified source code
    modified_source = "\n".join(lines)
    modified_source = dedent(modified_source.strip("\n"))

    # Create a temporary file to write the modified source code so I can still view the
    # source code when debugging.
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
        temp_file.write(modified_source)
        temp_file.flush()

        MODULE_PATH = temp_file.name
        MODULE_NAME = __name__.split('.')[0] + "_patch_modules"
        spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Pass global variables to the modified module
        globals_dict = original_func.__globals__
        for key, value in globals_dict.items():
            setattr(module, key, value)
        modified_module = module

        # Retrieve the modified function from the module
        modified_function = getattr(modified_module, original_func.__name__)

    # If the original function was a method, bind it to the first argument (self)
    if inspect.ismethod(original_func):
        modified_function = modified_function.__get__(original_func.__self__, original_func.__class__)

    # Update the metadata of the modified function to associate it with the original function
    functools.update_wrapper(modified_function, original_func)

    # Return the modified function
    return modified_function


# ========================================================================
# Hijack sampling

payload = [{
    "target_line": 'extra_args["denoise_mask"] = denoise_mask',
    "code_to_insert": """
            if (any([_p[1].get('from_smZ', False) for _p in positive]) or 
                any([_p[1].get('from_smZ', False) for _p in negative])):
                from ComfyUI_smZNodes.modules.shared import opts as smZ_opts
                if not smZ_opts.sgm_noise_multiplier: max_denoise = False
"""
},
{
    "target_line": 'positive = positive[:]',
    "code_to_insert": """
        if hasattr(self, 'model_denoise'): self.model_denoise.step = start_step if start_step != None else 0
"""
},
]

def hook_for_settings_node_and_sampling():
    if not hasattr(comfy.samplers, 'Sampler'):
        print(f"[smZNodes]: Your ComfyUI version is outdated. Please update to the latest version.")
        comfy.samplers.KSampler.sample = inject_code(comfy.samplers.KSampler.sample, payload)
    else:
        _KSampler_sample = comfy.samplers.KSampler.sample
        _Sampler = comfy.samplers.Sampler
        _max_denoise = comfy.samplers.Sampler.max_denoise
        _sample = comfy.samplers.sample

        def get_value_from_args(args, kwargs, key_to_lookup, fn, idx=None):
            value = None
            if key_to_lookup in kwargs:
                value = kwargs[key_to_lookup]
            else:
                try:
                    # Get its position in the formal parameters list and retrieve from args
                    arg_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
                    index = arg_names.index(key_to_lookup)
                    value = args[index] if index < len(args) else None
                except Exception as err:
                    if idx is not None and idx < len(args):
                        value = args[idx]
            return value

        def KSampler_sample(*args, **kwargs):
            start_step = get_value_from_args(args, kwargs, 'start_step', _KSampler_sample)
            if isinstance(start_step, int):
                args[0].model.start_step = start_step
            return _KSampler_sample(*args, **kwargs)

        def sample(*args, **kwargs):
            model = get_value_from_args(args, kwargs, 'model', _sample, 0)
            # positive = get_value_from_args(args, kwargs, 'positive', _sample, 2)
            # negative = get_value_from_args(args, kwargs, 'negative', _sample, 3)
            sampler = get_value_from_args(args, kwargs, 'sampler', _sample, 6)
            model_options = get_value_from_args(args, kwargs, 'model_options', _sample, 8)
            start_step = getattr(model, 'start_step', None)
            if 'smZ_opts' in model_options:
                model_options['smZ_opts'].start_step = start_step
                opts = model_options['smZ_opts']
                if hasattr(sampler, 'sampler_function'):
                    if not hasattr(sampler, 'sampler_function_orig'):
                        sampler.sampler_function_orig = sampler.sampler_function
                    sampler_function_sig_params = inspect.signature(sampler.sampler_function).parameters
                    params = {x: getattr(opts, x)  for x in ['eta', 's_churn', 's_tmin', 's_tmax', 's_noise'] if x in sampler_function_sig_params}
                    sampler.sampler_function = lambda *a, **kw: sampler.sampler_function_orig(*a, **{**kw, **params})
            model.model_options = model_options # Add model_options to CFGNoisePredictor
            return _sample(*args, **kwargs)

        class Sampler(_Sampler):
            def max_denoise(self, model_wrap: CFGNoisePredictor, sigmas):
                base_model = model_wrap.inner_model
                res = _max_denoise(self, model_wrap, sigmas)
                if (model_options:=base_model.model_options) is not None:
                    if 'smZ_opts' in model_options:
                        opts = model_options['smZ_opts']
                        if getattr(opts, 'start_step', None) is not None:
                            model_wrap.step = opts.start_step
                            opts.start_step = None
                        if not opts.sgm_noise_multiplier:
                            res = False
                return res

        comfy.samplers.Sampler.max_denoise = Sampler.max_denoise
        comfy.samplers.KSampler.sample = KSampler_sample
        comfy.samplers.sample = sample
    comfy.samplers.CFGNoisePredictor = CFGNoisePredictor

def hook_for_rng_orig():
    if not hasattr(comfy.sample, 'prepare_noise_orig'):
        comfy.sample.prepare_noise_orig = comfy.sample.prepare_noise

def hook_for_dtype_unet():
    if hasattr(comfy.model_management, 'unet_dtype'):
        if not hasattr(comfy.model_management, 'unet_dtype_orig'):
            comfy.model_management.unet_dtype_orig = comfy.model_management.unet_dtype
        from .modules import devices
        def unet_dtype(device=None, model_params=0, *args, **kwargs):
            dtype = comfy.model_management.unet_dtype_orig(device=device, model_params=model_params, *args, **kwargs)
            if model_params != 0:
                devices.dtype_unet = dtype
            return dtype
        comfy.model_management.unet_dtype = unet_dtype

def try_hook(fn):
    try:
        fn()
    except Exception as e:
        print("\033[92m[smZNodes] \033[0;33mWARNING:\033[0m", e)

def register_hooks():
    hooks = [
        hook_for_settings_node_and_sampling,
        hook_for_rng_orig,
        hook_for_dtype_unet,
    ]
    for hook in hooks:
        try_hook(hook)

# ========================================================================

# DPM++ 2M alt

from tqdm.auto import trange
@torch.no_grad()
def sample_dpmpp_2m_alt(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        sigma_progress = i / len(sigmas)
        adjustment_factor = 1 + (0.15 * (sigma_progress * sigma_progress))
        old_denoised = denoised * adjustment_factor
    return x


def add_sample_dpmpp_2m_alt():
    from comfy.samplers import KSampler, k_diffusion_sampling
    if "dpmpp_2m_alt" not in KSampler.SAMPLERS:
        try:
            idx = KSampler.SAMPLERS.index("dpmpp_2m")
            KSampler.SAMPLERS.insert(idx+1, "dpmpp_2m_alt")
            setattr(k_diffusion_sampling, 'sample_dpmpp_2m_alt', sample_dpmpp_2m_alt)
            import importlib
            importlib.reload(k_diffusion_sampling)
        except ValueError as e: ...

def add_custom_samplers():
    samplers = [
        add_sample_dpmpp_2m_alt,
    ]
    for add_sampler in samplers:
        add_sampler()
