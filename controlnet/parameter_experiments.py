import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import math
import os
osp = os.path
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
F = torch.nn.functional

import yaml
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def interpolate_linear(p0,p1, frac):
    return p0 + (p1 - p0) * frac

@torch.no_grad()
def slerp(p0, p1, fract_mixing: float):
    r""" Copied from lunarring/latentblending
    Helper function to correctly mix two random variables using spherical interpolation.
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """ 
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'
    
    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)
    
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1
    
    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
    return interp

def interpolate_new(model, ddim_sampler, img1, img2,  scale_control=1.5, prompt=None, n_prompt=None, min_steps=.3, max_steps=.55, ddim_steps=250,  guide_scale=7.5,  optimize_cond=0,  cond_lr=1e-4, bias=0, ddim_eta=0, out_dir='blend'):
    torch.manual_seed(49)

    if isinstance(img1, Image.Image):
        if img1.mode == 'RGBA':#
                img1 = img1.convert('RGB')
        if img2.mode == 'RGBA':
            img2 = img2.convert('RGB')
        img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
        img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()

    ldm = model
    ldm.control_scales = [1] * 13

    cond1 = ldm.get_learned_conditioning([prompt])
    uncond_base = ldm.get_learned_conditioning([n_prompt])
    cond = {"c_crossattn": [cond1], 'c_concat': None}
    un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}

    ddim_sampler.make_schedule(ddim_steps, ddim_eta=ddim_eta, verbose=False)
    timesteps = ddim_sampler.ddim_timesteps

    left_image= ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.float() / 127.5 - 1.0))
    right_image= ldm.get_first_stage_encoding(ldm.encode_first_stage(img2.float() / 127.5 - 1.0))


    cur_step=140
    t = timesteps[cur_step]

    l1, _ = ddim_sampler.encode(left_image, cond, cur_step,
    use_original_steps=False, return_intermediates=None,
    unconditional_guidance_scale=1, unconditional_conditioning=un_cond)
    l2, _ = ddim_sampler.encode(right_image, cond, cur_step,
    use_original_steps=False, return_intermediates=None,
    unconditional_guidance_scale=1, unconditional_conditioning=un_cond)

    out_dir = "results_gamma"
    for num in range(6):
        gamma_list = [0, math.sqrt(0.2), math.sqrt(0.4), math.sqrt(0.6), math.sqrt(0.8), 1]
        name_list=["0", "0.2", "0.4", "0.6", "0.8", "1"]
        frac=0.5
        name=name_list[num]
        latent_frac=frac
        noise = torch.randn_like(left_image)

        coef=2.0
        gamma=gamma_list[num]
        alpha=math.sin(math.radians(latent_frac*90))
        beta=math.cos(math.radians(latent_frac*90))
        l=alpha/beta

        if gamma == 1:
            dummy_gamma = 0.99
            alpha = ((1 - dummy_gamma * dummy_gamma) * l * l / (l * l + 1)) ** 0.5
            beta = ((1 - dummy_gamma * dummy_gamma) / (l * l + 1)) ** 0.5
        else:
            alpha=((1-gamma*gamma)*l*l/(l*l+1))**0.5
            beta=((1-gamma*gamma)/(l*l+1))**0.5

        mu=2*alpha/(alpha+beta)
        nu=2*beta/(alpha+beta)

        l1=torch.clip(l1,-coef,coef)
        l2=torch.clip(l2,-coef,coef)

        noisy_latent= alpha*l1+beta*l2+(mu-alpha)*ldm.sqrt_alphas_cumprod[t] * left_image+(nu-beta)*ldm.sqrt_alphas_cumprod[t] * right_image+gamma*noise*ldm.sqrt_one_minus_alphas_cumprod[t]

        noisy_latent=torch.clip(noisy_latent,-coef,coef)


        samples= ddim_sampler.decode(noisy_latent, cond, cur_step, unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond, use_original_steps=False)

        image = ldm.decode_first_stage(samples)

        image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        Image.fromarray(image[0]).save(f'{out_dir}/{name}.png')

    out_dir = "results_mu_nu"
    for num in range(5):
        mu_nu_list = [10, 2, 1, 0.5, 0.1]
        name_list = ["10", "2", "1", "0.5", "0.1"]
        frac = 0.5
        name = name_list[num]
        latent_frac = frac
        noise = torch.randn_like(left_image)

        coef = 2.0
        gamma = 0
        alpha = math.sin(math.radians(latent_frac * 90))
        beta = math.cos(math.radians(latent_frac * 90))
        l = alpha / beta

        alpha = ((1 - gamma * gamma) * l * l / (l * l + 1)) ** 0.5
        beta = ((1 - gamma * gamma) / (l * l + 1)) ** 0.5

        mu = nu = mu_nu_list[num]

        l1 = torch.clip(l1, -coef, coef)
        l2 = torch.clip(l2, -coef, coef)

        noisy_latent = alpha * l1 + beta * l2 + (mu - alpha) * ldm.sqrt_alphas_cumprod[t] * left_image + (
                    nu - beta) * ldm.sqrt_alphas_cumprod[t] * right_image + gamma * noise * \
                       ldm.sqrt_one_minus_alphas_cumprod[t]

        noisy_latent = torch.clip(noisy_latent, -coef, coef)

        samples = ddim_sampler.decode(noisy_latent, cond, cur_step,
                                           unconditional_guidance_scale=guide_scale,
                                           unconditional_conditioning=un_cond, use_original_steps=False)

        image = ldm.decode_first_stage(samples)

        image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        Image.fromarray(image[0]).save(f'{out_dir}/{name}.png')

    out_dir = "results_boundary"
    for num in range(7):
        coef_list = [2, 2.2, 2.4, 2.6, 2.8, 3, 3.2]
        name_list = ["2", "2.2", "2.4", "2.6", "2.8", "3", "3.2"]
        frac = 0.5
        name = name_list[num]
        latent_frac = frac
        noise = torch.randn_like(left_image)

        coef = coef_list[num]
        gamma = 0
        alpha = math.sin(math.radians(latent_frac * 90))
        beta = math.cos(math.radians(latent_frac * 90))
        l = alpha / beta

        alpha = ((1 - gamma * gamma) * l * l / (l * l + 1)) ** 0.5
        beta = ((1 - gamma * gamma) / (l * l + 1)) ** 0.5

        mu = 2 * alpha / (alpha + beta)
        nu = 2 * beta / (alpha + beta)

        l1 = torch.clip(l1, -coef, coef)
        l2 = torch.clip(l2, -coef, coef)

        noisy_latent = alpha * l1 + beta * l2 + (mu - alpha) * ldm.sqrt_alphas_cumprod[t] * left_image + (
                nu - beta) * ldm.sqrt_alphas_cumprod[t] * right_image + gamma * noise * \
                       ldm.sqrt_one_minus_alphas_cumprod[t]

        noisy_latent = torch.clip(noisy_latent, -coef, coef)

        samples = ddim_sampler.decode(noisy_latent, cond, cur_step,
                                           unconditional_guidance_scale=guide_scale,
                                           unconditional_conditioning=un_cond, use_original_steps=False)

        image = ldm.decode_first_stage(samples)

        image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        Image.fromarray(image[0]).save(f'{out_dir}/{name}.png')

filters = {}
mode = None
model = create_model('./models/cldm_v21.yaml').cuda()
ddim_sampler = DDIMSampler(model)
model.load_state_dict(load_state_dict('./models/control_v11p_sd21_openpose.ckpt', location='cuda'))

img1 = Image.open('sample_imgs/bedroom1.png').resize((768, 768))
img2 = Image.open('sample_imgs/bedroom2.png').resize((768, 768))

prompt='a photo of bed'
n_prompt='text, signature, logo, distorted, ugly, weird eyes, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality, drawing, blurry, faded'
interpolate_new(model, ddim_sampler, img1, img2,  prompt=prompt, n_prompt=n_prompt, ddim_steps=200,  guide_scale=10,  out_dir='results_gamma')