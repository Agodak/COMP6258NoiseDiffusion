import argparse
import torch
import math
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def slerp(p0, p1, fract_mixing):
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'

    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1 + epsilon, 1 - epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0 * s0 + p1 * s1

    if recast_to == 'fp16':
        return interp.half()
    return interp.float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image1", type=str, required=True)
    parser.add_argument("--image2", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--timestep", type=int, default=140)
    parser.add_argument("--frac", type=float, nargs='+', default=[0.3])
    parser.add_argument("--gamma", type=float, nargs='+', default=[0.0])
    parser.add_argument("--coef", type=float, nargs='+', default=[3.0])
    parser.add_argument("--mu", type=float, nargs='+', default=None)
    parser.add_argument("--nu", type=float, nargs='+', default=None)
    parser.add_argument("--method", type=str, choices=["noise_diffusion", "slerp", "noise"], default="noise_diffusion")
    parser.add_argument("--prompt", type=str, default="a person's face")
    parser.add_argument("--negative_prompt", type=str, default="not good, bad quality")
    args = parser.parse_args()

    multi_params = {k: v for k, v in vars(args).items() if isinstance(v, list) and len(v) > 1}
    if len(multi_params) > 1:
        raise ValueError("Only one parameter can vary at a time.")

    vary_param = list(multi_params.keys())[0] if multi_params else 'frac'
    vary_values = multi_params[vary_param] if multi_params else [getattr(args, vary_param)[0]]

    for p in ['frac', 'gamma', 'coef', 'mu', 'nu']:
        val = getattr(args, p)
        if val is not None and not isinstance(val, float):
            setattr(args, p, val[0])

    device = torch.device("cuda")
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(args.ckpt, location='cuda'), strict=False)
    model = model.cuda()
    model.control_scales = [1] * 13

    ddim_sampler = DDIMSampler(model)
    ddim_sampler.make_schedule(250, ddim_eta=0.0, verbose=False)
    timesteps = ddim_sampler.ddim_timesteps
    t = timesteps[args.timestep]

    prompt = args.prompt
    n_prompt = args.negative_prompt
    cond1 = model.get_learned_conditioning([prompt])
    cond = {"c_crossattn": [cond1], 'c_concat': None}
    uncond_base = model.get_learned_conditioning([n_prompt])
    un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}

    def preprocess_image(path):
        img = Image.open(path).resize((768, 768))
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).cuda()
        return img_tensor[:, :3, :, :] if img_tensor.shape[1] > 3 else img_tensor

    img1 = preprocess_image(args.image1)
    img2 = preprocess_image(args.image2)

    left_image = model.get_first_stage_encoding(model.encode_first_stage(img1.float() / 127.5 - 1.0))
    right_image = model.get_first_stage_encoding(model.encode_first_stage(img2.float() / 127.5 - 1.0))

    latent1, _ = ddim_sampler.encode(left_image, cond, args.timestep, use_original_steps=False, return_intermediates=None, unconditional_guidance_scale=1, unconditional_conditioning=un_cond)
    latent2, _ = ddim_sampler.encode(right_image, cond, args.timestep, use_original_steps=False, return_intermediates=None, unconditional_guidance_scale=1, unconditional_conditioning=un_cond)

    noise = torch.randn_like(left_image)

    for val in vary_values:
        setattr(args, vary_param, val)
        latent_frac = args.frac
        alpha = math.cos(math.radians(latent_frac * 90))
        beta = math.sin(math.radians(latent_frac * 90))
        l = alpha / beta
        alpha = math.sqrt((1 - args.gamma ** 2) * l**2 / (l**2 + 1))
        beta = math.sqrt((1 - args.gamma ** 2) / (l**2 + 1))
        mu = args.mu if args.mu is not None else 1.2 * alpha / (alpha + beta)
        nu = args.nu if args.nu is not None else 1.2 * beta / (alpha + beta)

        l1 = torch.clip(latent1.clone(), -args.coef, args.coef)
        l2 = torch.clip(latent2.clone(), -args.coef, args.coef)

        if args.method == "noise_diffusion":
            noisy_latent = (
                alpha * l1 + beta * l2 +
                (mu - alpha) * model.sqrt_alphas_cumprod[t] * left_image +
                (nu - beta) * model.sqrt_alphas_cumprod[t] * right_image +
                args.gamma * noise * model.sqrt_one_minus_alphas_cumprod[t]
            )
        elif args.method == "slerp":
            noisy_latent = slerp(l1, l2, latent_frac)
        elif args.method == "noise":
            l1 = model.sqrt_alphas_cumprod[t] * left_image + model.sqrt_one_minus_alphas_cumprod[t] * noise
            l2 = model.sqrt_alphas_cumprod[t] * right_image + model.sqrt_one_minus_alphas_cumprod[t] * noise
            noisy_latent = slerp(l1, l2, latent_frac)
        else:
            raise ValueError("Unknown interpolation method")

        noisy_latent = torch.clip(noisy_latent, -args.coef, args.coef)

        samples = ddim_sampler.decode(noisy_latent, cond, args.timestep, unconditional_guidance_scale=10, unconditional_conditioning=un_cond, use_original_steps=False)
        image = model.decode_first_stage(samples)
        image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        out_name = args.output.replace(".png", f"_{vary_param}_{val}.png")
        Image.fromarray(image[0]).save(out_name)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
