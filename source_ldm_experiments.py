import argparse
import torch
import math
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda().eval()
    return model

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
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image1", type=str, required=True)
    parser.add_argument("--image2", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--timestep", type=int, default=20)
    parser.add_argument("--frac", type=float, nargs='+', default=[0.3])
    parser.add_argument("--gamma", type=float, nargs='+', default=[0.0])
    parser.add_argument("--coef", type=float, nargs='+', default=[3.0])
    parser.add_argument("--mu", type=float, nargs='+', default=None)
    parser.add_argument("--nu", type=float, nargs='+', default=None)
    parser.add_argument("--method", type=str, choices=["noise_diffusion", "slerp", "noise"], default="noise_diffusion")
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device("cuda")
    model = torch.load(args.ckpt, weights_only=False)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=250, ddim_eta=0.0)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image_tensor1 = transform(Image.open(args.image1).convert("RGB")).unsqueeze(0).to(device)
    image_tensor2 = transform(Image.open(args.image2).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        latent1 = model.first_stage_model.encode(image_tensor1)
        latent2 = model.first_stage_model.encode(image_tensor2)

    timesteps = sampler.ddim_timesteps
    t = timesteps[args.timestep]
    sqrt_alpha_bar = sampler.alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alpha_bar = (1.0 - sampler.alphas_cumprod[t]) ** 0.5
    noise = torch.randn_like(latent1)

    zt1 = sqrt_alpha_bar * latent1 + sqrt_one_minus_alpha_bar * noise
    zt2 = sqrt_alpha_bar * latent2 + sqrt_one_minus_alpha_bar * noise

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

        if args.method == "noise_diffusion":
            noisy_latent = (
                alpha * zt1 + beta * zt2 +
                (mu - alpha) * sqrt_alpha_bar * latent1 +
                (nu - beta) * sqrt_alpha_bar * latent2 +
                args.gamma * noise * sqrt_one_minus_alpha_bar
            )
        elif args.method == "slerp":
            noisy_latent = slerp(zt1, zt2, latent_frac)
        elif args.method == "noise":
            l1 = sqrt_alpha_bar * latent1 + sqrt_one_minus_alpha_bar * noise
            l2 = sqrt_alpha_bar * latent2 + sqrt_one_minus_alpha_bar * noise
            noisy_latent = slerp(l1, l2, latent_frac)
        else:
            raise ValueError("Unknown interpolation method")

        timesteps_tensor = torch.tensor([t], device=device).long()
        with torch.no_grad():
            eps_pred = model.model.diffusion_model(noisy_latent, timesteps_tensor, c=None)

        z0_hat = (noisy_latent - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar
        img = model.decode_first_stage(z0_hat)

        out_name = args.output.replace(".png", f"_{vary_param}_{val}.png")
        save_image(img, out_name)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
