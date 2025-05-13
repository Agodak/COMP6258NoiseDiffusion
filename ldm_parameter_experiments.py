import argparse
import torch
import math
import numpy as np
from PIL import Image
from diffusers import UNet2DModel, DDIMScheduler, VQModel, DDIMInverseScheduler
from torchvision import transforms
import tqdm
import os

def parse_multi_float(value):
    try:
        values = [float(v) for v in value.split(",")]
        return values if len(values) > 1 else values[0]
    except:
        raise argparse.ArgumentTypeError("Must be a float or comma-separated list of floats")

def load_models(model_id, device):
    unet = UNet2DModel.from_pretrained(model_id, subfolder="unet").to(device)
    vqvae = VQModel.from_pretrained(model_id, subfolder="vqvae").to(device)
    scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    inv_sched = DDIMInverseScheduler.from_pretrained(model_id, subfolder="scheduler")
    return unet, vqvae, scheduler, inv_sched

def encode_image(image_path, transform, vqvae, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        encoded = vqvae.encode(image_tensor).latents
    return encoded

def decode_latent(latent, vqvae):
    with torch.no_grad():
        image = vqvae.decode(latent).sample.cpu().permute(0, 2, 3, 1)
    image = (image + 1.0) * 127.5
    image = image.clamp(0, 255).numpy().astype(np.uint8)
    return Image.fromarray(image[0])

def slerp(p0, p1, fract_mixing: float):
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
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
    return interp


def interpolate_latents(image1, image2, inv_sched, scheduler, unet, vqvae,frac, coef, gamma, mu_override, nu_override, method, device):

    inv_sched.set_timesteps(200)
    scheduler.set_timesteps(num_inference_steps=200)

    x_t1, x_t2 = image1.clone(), image2.clone()
    for t in tqdm.tqdm(inv_sched.timesteps, desc="Inverting Images", leave=False):
        with torch.no_grad():
            eps = unet(x_t1, t)["sample"]
            x_t1 = inv_sched.step(eps, t, x_t1)["prev_sample"]
            eps = unet(x_t2, t)["sample"]
            x_t2 = inv_sched.step(eps, t, x_t2)["prev_sample"]

    alpha_sqrt = torch.sqrt(inv_sched.alphas_cumprod[inv_sched.timesteps[-1]])
    convex_diff = 1 - alpha_sqrt
    noise = torch.randn_like(x_t1)

    latent_frac = frac
    alpha = math.cos(math.radians(latent_frac * 90))
    beta = math.sin(math.radians(latent_frac * 90))
    l = alpha / beta

    alpha = ((1 - gamma**2) * l**2 / (l**2 + 1))**0.5
    beta = ((1 - gamma**2) / (l**2 + 1))**0.5

    if method == "noise_diffusion":
        mu = mu_override if mu_override is not None else 1.2 * alpha / (alpha + beta)
        nu = nu_override if nu_override is not None else 1.2 * beta / (alpha + beta)

        x_t1 = torch.clip(x_t1, -coef, coef)
        x_t2 = torch.clip(x_t2, -coef, coef)

        noisy_latent = (
            alpha * x_t1 + beta * x_t2 +
            (mu - alpha) * alpha_sqrt * image1 +
            (nu - beta) * alpha_sqrt * image2 +
            gamma * noise * convex_diff
        )
        noisy_latent = torch.clip(noisy_latent, -coef, coef)

    elif method == "slerp":
        x_t1 = torch.clip(x_t1, -coef, coef)
        x_t2 = torch.clip(x_t2, -coef, coef)
        noisy_latent = slerp(x_t1, x_t2, latent_frac)

    elif method == "noise":
        l1 = alpha_sqrt * image1 + convex_diff * noise
        l2 = alpha_sqrt * image2 + convex_diff * noise
        noisy_latent = slerp(l1, l2, latent_frac)

    else:
        raise ValueError(f"Unknown method: {method}")

    for t in tqdm.tqdm(scheduler.timesteps, desc="Decoding Latent", leave=False):
        with torch.no_grad():
            residual = unet(noisy_latent, t)["sample"]
        noisy_latent = scheduler.step(residual, t, noisy_latent, eta=0.0)["prev_sample"]

    return decode_latent(noisy_latent, vqvae)

def main():
    parser = argparse.ArgumentParser(description="Latent Space Face Interpolation")
    parser.add_argument("--method", type=str, default="noise_diffusion",
    choices=["noise_diffusion", "slerp", "noise"],
    help="Interpolation method: 'noise_diffusion' (default), 'slerp', or 'noise'")
    parser.add_argument("--image1", type=str, required=True, help="Path to first image")
    parser.add_argument("--image2", type=str, required=True, help="Path to second image")
    parser.add_argument("--model_id", type=str, default="CompVis/ldm-celebahq-256", help="Hugging Face model ID")
    parser.add_argument("--output", type=str, required=True, help="Base output path (will append param values)")
    parser.add_argument("--frac", type=parse_multi_float, default=0.1, help="Interpolation fraction (float or list)")
    parser.add_argument("--coef", type=parse_multi_float, default=2.0, help="Clipping coefficient (float or list)")
    parser.add_argument("--gamma", type=parse_multi_float, default=0.0, help="Gamma (float or list)")
    parser.add_argument("--mu", type=parse_multi_float, default=None, help="Mu (float or list)")
    parser.add_argument("--nu", type=parse_multi_float, default=None, help="Nu (float or list)")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detect which param is a list (only one allowed)
    varying_param = None
    for name in ['frac', 'coef', 'gamma', 'mu', 'nu']:
        val = getattr(args, name)
        if isinstance(val, list):
            if varying_param:
                raise ValueError("Only one parameter can be a list at a time.")
            varying_param = name

    if not varying_param:
        varying_param = 'frac'
        setattr(args, varying_param, [getattr(args, varying_param)])

    values = getattr(args, varying_param)

    unet, vqvae, scheduler, inv_sched = load_models(args.model_id, device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image_encoded1 = encode_image(args.image1, transform, vqvae, device)
    image_encoded2 = encode_image(args.image2, transform, vqvae, device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for val in values:
        kwargs = {
            "frac": args.frac if varying_param != 'frac' else val,
            "coef": args.coef if varying_param != 'coef' else val,
            "gamma": args.gamma if varying_param != 'gamma' else val,
            "mu_override": args.mu if varying_param != 'mu' else val,
            "nu_override": args.nu if varying_param != 'nu' else val,
        }

        print(f"Generating with {varying_param} = {val}")
        result = interpolate_latents(
        image_encoded1, image_encoded2,
        inv_sched, scheduler, unet, vqvae,
        frac=kwargs["frac"], coef=kwargs["coef"], gamma=kwargs["gamma"],
        mu_override=kwargs["mu_override"], nu_override=kwargs["nu_override"],
        method=args.method,
        device=device
    )

        out_path = args.output.replace(".png", f"_{varying_param}_{val}.png")
        result.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
