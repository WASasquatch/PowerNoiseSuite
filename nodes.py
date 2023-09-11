import torch
import torch.nn.functional as F

import nodes
import comfy.samplers
import comfy.k_diffusion.sampling

from .modules.latent_util import (
    normalize, 
    latents_to_images
)
from .modules.latent_filters import (
    sharpen_latents, 
    high_pass_latents,
    blending_modes
)
from .modules.latent_noise import (
    CrossHatchPowerFractal, 
    CrossHatchLinearPowerFractal, 
    PerlinPowerFractal,
    PowerLawNoise
)

# PERLIN POWER FRACTAL NOISE LATENT

class PPFNoiseNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "max": 64, "min": 1, "step": 1}),
                "width": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "height": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "resampling": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
                "X": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "Y": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "Z": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "evolution": ("FLOAT", {"default": 0.0, "max": 1.0, "min": 0.0, "step": 0.01}),
                "frame": ("INT", {"default": 0, "max": 99999999, "min": 0, "step": 1}),
                "scale": ("FLOAT", {"default": 5, "max": 2048, "min": 2, "step": 0.01}),
                "octaves": ("INT", {"default": 8, "max": 8, "min": 1, "step": 1}),
                "persistence": ("FLOAT", {"default": 1.5, "max": 23.0, "min": 0.01, "step": 0.01}),
                "lacunarity": ("FLOAT", {"default": 2.0, "max": 99.0, "min": 0.01, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 4.0, "max": 38.0, "min": 0.01, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.01}),
                "clamp_min": ("FLOAT", {"default": 0.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "clamp_max": ("FLOAT", {"default": 1.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
                "device": (["cpu", "cuda"],),
            },
            "optional": {
                "optional_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT","IMAGE")
    RETURN_NAMES = ("latents","previews")
    FUNCTION = "power_fractal_latent"

    CATEGORY = "Power Noise Suite/Latent/Noise"
    
    def power_fractal_latent(self, batch_size, width, height, resampling, X, Y, Z, evolution, frame, scale, octaves, persistence, lacunarity, exponent, brightness, contrast, clamp_min, clamp_max, seed, device, optional_vae=None):
                    
        color_intensity = 1
        masking_intensity = 1
        
        batch_size = int(batch_size)
        width = int(width)
        height = int(height)

        channel_tensors = []
        for i in range(batch_size):
            nseed = seed + i * 12
            rgb_noise_maps = []
            
            rgb_image = torch.zeros(4, height, width)
            
            for j in range(3):
                rgba_noise_map = self.generate_noise_map(width, height, X, Y, Z, frame, device, evolution, octaves, persistence, lacunarity, exponent, scale, brightness, contrast, nseed + j, clamp_min, clamp_max)
                rgb_noise_map = rgba_noise_map.squeeze(-1)
                rgb_noise_map *= color_intensity
                rgb_noise_map *= masking_intensity
                
                rgb_image[j] = rgb_noise_map
                
            rgb_image[3] = torch.ones(height, width)
            
            channel_tensors.append(rgb_image)
            
        tensors = torch.stack(channel_tensors)
        tensors = normalize(tensors)
        
        if optional_vae is None:
            latents = F.interpolate(tensors, size=((height // 8), (width // 8)), mode=resampling)
            return {'samples': latents}, tensors.permute(0, 2, 3, 1)
            
        encoder = nodes.VAEEncode()
        
        latents = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(0)
            tensor = tensor.permute(0, 2, 3, 1)
            latents.append(encoder.encode(optional_vae, tensor)[0]['samples'])
            
        latents = torch.cat(latents)
        
        return {'samples': latents}, tensors.permute(0, 2, 3, 1)
        
    def generate_noise_map(self, width, height, X, Y, Z, frame, device, evolution, octaves, persistence, lacunarity, exponent, scale, brightness, contrast, seed, clamp_min, clamp_max):
        PPF = PerlinPowerFractal(width, height)
        noise_map = PPF(1, X, Y, Z, frame, device, evolution, octaves, persistence, lacunarity, exponent, scale, brightness, contrast, seed, clamp_min, clamp_max)
        return noise_map

 # POWER-LOW NOISE

class PPFNPowerLawNoise:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        pln = PowerLawNoise('cpu')
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "max": 64, "min": 1, "step": 1}),
                "width": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "height": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "resampling": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
                "noise_type": (pln.get_noise_types(),),
                "frequency": ("FLOAT", {"default": 64, "max": 1024.0, "min": 0.001, "step": 0.001}),
                "attenuation": ("FLOAT", {"default": 1.0, "max": 1024.0, "min": 0.001, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
                "device": (["cpu", "cuda"],),
            },
            "optional": {
                "optional_vae": ("VAE",),
            }
        }
        
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latents", "previews")
    FUNCTION = "power_noise"
    
    CATEGORY = "Power Noise Suite/Latent/Noise"
    
    def power_noise(self, batch_size, width, height, resampling, noise_type, frequency, attenuation, seed, device, optional_vae=None):
    
        power_law = PowerLawNoise(device=device)
        tensors = power_law(batch_size, width, height, frequency=frequency, attenuation=attenuation, noise_type=noise_type, seed=seed)     

        alpha_channel = torch.ones((batch_size, height, width, 1), dtype=tensors.dtype, device=device)
        tensors = torch.cat((tensors, alpha_channel), dim=3)
            
        if optional_vae is None:
            latents = tensors.permute(0, 3, 1, 2)
            latents = F.interpolate(latents, size=((height // 8), (width // 8)), mode=resampling)
            return {'samples': latents}, tensors
            
        encoder = nodes.VAEEncode()
        
        latents = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(0)
            latents.append(encoder.encode(optional_vae, tensor)[0]['samples'])
            
        latents = torch.cat(latents)
        
        return {'samples': latents}, tensors 
 
# CROSS-HATCH POWER FRACTAL LATENT

class PPFNCrossHatchNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "max": 64, "min": 1, "step": 1}),
                "width": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "height": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "resampling": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
                "frequency": ("FLOAT", {"default": 320.0, "max": 1024.0, "min": 0.001, "step": 0.001}),
                "octaves": ("INT", {"default": 12, "max": 32, "min": 1, "step": 1}),
                "persistence": ("FLOAT", {"default": 1.5, "max": 2.0, "min": 0.001, "step": 0.001}),
                "num_colors": ("INT", {"default": 16, "max": 256, "min": 2, "step": 1}),
                "color_tolerance": ("FLOAT", {"default": 0.05, "max": 1.0, "min": 0.001, "step": 0.001}),
                "angle_degrees": ("FLOAT", {"default": 45.0, "max": 360.0, "min": 0.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.001}),
                "contrast": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.001}),
                "blur": ("FLOAT", {"default": 2.5, "max": 1024, "min": 0, "step": 0.01}),
                "clamp_min": ("FLOAT", {"default": 0.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "clamp_max": ("FLOAT", {"default": 1.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
                "device": (["cpu", "cuda"],),
            },
            "optional": {
                "optional_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT","IMAGE")
    RETURN_NAMES = ("latents","previews")
    FUNCTION = "cross_hatch"

    CATEGORY = "Power Noise Suite/Latent/Noise"
    
    def cross_hatch(self, batch_size, width, height, resampling, frequency, octaves, persistence, color_tolerance, num_colors, angle_degrees, brightness, contrast, blur, clamp_min, clamp_max, seed, device, optional_vae=None):

        cross_hatch = CrossHatchPowerFractal(width=width, height=height, frequency=frequency, octaves=octaves, persistence=persistence, num_colors=num_colors, color_tolerance=color_tolerance, angle_degrees=angle_degrees, blur=blur, clamp_min=clamp_min, clamp_max=clamp_max)
        tensors = cross_hatch(batch_size, device, seed).to(device="cpu")
        tensors = torch.cat([tensors, torch.ones(batch_size, height, width, 1, dtype=tensors.dtype, device='cpu')], dim=-1)
                
        if optional_vae is None:
            latents = tensors.permute(0, 3, 1, 2)
            latents = F.interpolate(latents, size=((height // 8), (width // 8)), mode=resampling)
            return {'samples': latents}, tensors
            
        encoder = nodes.VAEEncode()
        
        latents = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(0)
            latents.append(encoder.encode(optional_vae, tensor)[0]['samples'])
            
        latents = torch.cat(latents)
        
        return {'samples': latents}, tensors 
 
# LINEAR CROSS-HATCH POWER FRACTAL LATENT

class PPFNLinearCrossHatchNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "max": 64, "min": 1, "step": 1}),
                "width": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "height": ("INT", {"default": 512, "max": 8192, "min": 64, "step": 1}),
                "resampling": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
                "frequency": ("FLOAT", {"default": 320.0, "max": 1024.0, "min": 0.001, "step": 0.001}),
                "gain": ("FLOAT", {"default": 0.25, "max": 1.0, "min": 0.0, "step": 0.001}),
                "octaves": ("INT", {"default": 12, "max": 32, "min": 1, "step": 1}),
                "persistence": ("FLOAT", {"default": 1.5, "max": 2.0, "min": 0.001, "step": 0.001}),
                "add_noise": ("FLOAT", {"default": 0.0, "max": 1.0, "min": 0.0, "step": 0.001}),
                "linear_range": ("INT", {"default": 16, "max": 256, "min": 2, "step": 1}),
                "linear_tolerance": ("FLOAT", {"default": 0.05, "max": 1.0, "min": 0.001, "step": 0.001}),
                "angle_degrees": ("FLOAT", {"default": 45.0, "max": 360.0, "min": 0.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.001}),
                "contrast": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
                "device": (["cpu", "cuda"],),
            },
            "optional": {
                "optional_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT","IMAGE")
    RETURN_NAMES = ("latents","previews")
    FUNCTION = "cross_hatch"

    CATEGORY = "Power Noise Suite/Latent/Noise"
    
    def cross_hatch(self, batch_size, width, height, resampling, frequency, gain, octaves, persistence, add_noise, linear_range, linear_tolerance, angle_degrees, brightness, contrast, seed, device, optional_vae=None):

        cross_hatch = CrossHatchLinearPowerFractal(width=width, height=height, frequency=frequency, gain=gain, octaves=octaves, persistence=persistence, add_noise_tolerance=add_noise, mapping_range=linear_range, angle_degrees=angle_degrees, brightness=brightness, contrast=contrast)
        
        tensors = cross_hatch(batch_size, device, seed)
        tensors = torch.cat([tensors, torch.ones(batch_size, height, width, 1, dtype=tensors.dtype, device='cpu')], dim=-1)
                
        if optional_vae is None:
            latents = tensors.permute(0, 3, 1, 2)
            latents = F.interpolate(latents, size=((height // 8), (width // 8)), mode=resampling)
            return {'samples': latents}, tensors
            
        encoder = nodes.VAEEncode()
        
        latents = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(0)
            latents.append(encoder.encode(optional_vae, tensor)[0]['samples'])
            
        latents = torch.cat(latents)
        
        return {'samples': latents}, tensors

# BLEND LATENTS

class PPFNBlendLatents:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "operation": (sorted(list(blending_modes.keys())),),
                "blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
                "set_noise_mask": (["false", "true"],),
                "normalize": (["false", "true"],),
                "clamp_min": ("FLOAT", {"default": 0.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "clamp_max": ("FLOAT", {"default": 1.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "latent2rgb_preview": (["false", "true"],),
            }
        }

    RETURN_TYPES = ("LATENT","IMAGE",)
    RETURN_NAMES = ("latents", "previews")
    FUNCTION = "latent_blend"
    
    CATEGORY = "Power Noise Suite/Latent/Adjustements"

    def latent_blend(self, latent_a, latent_b, operation, blend_ratio, blend_strength, mask=None, set_noise_mask=None, normalize=None, clamp_min=None, clamp_max=None, latent2rgb_preview=None):
        
        latent_a = latent_a["samples"][:, :-1]
        latent_b = latent_b["samples"][:, :-1]

        assert latent_a.shape == latent_b.shape, f"Input latents must have the same shape, but got: a {latent_a.shape}, b {latent_b.shape}"

        alpha_a = latent_a[:, -1:]
        alpha_b = latent_b[:, -1:]
        
        blended_rgb = self.blend_latents(latent_a, latent_b, operation, blend_ratio, blend_strength, clamp_min, clamp_max)
        blended_alpha = torch.ones_like(blended_rgb[:, :1])
        blended_latent = torch.cat((blended_rgb, blended_alpha), dim=1)
        
        tensors = latents_to_images(blended_latent, (True if latent2rgb_preview and latent2rgb_preview == "true" else False))

        if mask is not None:
            blend_mask = self.transform_mask(mask, latent_a["samples"].shape)
            blended_latent = blend_mask * blended_latent + (1 - blend_mask) * latent_a["samples"]
            if set_noise_mask == 'true':
                return ({"samples": blended_latent, "noise_mask": mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))}, tensors)
            else:
                return ({"samples": blended_latent}, tensors)
        else:
            return ({"samples": blended_latent}, tensors)
            
    def blend_latents(self, latent1, latent2, mode='add', blend_percentage=0.5, blend_strength=0.5, mask=None, clamp_min=0.0, clamp_max=1.0):
        blend_func = blending_modes.get(mode)
        if blend_func is None:
            raise ValueError(f"Unsupported blending mode. Please choose from the supported modes: {', '.join(list(blending_modes.keys()))}")
        
        blend_factor1 = blend_percentage
        blend_factor2 = 1 - blend_percentage
        blended_latent = blend_func(latent1, latent2, blend_strength * blend_factor1)

        if normalize and normalize == "true":
            blended_latent = normalize(blended_latent, clamp_min, clamp_max)
        return blended_latent

    def transform_mask(self, mask, shape):
        mask = mask.view(-1, 1, mask.shape[-2], mask.shape[-1])
        resized_mask = torch.nn.functional.interpolate(mask, size=(shape[2], shape[3]), mode="bilinear")
        expanded_mask = resized_mask.expand(-1, shape[1], -1, -1)
        if expanded_mask.shape[0] < shape[0]:
            expanded_mask = expanded_mask.repeat((shape[0] - 1) // expanded_mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        del mask, resized_mask
        return expanded_mask

# IMAGES TO LATENTS

class PPFNImageAsLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "resampling": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
            },
        }

    RETURN_TYPES = ("LATENT","IMAGE",)
    RETURN_NAMES = ("latents", "images")
    FUNCTION = "image_latent"
    
    CATEGORY = "latent/util"
    
    def image_latent(self, images, resampling):

        if images.shape[-1] != 4:
            ones_channel = torch.ones(images.shape[:-1] + (1,), dtype=images.dtype, device=images.device)
            images = torch.cat((images, ones_channel), dim=-1)
        
        latents = images.permute(0, 3, 1, 2)
        latents = F.interpolate(latents, size=((images.shape[1] // 8), (images.shape[2] // 8)), mode=resampling)
        
        return {'samples': latents}, images

# LATENTS TO CPU

class PPFNLatentToCPU:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "latent_to_cpu"
    
    CATEGORY = "Power Noise Suite/Latent/Util"
    
    def latent_to_cpu(self, latents):
        return ({'samples': latents['samples'].to(device="cpu")}, )


# LATENT ADJUSTMENT

class PPFNLatentAdjustment:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "brightness": ("FLOAT", {"default": 1.0, "max": 2.0, "min": -1.0, "step": 0.001}),
                "contrast": ("FLOAT", {"default": 1.0, "max": 2.0, "min": -1.0, "step": 0.001}),
                "saturation": ("FLOAT", {"default": 1.0, "max": 2.0, "min": 0.0, "step": 0.001}),
                "exposure": ("FLOAT", {"default": 0.0, "max": 2.0, "min": -1.0, "step": 0.001}),
                "alpha_sharpen": ("FLOAT", {"default": 0.0, "max": 10.0, "min": 0.0, "step": 0.01}),
                "high_pass_radius": ("FLOAT", {"default": 0.0, "max": 1024, "min": 0.0, "step": 0.01}),
                "high_pass_strength": ("FLOAT", {"default": 1.0, "max": 2.0, "min": 0.0, "step": 0.01}),
                "clamp_min": ("FLOAT", {"default": 0.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "clamp_max": ("FLOAT", {"default": 1.0, "max": 10.0, "min": -10.0, "step": 0.01}),

            },
            "optional": {
                "latent2rgb_preview": (["false", "true"],),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latents", "previews")
    FUNCTION = "adjust_latent"

    CATEGORY = "Power Noise Suite/Latent/Adjustements"

    def adjust_latent(self, latents, brightness, contrast, saturation, exposure, alpha_sharpen, high_pass_radius, high_pass_strength, clamp_min, clamp_max, latent2rgb_preview=False):
        original_latents = latents['samples']

        r, g, b, a = original_latents[:, 0:1], original_latents[:, 1:2], original_latents[:, 2:3], original_latents[:, 3:4]

        r = (r - 0.5) * contrast + 0.5 + (brightness - 1.0)
        g = (g - 0.5) * contrast + 0.5 + (brightness - 1.0)
        b = (b - 0.5) * contrast + 0.5 + (brightness - 1.0)

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        r = (r - gray) * saturation + gray
        g = (g - gray) * saturation + gray
        b = (b - gray) * saturation + gray

        r = r * (2 ** exposure)
        g = g * (2 ** exposure)
        b = b * (2 ** exposure)
        
        latents = torch.cat((r, g, b, a), dim=1)
        
        if alpha_sharpen > 0:
            latents = sharpen_latents(latents, alpha_sharpen)
            
        if high_pass_radius > 0:
            latents = high_pass_latents(latents, high_pass_radius, high_pass_strength)
        
        if clamp_min != 0:
            latents = normalize(latents, target_min=clamp_min)
        if clamp_max != 1:
            latents = normalize(latents, target_max=clamp_max)
        if clamp_min != 0 and clamp_max != 1.0:
            latents = normalize(latents, target_min=clamp_min, target_max=clamp_max)

        tensors = latents_to_images(latents, (True if latent2rgb_preview and latent2rgb_preview == "true" else False))

        return {'samples': latents}, tensors

# POWER KSAMPLER ADVANCED
        
class PPFNKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        pln = PowerLawNoise('cpu')
        noise_types = pln.get_noise_types()
        noise_types.append('vanilla_comfy')
        samplers = [x for x in comfy.samplers.KSampler.SAMPLERS if x in ['euler_ancestral', 'dpm_2_ancestral', 'dpmpp_2s_ancestral', 'dpm_fast', 'dpm_adaptive']]
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (samplers,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "add_noise": (["enable", "disable"],),
                "return_with_leftover_noise": (["disable", "enable"],),
            },
            "optional": {
                "noise_type": (noise_types,),
                "frequency": ("FLOAT", {"default": 3.141592653589793, "max": 1024.0, "min": 0.001, "step": 0.001}),
                "attenuation": ("FLOAT", {"default": 0.75, "max": 1024.0, "min": 0.001, "step": 0.001}),
                "sigma_tolerance": ("FLOAT", {"default": 0.5, "max": 1.0, "min": 0.0, "step": 0.001}),
                "ppf_settings": ("PPF_SETTINGS",),
                "ch_settings": ("CH_SETTINGS",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Power Noise Suite/Sampling"

    def sample(self, model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0, noise_type='grey', frequency=11.6, attenuation=0.75, sigma_tolerance=1.0, ppf_settings=None, ch_settings=None):
    
        def ppfn_noise_sampler(x):
            seed_base = seed
            noise_idx = [0]
            sigma_base = [None]
            height = int(x.shape[2] * 8)
            width = int(x.shape[3] * 8)
            method = noise_type
            freq = frequency
            atten = attenuation
            sigma_tol = sigma_tolerance
            ppfs = ppf_settings
            chs = ch_settings

            def ppfn_return_noise(seed, x, sigma, sigma_tol, method, freq, atten, blending_modes, ppfs, chs):
                seed = seed_base + noise_idx[0]
                rand_noise = torch.randn_like(x)
                
                if sigma_tol == 0.0:
                    return rand_noise

                sigma_min = 0
                sigma_max = 14.614643096923828
                scaled_sigma = ((sigma.item() - sigma_min) / (sigma_max - sigma_min)) * sigma_tol

                if not ppfs and not chs:
                    power_law = PowerLawNoise(device=rand_noise.device)
                    noise = power_law(1, width, height, noise_type=method, frequency=frequency, attenuation=attenuation, seed=seed)
                elif ppfs:
                    power_fractal = PPFNoiseNode()
                    noise = power_fractal.power_fractal_latent(1, width, height, 'nearest', ppfs['X'], ppfs['Y'], ppfs['Z'], ppfs['evolution'], ppfs['frame'], ppfs['scale'], ppfs['octaves'], ppfs['persistence'], ppfs['lacunarity'], ppfs['exponent'], ppfs['brightness'], ppfs['contrast'], 0.0, 1.0, seed, device=('cuda' if torch.cuda.is_available() else 'cpu'), optional_vae=None)[0]['samples'].to(device=rand_noise.device)
                elif chs:
                    ch_fractal = PPFNCrossHatchNode()
                    noise = ch_fractal.cross_hatch(1, width, height, 'nearest', chs['frequency'], chs['octaves'], chs['persistence'], chs['color_tolerance'], chs['num_colors'], chs['angle_degrees'], chs['brightness'], chs['contrast'], chs['blur'], 0.0, 1.0, seed, device=('cuda' if torch.cuda.is_available() else 'cpu'), optional_vae=None)[0]['samples'].to(device=rand_noise.device)
                
                noise = noise.permute(0, 3, 1, 2)
                noise = F.interpolate(noise, size=(x.shape[2], x.shape[3]), mode='nearest')
                noise = noise[:, :rand_noise.shape[1], :, :]

                if not ppfs and not chs:
                    alpha = torch.ones((1, x.shape[2], x.shape[3], 1), dtype=x.dtype, device=x.device).permute(0, 3, 1, 2)
                    noise = torch.cat((noise, alpha), dim=1)

                blended_noise = rand_noise + 0.25 * (blending_modes["bislerp"](rand_noise.to(device=rand_noise.device), noise.to(device=rand_noise.device), scaled_sigma) - rand_noise)

                noise_idx[0] += 1

                return blended_noise

            return lambda sigma, sigma_next, **kwargs: ppfn_return_noise(seed_base + noise_idx[0], x, sigma, sigma_tol, method, freq, atten, blending_modes, ppfs, chs)


        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        else:
            if noise_type != "vanilla_comfy":
                dns = comfy.k_diffusion.sampling.default_noise_sampler
                comfy.k_diffusion.sampling.default_noise_sampler = ppfn_noise_sampler
           
        result = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        
        if noise_type != "vanilla_comfy":
            comfy.k_diffusion.sampling.default_noise_sampler = dns
        
        return result

# PERLIN POWER FRACTAL SETTINGS

class PPFNoiseSettings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "Y": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "Z": ("FLOAT", {"default": 0, "max": 99999999, "min": -99999999, "step": 0.01}),
                "evolution": ("FLOAT", {"default": 0.0, "max": 1.0, "min": 0.0, "step": 0.01}),
                "frame": ("INT", {"default": 0, "max": 99999999, "min": 0, "step": 1}),
                "scale": ("FLOAT", {"default": 5, "max": 2048, "min": 2, "step": 0.01}),
                "octaves": ("INT", {"default": 8, "max": 8, "min": 1, "step": 1}),
                "persistence": ("FLOAT", {"default": 1.5, "max": 23.0, "min": 0.01, "step": 0.01}),
                "lacunarity": ("FLOAT", {"default": 2.0, "max": 99.0, "min": 0.01, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 4.0, "max": 38.0, "min": 0.01, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("PPF_SETTINGS",)
    RETURN_NAMES = ("ppf_settings",)
    FUNCTION = "power_fractal_settings"

    CATEGORY = "Power Noise Suite/Sampling/Settings"
    
    def power_fractal_settings(self, X, Y, Z, evolution, frame, scale, octaves, persistence, lacunarity, exponent, brightness, contrast):
        return ({"X": X, "Y": Y, "Z": Z, "evolution": evolution, "frame": frame, "scale": scale, "octaves": octaves, "persistence": persistence, "lacunarity": lacunarity, "exponent": exponent, "brightness": brightness, "contrast": contrast},)
        
# CROSS-HATCH POWER FRACTAL SETTINGS

class PPFNCrossHatchSettings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frequency": ("FLOAT", {"default": 320.0, "max": 1024.0, "min": 0.001, "step": 0.001}),
                "octaves": ("INT", {"default": 12, "max": 32, "min": 1, "step": 1}),
                "persistence": ("FLOAT", {"default": 1.5, "max": 2.0, "min": 0.001, "step": 0.001}),
                "num_colors": ("INT", {"default": 16, "max": 256, "min": 2, "step": 1}),
                "color_tolerance": ("FLOAT", {"default": 0.05, "max": 1.0, "min": 0.001, "step": 0.001}),
                "angle_degrees": ("FLOAT", {"default": 45.0, "max": 360.0, "min": 0.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.001}),
                "contrast": ("FLOAT", {"default": 0.0, "max": 1.0, "min": -1.0, "step": 0.001}),
                "blur": ("FLOAT", {"default": 2.5, "max": 1024, "min": 0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CH_SETTINGS",)
    RETURN_NAMES = ("ch_settings",)
    FUNCTION = "cross_hatch_settings"

    CATEGORY = "Power Noise Suite/Sampling/Settings"
    
    def cross_hatch_settings(self, frequency, octaves, persistence, color_tolerance, num_colors, angle_degrees, brightness, contrast, blur):
        return ({"frequency": frequency, "octaves": octaves, "persistence": persistence, "color_tolerance": color_tolerance, "num_colors": num_colors, "angle_degrees": angle_degrees, "brightness": brightness, "contrast": contrast, "blur": blur},)

        
NODE_CLASS_MAPPINGS = {
    "Blend Latents (PPF Noise)": PPFNBlendLatents,
    "Cross-Hatch Power Fractal (PPF Noise)": PPFNCrossHatchNode,
    "Cross-Hatch Power Fractal Settings (PPF Noise)": PPFNCrossHatchSettings,
    "Images as Latents (PPF Noise)": PPFNImageAsLatent,
    "Latent Adjustment (PPF Noise)": PPFNLatentAdjustment,
    "Latents to CPU (PPF Noise)": PPFNLatentToCPU,
    "Linear Cross-Hatch Power Fractal (PPF Noise)": PPFNLinearCrossHatchNode,
    "Perlin Power Fractal Latent (PPF Noise)": PPFNoiseNode,
    "Perlin Power Fractal Settings (PPF Noise)": PPFNoiseSettings,
    "Power-Law Noise (PPF Noise)": PPFNPowerLawNoise,
    "Power KSampler Advanced (PPF Noise)": PPFNKSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Blend Latents (PPF Noise)": "Blend Latents 🦚",
    "Cross-Hatch Power Fractal (PPF Noise)": "Cross-Hatch Power Fractal 🦚",
    "Cross-Hatch Power Fractal Settings (PPF Noise)": "Cross-Hatch Power Fractal Settings 🦚",
    "Images as Latents (PPF Noise)": "Images as Latents 🦚",
    "Latent Adjustment (PPF Noise)": "Latent Adjustment 🦚",
    "Latents to CPU (PPF Noise)": "Latents to CPU 🦚",
    "Linear Cross-Hatch Power Fractal (PPF Noise)": "Linear Cross-Hatch Power Fractal 🦚",
    "Perlin Power Fractal Latent (PPF Noise)": "Perlin Power Fractal Noise 🦚",
    "Perlin Power Fractal Settings (PPF Noise)": "Perlin Power Fractal Settings 🦚",
    "Power-Law Noise (PPF Noise)": "Power-Law Noise 🦚",
    "Power KSampler Advanced (PPF Noise)": "Power KSampler Advanced 🦚",
}

