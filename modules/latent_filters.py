import torch
import torch.nn.functional as F
import math

from .latent_util import normalize
    
def sharpen_latents(latent, alpha=1.5):
    """
    Sharpen the input latent tensor.

    Args:
        latent (torch.Tensor): The input latent tensor.
        alpha (float, optional): The sharpening strength. Defaults to 1.5.

    Returns:
        torch.Tensor: The sharpened latent tensor.
    """
    # Ensure that all operations are done on the input 'latent' device
    device = latent.device
    
    sharpen_kernel = torch.tensor([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]], dtype=torch.float32, device=device)
    
    sharpen_kernel = sharpen_kernel.view(1, 1, 3, 3)
    sharpen_kernel /= sharpen_kernel.sum()
    
    sharpened_tensors = []
    
    for channel in range(latent.size(1)):
        channel_tensor = latent[:, channel, :, :].unsqueeze(1)
        sharpened_channel = F.conv2d(channel_tensor, sharpen_kernel, padding=1)
        sharpened_tensors.append(sharpened_channel)
    
    sharpened_tensor = torch.cat(sharpened_tensors, dim=1)
    sharpened_tensor = latent + alpha * sharpened_tensor
    
    padding_size = (sharpen_kernel.shape[-1] - 1) // 2
    sharpened_tensor = sharpened_tensor[:, :, padding_size:-padding_size, padding_size:-padding_size]
    sharpened_tensor = torch.clamp(sharpened_tensor, 0, 1)
    sharpened_tensor = F.interpolate(sharpened_tensor, size=(latent.size(2), latent.size(3)), mode='nearest')
    
    return sharpened_tensor

def high_pass_latents(latent, radius=3, strength=1.0):
    """
    Apply a high-pass filter to the input latent image

    Args:
        latent (torch.Tensor): The input latent tensor.
        radius (int, optional): The radius of the high-pass filter. Defaults to 3.
        strength (float, optional): The strength of the high-pass filter. Defaults to 1.0.

    Returns:
        torch.Tensor: The high-pass filtered latent tensor.
    """
    sigma = radius / 3.0
    kernel_size = radius * 2 + 1
    x = torch.arange(-radius, radius+1).float().to(latent.device)
    gaussian_kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    high_pass_overlays = []

    for channel in range(latent.size(1)):
        channel_tensor = latent[:, channel, :, :].unsqueeze(1)

        weight_h = gaussian_kernel.view(1, 1, 1, -1)
        weight_v = gaussian_kernel.view(1, 1, -1, 1)

        input_blur_h = F.conv2d(
            channel_tensor,
            weight_h,
            padding=0,
        )
        input_blur_v = F.conv2d(
            input_blur_h,
            weight_v,
            padding=0,
        )

        input_blur_h = F.interpolate(input_blur_h, size=channel_tensor.shape[-2:], mode='bilinear', align_corners=False)
        input_blur_v = F.interpolate(input_blur_v, size=channel_tensor.shape[-2:], mode='bilinear', align_corners=False)

        high_pass_component = channel_tensor - input_blur_v

        high_pass_channel = channel_tensor + strength * high_pass_component
        high_pass_channel = torch.clamp(high_pass_channel, 0, 1)
        
        high_pass_overlays.append(high_pass_channel)

    high_pass_overlay = torch.cat(high_pass_overlays, dim=1)

    return high_pass_overlay

def hslerp(a, b, t):
    """
    Perform Hybrid Spherical Linear Interpolation (HSLERP) between two tensors.

    This function combines two input tensors `a` and `b` using HSLERP, which is a specialized
    interpolation method for smooth transitions between orientations or colors.

    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.

    Returns:
        tensor: The result of HSLERP interpolation between `a` and `b`.

    Note:
        HSLERP provides smooth transitions between orientations or colors, particularly useful
        in applications like image processing and 3D graphics.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")

    num_channels = a.size(1)
    
    interpolation_tensor = torch.zeros(1, num_channels, 1, 1, device=a.device, dtype=a.dtype)
    interpolation_tensor[0, 0, 0, 0] = 1.0

    result = (1 - t) * a + t * b

    if t < 0.5:
        result += (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor
    else:
        result -= (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor

    return result

blending_modes = {

    # Args:
    #   - a (tensor): Latent input 1
    #   - b (tensor): Latent input 2
    #   - t (float): Blending factor

    # Linearly combines the two input tensors a and b using the parameter t.
    'add': lambda a, b, t: (a * t + b * t),

    # Interpolates between tensors a and b using normalized linear interpolation.
    'bislerp': lambda a, b, t: normalize((1 - t) * a + t * b),

    # Simulates a brightening effect by dividing a by (1 - b) with a small epsilon to avoid division by zero.
    'color dodge': lambda a, b, t: a / (1 - b + 1e-6),

    # Interpolates between tensors a and b using cosine interpolation.
    'cosine interp': lambda a, b, t: (a + b - (a - b) * torch.cos(t * torch.tensor(math.pi))) / 2,

    # Interpolates between tensors a and b using cubic interpolation.
    'cuberp': lambda a, b, t: a + (b - a) * (3 * t ** 2 - 2 * t ** 3),

    # Computes the absolute difference between tensors a and b, scaled by t.
    'difference': lambda a, b, t: normalize(abs(a - b) * t),

    # Combines tensors a and b using an exclusion formula, scaled by t.
    'exclusion': lambda a, b, t: normalize((a + b - 2 * a * b) * t),

    # Interpolates between tensors a and b using normalized linear interpolation,
    # with a twist when t is greater than or equal to 0.5.
    'hslerp': hslerp,

    # Simulates a glowing effect by applying a formula based on the input tensors a and b, scaled by t.
    'glow': lambda a, b, t: torch.where(a <= 1, a ** 2 / (1 - b + 1e-6), b * (a - 1) / (a + 1e-6)),

    # Combines tensors a and b using the Hard Light formula, scaled by t.
    'hard light': lambda a, b, t: (2 * a * b * (a < 0.5).float() + (1 - 2 * (1 - a) * (1 - b)) * (a >= 0.5).float()) * t,

    # Adds tensor b to tensor a, scaled by t.
    'inject': lambda a, b, t: a + b * t,

    # Interpolates between tensors a and b using linear interpolation.
    'lerp': lambda a, b, t: (1 - t) * a + t * b,

    # Simulates a brightening effect by adding tensor b to tensor a, scaled by t.
    'linear dodge': lambda a, b, t: normalize(a + b * t),

    # Combines tensors a and b using the Linear Light formula.
    'linear light': lambda a, b, t: torch.where(b <= 0.5, a + 2 * b - 1, a + 2 * (b - 0.5)),

    # Multiplies tensors a and b element-wise, scaled by t.
    'multiply': lambda a, b, t: normalize(a * t * b * t),

    # Combines tensors a and b using the Overlay formula, with a twist when b is less than 0.5.
    'overlay': lambda a, b, t: (2 * a * b + a**2 - 2 * a * b * a) * t if torch.all(b < 0.5) else (1 - 2 * (1 - a) * (1 - b)) * t,

    # Combines tensors a and b using the Pin Light formula.
    'pin light': lambda a, b, t: torch.where(b <= 0.5, torch.min(a, 2 * b), torch.max(a, 2 * b - 1)),

    # Generates random values and combines tensors a and b with random weights, scaled by t.
    'random': lambda a, b, t: normalize(torch.rand_like(a) * a * t + torch.rand_like(b) * b * t),

    # Combines tensors a and b using the Reflect formula.
    'reflect': lambda a, b, t: torch.where(b <= 1, b ** 2 / (1 - a + 1e-6), a * (b - 1) / (b + 1e-6)),

    # Combines tensors a and b using the Screen formula, scaled by t.
    'screen': lambda a, b, t: normalize(1 - (1 - a) * (1 - b) * (1 - t)),

    # Interpolates between tensors a and b using spherical linear interpolation (SLERP).
    'slerp': lambda a, b, t: normalize(((a * torch.sin((1 - t) * torch.acos(torch.clamp(torch.sum(a * b, dim=1), -1.0, 1.0))) + b * torch.sin(t * torch.acos(torch.clamp(torch.sum(a * b, dim=1), -1.0, 1.0)))) / torch.sin(torch.acos(torch.clamp(torch.sum(a * b, dim=1), -1.0, 1.0))))),

    # Subtracts tensor b from tensor a, scaled by t.
    'subtract': lambda a, b, t: (a * t - b * t),

    # Combines tensors a and b using the Vivid Light formula.
    'vivid light': lambda a, b, t: torch.where(b <= 0.5, a / (1 - 2 * b + 1e-6), (a + 2 * b - 1) / (2 * (1 - b) + 1e-6)),
    
}