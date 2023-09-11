import torch
import torch.nn.functional as F

def normalize(latent, target_min=None, target_max=None):
    """
    Normalize a tensor `latent` between `target_min` and `target_max`.

    Args:
        latent (torch.Tensor): The input tensor to be normalized.
        target_min (float, optional): The minimum value after normalization. 
            - When `None` min will be tensor min range value.
        target_max (float, optional): The maximum value after normalization. 
            - When `None` max will be tensor max range value.

    Returns:
        torch.Tensor: The normalized tensor
    """
    min_val = latent.min()
    max_val = latent.max()
    
    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val
        
    normalized = (latent - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled
        
def latents_to_images(latents, l2rgb=False):
    """
    Convert latent representation to RGB images. Convert latent to RGB color values, 
    or return the latent directly a tensor image.

    Args:
        latents (torch.Tensor): The input elatent tensor.
        l2rgb (bool, optional): Whether to apply L2RGB conversion. Defaults to False.

    Returns:
        torch.Tensor: The resulting tensor containing RGB images.
    """
    if l2rgb:
        l2rgb = torch.tensor([
            #   R     G      B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], device=latents.device)
        tensors = torch.einsum('...lhw,lr->...rhw', latents.float(), l2rgb)
        tensors = ((tensors + 1) / 2).clamp(0, 1)
        tensors = tensors.movedim(1, -1)
    else:
        tensors = latents.permute(0, 2, 3, 1)
        
    return tensors
