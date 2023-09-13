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
    
def noise_sigma_scale(noise, sigma_min, sigma_max):
    """
    Scales the input noise values to a specified sigma range.

    Args:
        noise (Tensor): The input noise tensor.
        sigma_min (float): The lower bound of the sigma range.
        sigma_max (float): The upper bound of the sigma range.

    Returns:
        Tensor: The scaled noise tensor within the specified sigma range.
    """
    normalized_noise = normalize(noise)
    scaled_noise = sigma_min + (sigma_max - sigma_min) * normalized_noise
    
    return scaled_noise
    
def within_percentage_range(num, total, percentage, tolerance_factor):
    """
    Check if a number is within a specified percentage of a total accounting for a tolerance.

    Args:
        numb (float): The number to check.
        total (float): The total value against which to compare.
        percentage (float): The desired percentage (between 0 and 1, e.g., 0.5 for 50%).
        tolerance_factor (float): The tolerance factor as a percentage (between 0 and 1, e.g., 0.1 for 10%).

    Returns:
        bool: True if the number is within the specified range, False otherwise.
    """
    lower_bound = total * (percentage - tolerance_factor)
    upper_bound = total * (percentage + tolerance_factor)
    return lower_bound <= num <= upper_bound
    
    
def scale_from_perentage(number, total, percentage, tolerance_factor):
    """
    Calculate a scaling factor within a percentage tolerance range.

    This function calculates a scaling factor that is 1.0 at the exact percentage value
    and scales down towards 0.0 within the tolerance_factor range.

    Args:
        number (float): The number to check against the percentage.
        total (float): The total value against which the percentage is calculated.
        percentage (float): The target percentage value (between 0.0 and 1.0).
        tolerance_factor (float): The tolerance factor for the percentage range.

    Returns:
        float: A scaling factor that indicates how close the number is to the target percentage
               within the specified tolerance range. A value of 1.0 indicates an exact match,
               while values closer to 0.0 indicate deviation from the target within the tolerance range.
    """

    lower_bound = percentage - tolerance_factor
    upper_bound = percentage + tolerance_factor

    if lower_bound <= (number / total) <= upper_bound:
        return 1.0
    if lower_bound <= percentage and (number / total) >= lower_bound:
        return (number / (total / 2)) * 0.25
    elif upper_bound >= percentage and (number / total) <= upper_bound:
        return (number / (total / 2)) * 0.25