import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fft as fft
import math
import random

from .latent_util import normalize
from .latent_filters import blending_modes
    
# PERLIN POWER FRACTAL
    
class PerlinPowerFractal(nn.Module):
    """
    Generate a batch of images with a Perlin power fractal effect.

    Args:
        width (int): Width of each tensor in pixels. Specifies the width of the generated image. Range: [64, 8192].
        height (int): Height of each image in pixels. Specifies the height of the generated image. Range: [64, 8192].
        batch_size (int): Number of noisy tensors to generate in the batch. Determines the number of images generated simultaneously. Range: [1, 64].
        X (float): X-coordinate offset for noise sampling. Shifts the noise pattern along the X-axis. Range: [-99999999, 99999999].
        Y (float): Y-coordinate offset for noise sampling. Shifts the noise pattern along the Y-axis. Range: [-99999999, 99999999].
        Z (float): Z-coordinate offset for noise sampling. Shifts the noise pattern along the Z-axis for time evolution. Range: [-99999999, 99999999].
        frame (int): The current frame number for time evolution. Controls how the noise pattern evolves over time. Range: [0, 99999999].
        evolution_factor (float): Factor controlling time evolution. Determines how much the noise evolves over time based on the batch index. Range: [0.0, 1.0].
        octaves (int): Number of octaves for fractal generation. Controls the level of detail and complexity in the output. More octaves create finer details. Range: [1, 8].
        persistence (float): Persistence parameter for fractal generation. Determines the amplitude decrease of each octave. Higher values amplify the effect of each octave. Range: [0.01, 23.0].
        lacunarity (float): Lacunarity parameter for fractal generation. Controls the increase in frequency from one octave to the next. Higher values result in more variation between octaves. Range: [0.01, 99.0].
        exponent (float): Exponent applied to the noise values. Adjusting this parameter controls the overall intensity and contrast of the output. Higher values increase intensity and contrast. Range: [0.01, 38.0].
        scale (float): Scaling factor for frequency of noise. Larger values produce smaller, more detailed patterns, while smaller values create larger patterns. Range: [2, 2048].
        brightness (float): Adjusts the overall brightness of the generated noise.
            - -1.0 makes the noise completely black.
            - 0.0 has no effect on brightness.
            - 1.0 makes the noise completely white. Range: [-1.0, 1.0].
        contrast (float): Adjusts the contrast of the generated noise.
            - -1.0 reduces contrast, enhancing the difference between dark and light areas.
            - 0.0 has no effect on contrast.
            - 1.0 increases contrast, enhancing the difference between dark and light areas. Range: [-1.0, 1.0].
        seed (int, optional): Seed for random number generation. If None, uses random seeds for each batch. Controls the reproducibility of the generated noise. Range: [0, 0xffffffffffffffff].

    Methods:
        forward(batch_size, X, Y, Z, frame, device='cpu', evolution_factor=0.1, octaves=4, persistence=0.5, lacunarity=2.0, exponent=4.0, scale=100, brightness=0.0, contrast=0.0, seed=None, min_clamp=0.0, max_clamp=1.0):
            Generate the batch of images with Perlin power fractal effect.

    Returns:
        torch.Tensor: A tensor containing the generated images in the shape (batch_size, height, width, 1).
    """
    def __init__(self, width, height):
        """
        Initialize the PerlinPowerFractal.

        Args:
            width (int): Width of each tensor in pixels. Range: [64, 8192].
            height (int): Height of each image in pixels. Range: [64, 8192].
        """
        super(PerlinPowerFractal, self).__init__()
        self.width = width
        self.height = height

    def forward(self, batch_size, X, Y, Z, frame, device='cpu', evolution_factor=0.1,
                octaves=4, persistence=0.5, lacunarity=2.0, exponent=4.0, scale=100,
                brightness=0.0, contrast=0.0, seed=None, min_clamp=0.0, max_clamp=1.0):
        """
        Generate a batch of images with Perlin power fractal effect.

        Args:
            batch_size (int): Number of noisy tensors to generate in the batch.
            X (float): X-coordinate offset for noise sampling.
            Y (float): Y-coordinate offset for noise sampling.
            Z (float): Z-coordinate offset for noise sampling.
            frame (int): The current frame number for time evolution.
            device (str, optional): The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            evolution_factor (float, optional): Factor controlling time evolution. Default is 0.1.
            octaves (int, optional): Number of octaves for fractal generation. Default is 4.
            persistence (float, optional): Persistence parameter for fractal generation. Default is 0.5.
            lacunarity (float, optional): Lacunarity parameter for fractal generation. Default is 2.0.
            exponent (float, optional): Exponent applied to the noise values. Default is 4.0.
            scale (float, optional): Scaling factor for frequency of noise. Default is 100.
            brightness (float, optional): Adjusts the overall brightness of the generated noise. Default is 0.0.
            contrast (float, optional): Adjusts the contrast of the generated noise. Default is 0.0.
            seed (int, optional): Seed for random number generation. If None, uses random seeds for each batch. Default is None.
            min_clamp (float, optional): Minimum value to clamp the pixel values to. Default is 0.0.
            max_clamp (float, optional): Maximum value to clamp the pixel values to. Default is 1.0.

        Returns:
            torch.Tensor: A tensor containing the generated images in the shape (batch_size, height, width, 1).
        """

        def fade(t):
            return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

        def lerp(t, a, b):
            return a + t * (b - a)

        def grad(hash, x, y, z):
            h = hash & 15
            u = torch.where(h < 8, x, y)
            v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))
            return torch.where(h & 1 == 0, u, -u) + torch.where(h & 2 == 0, v, -v)

        def noise(x, y, z, p):
            X = (x.floor() % 255).to(torch.int32)
            Y = (y.floor() % 255).to(torch.int32)
            Z = (z.floor() % 255).to(torch.int32)

            x -= x.floor()
            y -= y.floor()
            z -= z.floor()

            u = fade(x)
            v = fade(y)
            w = fade(z)

            A = p[X] + Y
            AA = p[A] + Z
            AB = p[A + 1] + Z
            B = p[X + 1] + Y
            BA = p[B] + Z
            BB = p[B + 1] + Z

            r = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                              lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
                     lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                              lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))))

            return r

        device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'

        unique_seed = seed if seed is not None else torch.randint(0, 10000, (1,)).item()
        torch.manual_seed(unique_seed)

        p = torch.randperm(max(self.width, self.height) ** 2, dtype=torch.int32, device=device)
        p = torch.cat((p, p))

        noise_map = torch.zeros(batch_size, self.height, self.width, dtype=torch.float32, device=device)

        X = torch.arange(self.width, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) + X
        Y = torch.arange(self.height, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(0) + Y
        Z = evolution_factor * torch.arange(batch_size, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(1) + Z + frame

        for octave in range(octaves):
            frequency = lacunarity ** octave
            amplitude = persistence ** octave

            nx = X / scale * frequency
            ny = Y / scale * frequency
            nz = (Z + frame * evolution_factor) / scale * frequency

            noise_values = noise(nx, ny, nz, p) * (amplitude ** exponent)

            noise_map += noise_values.squeeze(-1) * amplitude

        noise_map = normalize(noise_map, min_clamp, max_clamp)

        latent = (noise_map + brightness) * (1.0 + contrast)
        latent = normalize(latent)
        latent = latent.unsqueeze(-1)

        return latent
    
# CROSS-HATCH POWER FRACTAL
    
class CrossHatchPowerFractal(nn.Module):
    """
    Generate a batch of crosshatch-patterned images with a power fractal effect.

    Args:
        width (int): Width of each image in pixels.
        height (int): Height of each image in pixels.
        frequency (int, optional): Frequency of the crosshatch pattern. Default is 320.
        octaves (int, optional): Number of octaves for fractal generation. Controls the level of detail and complexity in the output. Default is 12.
        persistence (float, optional): Persistence parameter for fractal generation. Determines the amplitude decrease of each octave. Default is 1.5.
        num_colors (int, optional): Number of colors to map the generated noise to. Default is 16.
        color_tolerance (float, optional): Color tolerance for mapping noise values to colors. Default is 0.05.
        angle_degrees (float, optional): Angle in degrees for the crosshatch pattern orientation. Default is 45.
        blur (int, optional): Amount of blur to apply to the generated image. Default is 2.
        brightness (float, optional): Adjusts the overall brightness of the generated images. Default is 0.0.
        contrast (float, optional): Adjusts the contrast of the generated images. Default is 0.0.
        clamp_min (float, optional): Minimum value to clamp the pixel values to. Default is 0.0.
        clamp_max (float, optional): Maximum value to clamp the pixel values to. Default is 1.0.

    Methods:
        forward(batch_size=1, device='cpu', seed=1):
            Generate a batch of crosshatch-patterned images.

    Returns:
        torch.Tensor: A tensor containing the generated images in the shape (batch_size, height, width, 3).
    """
    def __init__(self, width, height, frequency=320, octaves=12, persistence=1.5, num_colors=16, color_tolerance=0.05, angle_degrees=45, blur=2, brightness=0.0, contrast=0.0, clamp_min=0.0, clamp_max=1.0):
        """
        Initialize the CrossHatchPowerFractal.

        Args:
            width (int): Width of each image in pixels.
            height (int): Height of each image in pixels.
            frequency (int, optional): Frequency of the crosshatch pattern. Default is 320.
            octaves (int, optional): Number of octaves for fractal generation. Default is 12.
            persistence (float, optional): Persistence parameter for fractal generation. Default is 1.5.
            num_colors (int, optional): Number of colors to map the generated noise to. Default is 16.
            color_tolerance (float, optional): Color tolerance for mapping noise values to colors. Default is 0.05.
            angle_degrees (float, optional): Angle in degrees for the crosshatch pattern orientation. Default is 45.
            blur (int, optional): Amount of blur to apply to the generated image. Default is 2.
            brightness (float, optional): Adjusts the overall brightness of the generated images. Default is 0.0.
            contrast (float, optional): Adjusts the contrast of the generated images. Default is 0.0.
            clamp_min (float, optional): Minimum value to clamp the pixel values to. Default is 0.0.
            clamp_max (float, optional): Maximum value to clamp the pixel values to. Default is 1.0.
        """
        super(CrossHatchPowerFractal, self).__init__()
        self.width = width
        self.height = height
        self.frequency = frequency
        self.num_octaves = octaves
        self.persistence = persistence
        self.angle_radians = math.radians(angle_degrees)
        self.num_colors = num_colors
        self.color_tolerance = color_tolerance
        self.blur = blur
        self.brightness = brightness
        self.contrast = contrast
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, batch_size=1, device='cpu', seed=1):
        """
        Generate a batch of crosshatch-patterned images.

        Args:
            batch_size (int, optional): Number of images to generate. Default is 1.
            device (str, optional): The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            seed (int, optional): Seed for random number generation. Default is 1.

        Returns:
            torch.Tensor: A tensor containing the generated crosshatch-patterned images in the shape (batch_size, height, width, 3).
        """
        device_index = [torch.cuda.current_device()] if device.startswith('cuda') else None

        with torch.random.fork_rng(devices=device_index):
            x = torch.linspace(0, 1, self.width, dtype=torch.float32, device=device)
            y = torch.linspace(0, 1, self.height, dtype=torch.float32, device=device)
            x, y = torch.meshgrid(x, y, indexing="ij")

            batched_noises = []

            for i in range(batch_size):
                batch_seed = int(seed + i)
                noise = torch.zeros(self.width, self.height, device=device)

                for octave in range(self.num_octaves):
                    frequency = self.frequency * 2 ** octave
                    octave_noise = self.generate_octave(x, y, frequency)
                    noise += octave_noise * self.persistence ** octave

                noise = (noise - noise.min()) / (noise.max() - noise.min())
                colored_noise = self.apply_color_mapping(noise, device, batch_seed)
                colored_noise = colored_noise.cpu()

                r_channel = colored_noise[:, :, 0]
                g_channel = colored_noise[:, :, 1]
                b_channel = colored_noise[:, :, 2]

                kernel_size = int(self.blur * 2 + 1)
                uniform_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)

                blurred_r = F.conv2d(r_channel.unsqueeze(0).unsqueeze(0), uniform_kernel, padding=int(self.blur))
                blurred_g = F.conv2d(g_channel.unsqueeze(0).unsqueeze(0), uniform_kernel, padding=int(self.blur))
                blurred_b = F.conv2d(b_channel.unsqueeze(0).unsqueeze(0), uniform_kernel, padding=int(self.blur))
                blurred_noise = torch.cat((blurred_r, blurred_g, blurred_b), dim=1)
                blurred_noise = F.interpolate(blurred_noise, size=(self.height, self.width), mode='bilinear')

                batched_noises.append(blurred_noise.permute(0, 2, 3, 1))

            batched_noises = torch.cat(batched_noises, dim=0).to(device='cpu')
            batched_noises = (batched_noises + self.brightness) * (1.0 + self.contrast)
            
            return normalize(batched_noises, self.clamp_min, self.clamp_max)

    def generate_octave(self, x, y, frequency):
        """
        Generate an octave of the crosshatch pattern.

        Args:
            x (torch.Tensor): X-coordinate grid.
            y (torch.Tensor): Y-coordinate grid.
            frequency (int): Frequency of the crosshatch pattern.

        Returns:
            torch.Tensor: Octave of the crosshatch pattern.
        """
        grid_hatch_x = torch.sin(x * frequency * math.pi)
        grid_hatch_y = torch.sin(y * frequency * math.pi)

        grid_hatch_x = (grid_hatch_x - grid_hatch_x.min()) / (grid_hatch_x.max() - grid_hatch_x.min())
        grid_hatch_y = (grid_hatch_y - grid_hatch_y.min()) / (grid_hatch_y.max() - grid_hatch_y.min())
        grid_hatch = grid_hatch_x + grid_hatch_y

        return grid_hatch

    def apply_color_mapping(self, noise, device, seed):
        """
        Apply color mapping to noise values fir a consisten look

        Args:
            noise (torch.Tensor): Noise values.
            device (str): The device to use for computation ('cpu' or 'cuda').
            seed (int): Seed for random number generation.

        Returns:
            torch.Tensor: Noise values after color mapping.
        """
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        random_colors = torch.rand(self.num_colors, 3, generator=generator, dtype=torch.float32, device=device)

        noise_scaled = noise * (self.num_colors - 1)
        tolerance = self.color_tolerance * (self.num_colors - 1)
        noise_scaled_rounded = torch.round(noise_scaled)
        colored_noise = random_colors[noise_scaled_rounded.long()]

        return colored_noise

# LINEAR CROSS-HATCH POWER FRACTAL

class CrossHatchLinearPowerFractal(nn.Module):
    """
    Generate a batch of linear crosshatch-patterned images with a power fractal effect.

    Args:
        width (int): Width of each image in pixels.
        height (int): Height of each image in pixels.
        frequency (int, optional): Frequency of the crosshatch pattern. Default is 320.
        num_octaves (int, optional): Number of octaves for fractal generation. Controls the level of detail and complexity in the output. Default is 12.
        persistence (float, optional): Persistence parameter for fractal generation. Determines the amplitude decrease of each octave. Default is 1.5.
        angle_degrees (float, optional): Angle in degrees for the crosshatch pattern orientation. Default is 45.
        gain (float, optional): Gain factor for controlling the amplitude of the generated noise. Default is 0.1.
        add_noise_tolerance (float, optional): Tolerance for adding random noise to the generated pattern. Default is 0.25.
        mapping_range (int, optional): Range for mapping noise values to different levels. Default is 24.
        brightness (float, optional): Brightness adjustment factor for the generated images. Default is 0.0.
        contrast (float, optional): Contrast adjustment factor for the generated images. Default is 0.0.
        seed (int, optional): Seed for random number generation. Default is 0.

    Methods:
        forward(batch_size=1, device='cpu', seed=0):
            Generate a batch of crosshatch-patterned images.

    Returns:
        torch.Tensor: A tensor containing the generated crosshatch-patterned images in shape (batch_size, height, width, 3).
    """
    def __init__(self, width, height, frequency=320, octaves=12, persistence=1.5, angle_degrees=45, gain=0.1, add_noise_tolerance=0.25, mapping_range=24, brightness=0.0, contrast=0.0, seed=0):
        """
        Initialize the CrossHatchLinearPowerFractal.

        Args:
            width (int): Width of each image in pixels.
            height (int): Height of each image in pixels.
            frequency (int, optional): Frequency of the crosshatch pattern. Default is 320.
            num_octaves (int, optional): Number of octaves for fractal generation. Default is 12.
            persistence (float, optional): Persistence parameter for fractal generation. Default is 1.5.
            angle_degrees (float, optional): Angle in degrees for the crosshatch pattern orientation. Default is 45.
            gain (float, optional): Gain factor for controlling the amplitude of the generated noise. Default is 0.1.
            add_noise_tolerance (float, optional): Tolerance for adding random noise to the generated pattern. Default is 0.25.
            mapping_range (int, optional): Range for mapping noise values to different levels. Default is 24.
            brightness (float, optional): Brightness adjustment factor for the generated images. Default is 0.0.
            contrast (float, optional): Contrast adjustment factor for the generated images. Default is 0.0.
            seed (int, optional): Seed for random number generation. Default is 0.
        """
        super(CrossHatchLinearPowerFractal, self).__init__()
        self.width = width
        self.height = height
        self.frequency = frequency
        self.num_octaves = octaves
        self.persistence = persistence
        self.angle_radians = math.radians(angle_degrees)
        self.gain = gain
        self.noise_tolerance = add_noise_tolerance
        self.mapping_range = mapping_range
        self.brightness = brightness
        self.contrast = contrast
        self.seed = seed

    def forward(self, batch_size=1, device='cpu', seed=0):
        """
        Generate a batch of crosshatch-patterned images.

        Args:
            batch_size (int, optional): Number of images to generate. Default is 1.
            device (str, optional): The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            seed (int, optional): Seed for random number generation. Default is 0.

        Returns:
            torch.Tensor: A tensor containing the generated crosshatch-patterned images in shape (batch_size, height, width, 3).
        """
        x = torch.linspace(0, 1, self.width, dtype=torch.float32, device=device)
        y = torch.linspace(0, 1, self.height, dtype=torch.float32, device=device)
        x, y = torch.meshgrid(x, y, indexing="ij")

        noises = []
        for batch_idx in range(batch_size):
            noise = torch.zeros(self.width, self.height, dtype=torch.float32, device=device)

            for octave in range(self.num_octaves):
                oct_seed = seed + octave
                frequency = self.frequency * 2 ** octave
                octave_noise = self.generate_octave(x, y, frequency, device, oct_seed)
                noise += octave_noise * self.persistence ** octave

            noise = normalize(noise, 0, 1)
            mapped_noise = self.apply_mapping(noise.permute(1, 0), device)

            # Expand the tensor to have 3 channels
            mapped_noise = mapped_noise.unsqueeze(-1).expand(-1, -1, 3)

            noises.append(mapped_noise)

        batched_noises = torch.stack(noises, dim=0)

        return batched_noises.to(device='cpu')

    def generate_octave(self, x, y, frequency, device, seed):
        """
        Generate an octave of the crosshatch pattern.

        Args:
            x (torch.Tensor): X-coordinate grid.
            y (torch.Tensor): Y-coordinate grid.
            frequency (int): Frequency of the crosshatch pattern.
            device (str): The device to use for computation ('cpu' or 'cuda').
            seed (int): Seed for random number generation.

        Returns:
            torch.Tensor: Octave of the crosshatch pattern.
        """
        torch.manual_seed(seed)

        grid_hatch_x = torch.sin(x * (frequency * self.gain) * math.pi)
        grid_hatch_y = torch.sin(y * (frequency * self.gain) * math.pi)

        noise = torch.randint(int(self.frequency + 1), (grid_hatch_x.shape[0], grid_hatch_x.shape[1]), device=grid_hatch_x.device, dtype=grid_hatch_x.dtype)
        
        grid_hatch = (normalize(grid_hatch_x) + normalize(grid_hatch_y)) + (noise * self.noise_tolerance)

        return grid_hatch

    def apply_mapping(self, noise, device):
        """
        Apply mapping to noise values for consistent look.

        Args:
            noise (torch.Tensor): Noise values.
            device (str): The device to use for computation ('cpu' or 'cuda').

        Returns:
            torch.Tensor: Noise values after mapping.
        """
        steps = min(max(self.mapping_range, 4), 256)

        step_mapping = torch.linspace(0, 1, steps, dtype=torch.float32, device=device)
        noise_scaled = noise * (steps - 1)
        noise_scaled_rounded = torch.round(noise_scaled)
        noise_scaled_rounded = torch.clamp(noise_scaled_rounded, 0, steps - 1)

        noise = step_mapping[noise_scaled_rounded.long()]

        return noise
        
# POWER-LAW NOISE
        
class PowerLawNoise(nn.Module):
    """
    Generate various types of power-law noise.
    """
    def __init__(self, device='cpu'):
        """
        Initialize the PowerLawNoise.

        Args:
            device (str, optional): The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            alpha (float, optional): The exponent of the power-law distribution. Default is 2.0.
        """
        super(PowerLawNoise, self).__init__()
        self.device = device
        
    @staticmethod
    def get_noise_types():
        """
        Return the valid noise types

        Returns:
            (list): a list of noise types to use for noise_type parameter
        """
        return ["white", "grey", "pink", "green", "blue", "random_mix", "brownian_fractal", "velvet", "violet"]

    def get_generator(self, noise_type):
        if noise_type in self.get_noise_types():
            if noise_type == "white":
                return self.white_noise
            elif noise_type == "grey":
                return self.grey_noise
            elif noise_type == "pink":
                return self.pink_noise
            elif noise_type == "green":
                return self.green_noise
            elif noise_type == "blue":
                return self.blue_noise
            elif noise_type == "velvet":
                return self.velvet_noise
            elif noise_type == "violet":
                return self.violet_noise
            elif noise_type == "random_mix":
                return self.mix_noise
            elif noise_type == "brownian_fractal":
                return self.brownian_fractal_noise
        else:
            raise ValueError(f"`noise_type` is invalid. Valid types are {', '.join(self.get_noise_types())}")

    def set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The random seed value.
        """
        if seed is not None:
            torch.manual_seed(seed)

    def white_noise(self, batch_size, width, height, scale, seed, alpha=0.0, **kwargs):
        """
        Generate white noise with a power-law distribution.

        Args:
            batch_size (int): Number of noise images to generate.
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            scale (float): Amplitude scale factor.
            seed (int): The random seed value.
            alpha (float, optional): The exponent of the power-law distribution. Default is 0.0.

        Returns:
            torch.Tensor: White power-law noise image.
        """
        self.set_seed(seed)
        scale = scale
        noise_real = torch.randn((batch_size, 1, height, width), device=self.device)
        noise_power_law = torch.sign(noise_real) * torch.abs(noise_real) ** alpha
        noise_power_law *= scale
        return noise_power_law.to(self.device)

    def grey_noise(self, batch_size, width, height, scale, seed, alpha=1.0, **kwargs):
        """
        Generate grey noise with a flat power spectrum and modulation.

        Args:
            batch_size (int): Number of noise images to generate.
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            scale (float): Amplitude scale factor.
            seed (int): The random seed value.
            alpha (float, optional): The exponent of the power-law distribution. Default is 1.0.

        Returns:
            torch.Tensor: Grey noise image with modulation and a flat power spectrum.
        """
        self.set_seed(seed)
        scale = scale
        noise_real = torch.randn((batch_size, 1, height, width), device=self.device)
        modulation = torch.abs(noise_real) ** (alpha - 1)
        noise_modulated = noise_real * modulation
        noise_modulated *= scale
        return noise_modulated.to(self.device)

    def blue_noise(self, batch_size, width, height, scale, seed, alpha=2.0, **kwargs):
        """
        Generate blue noise using the power spectrum method.

        Args:
            batch_size (int): Number of noise images to generate.
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            scale (float): Amplitude scale factor.
            seed (int): The random seed value.
            alpha (float, optional): The exponent of the power-law distribution. Default is 2.0.

        Returns:
            torch.Tensor: Blue noise image with shape [batch_size, 1, height, width].
        """
        self.set_seed(seed)

        noise = torch.randn(batch_size, 1, height, width, device=self.device)
        
        freq_x = fft.fftfreq(width, 1.0)
        freq_y = fft.fftfreq(height, 1.0)
        Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        
        power = (Fx**2 + Fy**2)**(alpha / 2.0)
        power[0, 0] = 1.0
        power = power.unsqueeze(0).expand(batch_size, 1, width, height).permute(0, 1, 3, 2).to(device=self.device)
        
        noise_fft = fft.fftn(noise)
        power = power.to(noise_fft)
        noise_fft = noise_fft / torch.sqrt(power)
        
        noise_real = fft.ifftn(noise_fft).real
        noise_real = noise_real - noise_real.min()
        noise_real = noise_real / noise_real.max()
        noise_real = noise_real * scale
        
        return noise_real.to(self.device)

    def green_noise(self, batch_size, width, height, scale, seed, alpha=1.5, **kwargs):
        """
        Generate green noise using the power spectrum method.

        Args:
            batch_size (int): Number of noise images to generate.
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            scale (float): Amplitude scale factor.
            seed (int): The random seed value.
            alpha (float, optional): The exponent of the power-law distribution. Default is 1.5.

        Returns:
            torch.Tensor: Green noise image with shape [batch_size, 1, height, width].
        """
        self.set_seed(seed)

        noise = torch.randn(batch_size, 1, height, width, device=self.device)
        
        freq_x = fft.fftfreq(width, 1.0)
        freq_y = fft.fftfreq(height, 1.0)
        Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        
        power = (Fx**2 + Fy**2)**(alpha / 2.0)
        power[0, 0] = 1.0
        power = power.unsqueeze(0).expand(batch_size, 1, width, height).permute(0, 1, 3, 2).to(device=self.device)
        
        noise_fft = fft.fftn(noise)
        power = power.to(noise_fft)
        noise_fft = noise_fft / torch.sqrt(power)
        
        noise_real = fft.ifftn(noise_fft).real
        noise_real = noise_real - noise_real.min()
        noise_real = noise_real / noise_real.max()
        noise_real = noise_real * scale
        
        return noise_real.to(self.device)
        
    def pink_noise(self, batch_size, width, height, scale, seed, alpha=1.0, **kwargs):
        """
        Generate pink noise using the power spectrum method.

        Args:
            batch_size (int): Number of noise images to generate.
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            scale (float): Amplitude scale factor.
            seed (int): The random seed value.
            alpha (float, optional): The exponent of the power-law distribution. Default is 1.0.

        Returns:
            torch.Tensor: Pink noise image with shape [batch_size, 1, height, width].
        """
        self.set_seed(seed)

        noise = torch.randn(batch_size, 1, height, width, device=self.device)
        
        freq_x = fft.fftfreq(width, 1.0)
        freq_y = fft.fftfreq(height, 1.0)
        Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        
        power = (Fx**2 + Fy**2)**(alpha / 2.0)
        power[0, 0] = 1.0
        power = power.unsqueeze(0).expand(batch_size, 1, width, height).permute(0, 1, 3, 2).to(device=self.device)

        noise_fft = fft.fftn(noise)
        noise_fft = noise_fft / torch.sqrt(power.to(noise_fft.dtype))

        noise_real = fft.ifftn(noise_fft).real
        noise_real = noise_real - noise_real.min()
        noise_real = noise_real / noise_real.max()
        noise_real = noise_real * scale
        
        return noise_real.to(self.device)
    
    def velvet_noise(self, batch_size, width, height, alpha=1.0, device='cpu', **kwargs):
        """
        Generate true Velvet noise with specified width and height using PyTorch.

        Args:
            batch_size (int): Number of noise images to generate.
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            alpha (float, optional): The exponent of the power-law distribution. Default is 1.0.
            device (str): Device to run on ('cpu' or 'cuda').

        Returns:
            torch.Tensor: Velvet noise image.
        """
        white_noise = torch.randn((batch_size, 1, height, width), device=device)
        velvet_noise = torch.sign(white_noise) * torch.abs(white_noise) ** (1 / alpha)
        velvet_noise /= torch.max(torch.abs(velvet_noise))
        
        return velvet_noise

    def violet_noise(self, batch_size, width, height, alpha=1.0, device='cpu', **kwargs):
        """
        Generate true Violet noise with specified width and height using PyTorch.

        Args:
            batch_size (int): Number of noise images to generate.
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            alpha (float, optional): The exponent of the power-law distribution. Default is 1.0.
            device (str): Device to run on ('cpu' or 'cuda').

        Returns:
            torch.Tensor: Violet noise image.
        """
        white_noise = torch.randn((batch_size, 1, height, width), device=device)
        violet_noise = torch.sign(white_noise) * torch.abs(white_noise) ** (alpha / 2.0)
        violet_noise /= torch.max(torch.abs(violet_noise))
        
        return violet_noise

    def brownian_fractal_noise(self, batch_size, width, height, scale, seed, alpha=1.0, modulator=1.0, **kwargs):
        """
        Generate Brownian fractal noise using the power spectrum method.

        Args:
            batch_size (int): Number of noise images to generate.
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            scale (float): Noise pattern size control (higher values result in smaller patterns).
            seed (int): The random seed value.
            alpha (float, optional): The exponent of the power-law distribution. Default is 1.0.
            modulator (float, optional): Modulate the number of iterations for brownian tree growth. Default is 10000.

        Returns:
            torch.Tensor: Brownian Tree noise image with shape (batch_size, 1, height, width).
        """
        def add_particles_to_grid(grid, particle_x, particle_y):
            for x, y in zip(particle_x, particle_y):
                grid[y, x] = 1

        def move_particles(particle_x, particle_y):
            dx = torch.randint(-1, 2, (batch_size, n_particles), device=self.device)
            dy = torch.randint(-1, 2, (batch_size, n_particles), device=self.device)
            particle_x = torch.clamp(particle_x + dx, 0, width - 1)
            particle_y = torch.clamp(particle_y + dy, 0, height - 1)
            return particle_x, particle_y

        self.set_seed(seed)
        n_iterations = int(5000 * modulator)
        fy = fft.fftfreq(height).unsqueeze(1) ** 2
        fx = fft.fftfreq(width) ** 2
        f = fy + fx
        power = torch.sqrt(f) ** alpha
        power[0, 0] = 1.0

        grid = torch.zeros(height, width, dtype=torch.uint8, device=self.device)

        n_particles = n_iterations // 10 
        particle_x = torch.randint(0, int(width), (batch_size, n_particles), device=self.device)
        particle_y = torch.randint(0, int(height), (batch_size, n_particles), device=self.device)

        neighborhood = torch.tensor([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=torch.uint8, device=self.device)

        for _ in range(n_iterations):
            add_particles_to_grid(grid, particle_x, particle_y)
            particle_x, particle_y = move_particles(particle_x, particle_y)

        brownian_tree = grid.clone().detach().float().to(self.device)
        brownian_tree = brownian_tree / brownian_tree.max()
        brownian_tree = F.interpolate(brownian_tree.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False)
        brownian_tree = brownian_tree.squeeze(0).squeeze(0)

        fy = fft.fftfreq(height).unsqueeze(1) ** 2
        fx = fft.fftfreq(width) ** 2
        f = fy + fx
        power = torch.sqrt(f) ** alpha
        power[0, 0] = 1.0

        noise_real = brownian_tree * scale

        amplitude = 1.0 / (scale ** (alpha / 2.0))
        noise_real *= amplitude

        noise_fft = fft.fftn(noise_real.to(self.device))
        noise_fft = noise_fft / power.to(self.device)
        noise_real = fft.ifftn(noise_fft).real
        noise_real *= scale

        return noise_real.unsqueeze(0).unsqueeze(0)

    def noise_masks(self, batch_size, width, height, scale, seed, num_masks=3, alpha=2.0):
        """
        Generate a fixed number of random masks.

        Args:
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            scale (float): Amplitude scale factor.
            seed (int): The random seed value for mask generation.
            num_masks (int, optional): Number of masks to generate. Default is 3.
            batch_size (int, optional): Number of masks per batch. Default is 1.
            alpha (float, optional): The exponent of the power-law distribution. Default is 2.0.

        Returns:
            List[torch.Tensor]: List of blue noise masks.
        """
        masks = []
        for i in range(num_masks):
            mask_seed = seed + (i * 100)
            random.seed(mask_seed)
            noise_type = random.choice(self.get_noise_types())
            mask = self.get_generator(noise_type)(batch_size, width, height, scale=scale, seed=mask_seed, alpha=alpha)
            masks.append(mask)
        return masks

    def mix_noise(self, batch_size, width, height, scale, seed, alpha=2.0, **kwargs):
        """
        Mix white, grey, and pink noise with blue noise masks.

        Args:
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            scale (float): Amplitude scale factor.
            seed (int): The random seed value.
            batch_size (int): Number of images to generate.
            alpha (float, optional): The exponent of the power-law distribution. Default is 2.0.

        Returns:
            torch.Tensor: Mixed noise image.
        """
        noise_types = [random.choice(self.get_noise_types()) for _ in range(3)]
        scales = [scale] * 3
        noise_alpha = random.uniform(0.5, 2.0)
        print("Random Alpha:", noise_alpha)

        mixed_noise = torch.zeros(batch_size, 1, height, width, device=self.device)
        
        for noise_type in noise_types:
            noise_seed = seed + random.randint(0, 1000)
            noise = self.get_generator(noise_type)(batch_size, width, height, seed=noise_seed, scale=scale, alpha=noise_alpha).to(self.device)
            mixed_noise += noise

        return mixed_noise

    def forward(self, batch_size, width, height, alpha=2.0, scale=1.0, modulator=1.0, noise_type="white", seed=None):
        """
        Generate a noise image with options for type, frequency, and seed.

        Args:
            batch_size (int): Number of noise images to generate.
            width (int): Width of the noise image.
            height (int): Height of the noise image.
            scale (float): Amplitude scale factor for the noise. Default is 1.0.
            modulator (float, optional): Modulate the noise. Currently only available for brownian_fractal. Default is 10000.
            noise_type (str): Type of noise to generate ('white', 'grey', 'pink', 'green', 'blue', 'mix').
            seed (int, optional): The random seed value. Default is None.

        Returns:
            torch.Tensor: Generated noise image.
        """

        if noise_type not in self.get_noise_types():
            raise ValueError(f"`noise_type` is invalid. Valid types are {', '.join(self.get_noise_types())}")

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        channels = []
        for i in range(3):
            gen_seed = seed + i
            random.seed(gen_seed)
            noise = normalize(self.get_generator(noise_type)(batch_size, width, height, scale=scale, seed=gen_seed, alpha=alpha, modulator=modulator))
            channels.append(noise)

        noise_image = torch.cat((channels[0], channels[1], channels[2]), dim=1)
        noise_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())
        noise_image = noise_image.permute(0, 2, 3, 1).float()

        return noise_image.to(device="cpu")