from typing import List, Optional, Tuple, Union
from typing import Optional
import torch
import diffusers
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, ImagePipelineOutput


# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class MyPipeline(DiffusionPipeline):
    r"""
    Pipeline for unconditional image generation using latent diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        vae ([`AutoencoderKL`]):
            Vector-quantized (VQ) model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`DDIMScheduler`]):
            [`DDIMScheduler`] is used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler,vae):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

    def latents_to_pil(self, latents):
        # bath of latents -> list of images
        latents = (1 / 0.13025) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 96,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 100,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 96):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.


        Example:

        ```py
        >>> from ... import MyPipeline

        >>> # load model and scheduler
        >>> pipe = MyPipeline.from_pretrained("directory that contains unet and scheduler as folder.")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images
        ```

        Returns:
            List of generated PIL images.
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            latent_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            latent_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(latent_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        latents = latents * self.scheduler.init_noise_sigma # 1 for DDIMScheduler
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(latents, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            latents = self.scheduler.step(
                model_output, t, latents, eta=eta, generator=generator
            ).prev_sample

        image_output = self.latents_to_pil(latents)
        return ImagePipelineOutput(images=image_output)