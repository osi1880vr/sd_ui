'''
ldm.invoke.generator.inpaint descends from ldm.invoke.generator
'''

import torch
import numpy as  np
from einops import rearrange, repeat
from torch import autocast

from ldm.devices.devices             import choose_autocast_device
from ldm.dream.generator.img2img   import Img2Img
from ldm.models.diffusion.ddim     import DDIMSampler
from ldm.models.diffusion.ksampler import KSampler

class Inpaint(Img2Img):
    def __init__(self, model, precision, **kwargs):
        self.init_latent = None
        self.precision = precision
        super().__init__(model)

    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,mask_image,strength,
                       step_callback=None,inpaint_replace=False,**kwargs):
        print(self.precision)
        """
        Returns a function returning an image derived from the prompt and
        the initial image + mask.  Return value depends on the seed at
        the time you call it.  kwargs are 'init_latent' and 'strength'
        """
        # klms samplers not supported yet, so ignore previous sampler
        if isinstance(sampler,KSampler):
            print(
                f">> Using recommended DDIM sampler for inpainting."
            )
            sampler = DDIMSampler(self.model, device=self.model.device)

        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

        mask_image = mask_image[0][0].unsqueeze(0).repeat(4,1,1).unsqueeze(0)
        mask_image = repeat(mask_image, '1 ... -> b ...', b=1)
        print(self.precision)
        scope = self.precision
        print(self.model.device.type)

        #with scope(self.model.device.type):
        with autocast("cuda"):
            self.init_latent = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(init_image)
            ) # move to latent space

        t_enc   = int(strength * steps)
        uc, c   = conditioning

        print(f">> target t_enc is {t_enc} steps")

        @torch.no_grad()
        def make_image(x_T):
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                self.init_latent,
                torch.tensor([t_enc]).to(self.model.device),
                noise=x_T
            )

            # to replace masked area with latent noise, weighted by inpaint_replace strength
            if inpaint_replace > 0.0:
                print(f'>> inpaint will replace what was under the mask with a strength of {inpaint_replace}')
                l_noise = self.get_noise(kwargs['width'],kwargs['height'])
                inverted_mask = 1.0-mask_image  # there will be 1s where the mask is
                masked_region = (1.0-inpaint_replace) * inverted_mask * z_enc + inpaint_replace * inverted_mask * l_noise
                z_enc   = z_enc * mask_image + masked_region

            # decode it
            samples = sampler.decode(
                z_enc,
                c,
                t_enc,
                img_callback                 = step_callback,
                unconditional_guidance_scale = cfg_scale,
                unconditional_conditioning = uc,
                mask                       = mask_image,
                init_latent                = self.init_latent
            )

            return self.sample_to_image(samples)

        return make_image


