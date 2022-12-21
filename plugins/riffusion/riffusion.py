import io
import os

import PIL
from PIL import Image
from PySide6 import QtUiTools
from PySide6.QtCore import QObject, QFile

from plugins.riffusion import audio


class Riffusion(QObject):
    def __init__(self, *args, **kwargs):
        self.params = None
        loader = QtUiTools.QUiLoader()
        file = QFile("plugins/riffusion/riffusion.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()


class aiPixelsPlugin():
    def __init__(self, parent):
        self.parent = parent
        self.riffusion = Riffusion()
        self.base_dir = 'plugins/riffusion'
        self.seed_image_dir = os.path.join(self.base_dir,'seed_images')
        self.mask_image_dir = os.path.join(self.base_dir,'mask_images')
        self.prepare_seed_lists()

    def initme(self):
        print("Initializing Riffusion")
        os.makedirs('data/output/riffusion', exist_ok=True)
        self.riffusion.w.show()

    def prepare_seed_lists(self):
        self.seed_images = self.get_file_names(self.seed_image_dir)
        print(self.seed_images)

        self.mask_images = self.get_file_names(self.mask_image_dir)
        print(self.mask_images)
        self.riffusion.w.selected_seed_image.addItems(self.seed_images)
        self.riffusion.w.selected_mask_image.addItems(self.mask_images)


    def image_bytes_from_image(self,image: PIL.Image, mode: str = "PNG") -> io.BytesIO:
        """
        Convert a PIL image into bytes of the given image format.
        """
        image_bytes = io.BytesIO()
        image.save(image_bytes, mode)
        image_bytes.seek(0)
        return image_bytes

    def create_audio(self,image):

        image = Image.open(image)
        # Reconstruct audio from the image
        wav_bytes, duration_s = audio.wav_bytes_from_spectrogram_image(image)
        mp3_bytes = audio.mp3_bytes_from_wav_bytes(wav_bytes)
        newFile = open("filename.mp3", "ab")
        newFile.write(mp3_bytes.read())
        newFile.close()

    def get_file_names(self,path):
        from os import walk
        return[f for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.png']


    def get_files(self,path):
        from os import walk
        return[os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.png']


    def run_riffusion(self):
        files = self.get_files('images')
        for image in files:
            self.create_audio(image)


    def run_it(self):
        self.deforum_six.run_deforum_six(W=int(self.params.W),
                                         H=int(self.params.H),
                                         seed=int(self.params.seed) if self.params.seed != '' else self.params.seed,
                                         sampler=str(self.params.sampler),
                                         steps=int(self.params.steps),
                                         scale=float(self.params.scale),
                                         ddim_eta=float(self.params.ddim_eta),
                                         save_settings=bool(self.params.save_settings),
                                         save_samples=bool(self.params.save_samples),
                                         show_sample_per_step=bool(self.params.show_sample_per_step),
                                         n_batch=int(self.params.n_batch),
                                         seed_behavior=self.params.seed_behavior,
                                         make_grid=self.params.make_grid,
                                         grid_rows=self.params.grid_rows,
                                         use_init=self.params.use_init,
                                         init_image=self.params.init_image,
                                         strength=float(self.params.strength),
                                         strength_0_no_init=self.params.strength_0_no_init,
                                         device=self.params.device,
                                         animation_mode=self.params.animation_mode,
                                         prompts=self.params.prompts,
                                         max_frames=self.params.max_frames,
                                         outdir=self.params.outdir,
                                         n_samples=self.params.n_samples,
                                         mean_scale=self.params.mean_scale,
                                         var_scale=self.params.var_scale,
                                         exposure_scale=self.params.exposure_scale,
                                         exposure_target=self.params.exposure_target,
                                         colormatch_scale=float(self.params.colormatch_scale),
                                         colormatch_image=self.params.colormatch_image,
                                         colormatch_n_colors=self.params.colormatch_n_colors,
                                         ignore_sat_weight=self.params.ignore_sat_weight,
                                         clip_name=self.params.clip_name,
                                         # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
                                         clip_scale=self.params.clip_scale,
                                         aesthetics_scale=self.params.aesthetics_scale,
                                         cutn=self.params.cutn,
                                         cut_pow=self.params.cut_pow,
                                         init_mse_scale=self.params.init_mse_scale,
                                         init_mse_image=self.params.init_mse_image,
                                         blue_scale=self.params.blue_scale,
                                         gradient_wrt=self.params.gradient_wrt,  # ["x", "x0_pred"]
                                         gradient_add_to=self.params.gradient_add_to,  # ["cond", "uncond", "both"]
                                         decode_method=self.params.decode_method,  # ["autoencoder","linear"]
                                         grad_threshold_type=self.params.grad_threshold_type,
                                         # ["dynamic", "static", "mean", "schedule"]
                                         clamp_grad_threshold=self.params.clamp_grad_threshold,
                                         clamp_start=self.params.clamp_start,
                                         clamp_stop=self.params.clamp_stop,
                                         grad_inject_timing=1,
                                         # if self.parent.widgets[self.parent.current_widget].w.grad_inject_timing.text() == '' else self.parent.widgets[self.parent.current_widget].w.grad_inject_timing.text(), #it is a float an int or a list of floats
                                         cond_uncond_sync=self.params.cond_uncond_sync,
                                         step_callback=self.parent.tensor_preview_signal if self.params.show_sample_per_step is not False else None,
                                         image_callback=self.parent.image_preview_signal,
                                         negative_prompts=self.params.negative_prompts if self.params.negative_prompts is not False else None,
                                         hires=self.params.hires,
                                         prompt_weighting=self.params.prompt_weighting,
                                         normalize_prompt_weights=self.params.normalize_prompt_weights,
                                         lowmem=self.params.lowmem)
