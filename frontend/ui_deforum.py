import itertools
import os
import re
import time
import random

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PySide6.QtCore import QObject, Signal, QPoint
from PySide6.QtGui import QMouseEvent

from backend.deforum.deforum_adapter import DeforumSix
from backend.hypernetworks.modules.images import GridAnnotation
from backend.singleton import singleton
import torchvision.transforms as T
from torchvision.utils import make_grid
from einops import rearrange
from fonts.ttf import Roboto
from backend.worker import Worker
from backend.shared import model_killer

gs = singleton


class Callbacks(QObject):
    txt2img_step = Signal()
    reenable_runbutton = Signal()
    txt2img_image_cb = Signal()
    deforum_step = Signal()
    deforum_image_cb = Signal()
    compviscallback = Signal()
    add_image_to_thumbnail_signal = Signal(str)
    setStatusBar = Signal(str)
    vid2vid_one_percent = Signal(int)
    prepare_hires_batch = Signal(str)


class Deforum_UI(QObject):
    def __init__(self, parent):
        # super(QObject, self).__init__()
        self.renderedFrames = None
        self.currentFrames = None
        self.onePercent = None
        self.updateRate = None
        self.update = None
        self.progress = None
        self.deforum = None
        self.parent = parent
        #self.deforum = DeforumGenerator()
        self.deforum_six = DeforumSix(self)
        self.signals = Callbacks()
        #self.deforum_six = DeforumSix()
    def run(self):
        params = self.parent.sessionparams.update_params()
        ##print(f"updated parameters to: {params}")
        self.run_it()

    def set_aesthetics(self):
        gs.diffusion.selected_aesthetic_embedding = self.parent.widgets[self.parent.current_widget].w.selected_aesthetic_embedding.currentText()
        gs.T = self.parent.widgets[self.parent.current_widget].w.gradient_steps.value()
        gs.lr = self.parent.widgets[self.parent.current_widget].w.gradient_scale.value()
        gs.aesthetic_weight = self.parent.widgets[self.parent.current_widget].w.aesthetic_weight.value()
        gs.slerp = self.parent.widgets[self.parent.current_widget].w.slerp.isChecked()
        gs.aesthetic_imgs_text = self.parent.widgets[self.parent.current_widget].w.aesthetic_imgs_text.toPlainText()
        gs.slerp_angle = self.parent.widgets[self.parent.current_widget].w.slerp_angle.value()
        gs.aesthetic_text_negative = self.parent.widgets[self.parent.current_widget].w.aesthetic_text_negative.toPlainText()

        #not aesthetics but gs global
        gs.karras = self.parent.widgets[self.parent.current_widget].w.karras.isChecked()



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
                                         lowmem=self.params.lowmem,

                                         keyframes=self.params.keyframes,

                                         dynamic_threshold=self.params.dynamic_threshold,
                                         static_threshold=self.params.static_threshold,
                                         # @markdown **Save & Display Settings**
                                         display_samples=self.params.display_samples,  # @param {type:"boolean"}
                                         save_sample_per_step=self.params.save_sample_per_step,  # @param {type:"boolean"}
                                        # normalize_prompt_weights=True,  # @param {type:"boolean"}
                                         log_weighted_subprompts=self.params.log_weighted_subprompts,  # @param {type:"boolean"}
                                         adabins=self.params.adabins,

                                         batch_name=self.params.batch_name,  # @param {type:"string"}
                                         filename_format=self.params.filename_format,
                                         # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]

                                         # Whiter areas of the mask are areas that change more
                                         use_mask=self.params.use_mask,  # @param {type:"boolean"}
                                         use_alpha_as_mask=self.params.use_alpha_as_mask,  # use the alpha channel of the init image as the mask
                                         mask_file=self.params.mask_file,  # @param {type:"string"}
                                         invert_mask=self.params.invert_mask,  # @param {type:"boolean"}
                                         # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
                                         mask_brightness_adjust=self.params.mask_brightness_adjust,  # @param {type:"number"}
                                         mask_contrast_adjust=self.params.mask_contrast_adjust,  # @param {type:"number"}
                                         # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
                                         overlay_mask=self.params.overlay_mask,  # {type:"boolean"}
                                         # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
                                         mask_overlay_blur=self.params.mask_overlay_blur,  # {type:"number"}

                                         precision=self.params.precision,

                                         # prompt="",
                                         timestring=self.params.timestring,
                                         init_latent=self.params.init_latent,
                                         init_sample=self.params.init_sample,
                                         init_c=self.params.init_c,

                                         # Anim Args

                                         # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}

                                         border=self.params.border,  # @param ['wrap', 'replicate'] {type:'string'}
                                         angle=self.params.angle,  # @param {type:"string"}
                                         zoom=self.params.zoom,  # @param {type:"string"}
                                         translation_x=self.params.translation_x,  # @param {type:"string"}
                                         translation_y=self.params.translation_y,  # @param {type:"string"}
                                         translation_z=self.params.translation_z,  # @param {type:"string"}
                                         rotation_3d_x=self.params.rotation_3d_x,  # @param {type:"string"}
                                         rotation_3d_y=self.params.rotation_3d_y,  # @param {type:"string"}
                                         rotation_3d_z=self.params.rotation_3d_z,  # @param {type:"string"}
                                         flip_2d_perspective=self.params.flip_2d_perspective,  # @param {type:"boolean"}
                                         perspective_flip_theta=self.params.perspective_flip_theta,  # @param {type:"string"}
                                         perspective_flip_phi=self.params.perspective_flip_phi,  # @param {type:"string"}
                                         perspective_flip_gamma=self.params.perspective_flip_gamma,  # @param {type:"string"}
                                         perspective_flip_fv=self.params.perspective_flip_fv,  # @param {type:"string"}
                                         noise_schedule=self.params.noise_schedule,  # @param {type:"string"}
                                         strength_schedule=self.params.strength_schedule,  # @param {type:"string"}
                                         contrast_schedule=self.params.contrast_schedule,  # @param {type:"string"}
                                         # @markdown ####**Coherence:**
                                         color_coherence=self.params.color_coherence,
                                         diffusion_cadence=self.params.diffusion_cadence,  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

                                         # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}

                                         # @markdown ####**3D Depth Warping:**
                                         use_depth_warping=self.params.use_depth_warping,  # @param {type:"boolean"}
                                         midas_weight=self.params.midas_weight,  # @param {type:"number"}
                                         near_plane=self.params.near_plane,
                                         far_plane=self.params.far_plane,
                                         fov=self.params.fov,  # @param {type:"number"}
                                         padding_mode=self.params.padding_mode,  # @param ['border', 'reflection', 'zeros'] {type:'string'}
                                         sampling_mode=self.params.sampling_mode,  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
                                         save_depth_maps=self.params.save_depth_maps,  # @param {type:"boolean"}

                                         # @markdown ####**Video Input:**
                                         video_init_path=self.params.video_init_path,  # @param {type:"string"}
                                         extract_nth_frame=self.params.extract_nth_frame,  # @param {type:"number"}
                                         overwrite_extracted_frames=self.params.overwrite_extracted_frames,  # @param {type:"boolean"}
                                         use_mask_video=self.params.use_mask_video,  # @param {type:"boolean"}
                                         video_mask_path=self.params.video_mask_path,  # @param {type:"string"}

                                         # @markdown ####**Interpolation:**
                                         interpolate_key_frames=self.params.interpolate_key_frames,  # @param {type:"boolean"}
                                         interpolate_x_frames=self.params.interpolate_x_frames,  # @param {type:"number"}

                                         # @markdown ####**Resume Animation:**
                                         resume_from_timestring=self.params.resume_from_timestring,  # @param {type:"boolean"}
                                         resume_timestring=self.params.resume_timestring,
                                         # prev_sample=None,
                                         clear_latent=self.params.clear_latent,
                                         clear_sample=self.params.clear_sample,
                                         shouldStop=self.params.shouldStop,
                                         # keys={}
                                         cpudepth=self.params.cpudepth,
                                         # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
                                         skip_video_for_run_all=self.params.skip_video_for_run_all,
                                         prompt=self.params.prompt,
                                         #use_hypernetwork=None,
                                         apply_strength=self.params.apply_strength,
                                         apply_circular=self.params.apply_circular
                                         )

    def set_multi_dim_seed(self):
        if self.params.multi_dim_seed_mode == 'iter':
            self.params.seed += 1
        elif self.params.multi_dim_seed_mode == 'fixed':
            pass  # always keep seed the same
        else:
            self.params.seed = random.randint(0, 2**32 - 1)



    def multi_dim_loop(self):
        ints_vals=['W','H','seed','steps', 'n_batch','n_samples','mean_scale','var_scale','exposure_scale','exposure_target','colormatch_n_colors', 'ignore_sat_weight','clip_scale','aesthetics_scale','cutn', 'init_mse_scale', 'blue_scale',]
        float_vals=['scale','ddim_eta','strength','colormatch_scale']
        bool_vals=['save_settings','save_samples','make_grid','use_init','strength_0_no_init','hires','prompt_weighting','normalize_prompt_weights']

        raw_prompts = self.params.prompts.split('\n')
        regex = r"(.*?)(--.*)"
        for prompt in raw_prompts:
            matches = re.match(regex, prompt, re.MULTILINE)
            work_prompt = matches.groups()[0]
            prompt_dimensions = matches.groups()[1]
            prompt_args = prompt_dimensions.split('--')
            args_dict={}
            for arg in prompt_args:
                if arg != '':
                    arg = arg.rstrip()
                    arg_name, args_list = arg.split('=')
                    args_list=args_list.split(',')
                    args_dict[arg_name] = args_list
            print(args_dict)
            pairs = [[(k, v) for v in args_dict[k]] for k in args_dict]
            arg_combinations = list(itertools.product(*pairs))
            for set in arg_combinations:
                for arg in set:
                    name = arg[0]
                    value = arg[1]
                    if name == 'aesthetic_weight': gs.aesthetic_weight = float(value)
                    if name == 'gradient_steps': gs.T = int(value)
                    if name == 'selected_aesthetic_embedding': gs.diffusion.selected_aesthetic_embedding = str(value)
                    if name == 'slerp': gs.slerp = bool(value)
                    if name == 'aesthetic_imgs_text': gs.aesthetic_imgs_text = str(value)
                    if name == 'slerp_angle': gs.slerp_angle = float(value)
                    if name == 'aesthetic_text_negative': gs.aesthetic_text_negative = str(value)
                    if name == 'gradient_scale': gs.lr = float(value)
                    if name in ints_vals:
                        value = int(value)
                    if name in float_vals:
                        value = float(value)
                    if name in bool_vals:
                        value = bool(value)
                    self.params.__dict__[name] = value
                self.params.prompts = work_prompt
                self.set_multi_dim_seed()
                self.run_it()


    def run_deforum_six_txt2img(self, progress_callback=None, plotting=True):
        gs.stop_all = False
        id = None
        self.params = self.parent.sessionparams.update_params()
        self.parent.params = self.params

        index = 0

        if self.parent.canvas.canvas.rectlist != []:
            for i in self.parent.canvas.canvas.rectlist:
                try:
                    i.stop()
                except:
                    pass
                id = i.id
                index = self.parent.canvas.canvas.rectlist.index(i)
        else:
            index = 0
            self.parent.params.advanced = False
        self.parent.canvas.canvas.stop_main_clock()

        if id is not None:
            self.parent.canvas.canvas.render_item = id

        gs.karras = self.parent.widgets[self.parent.current_widget].w.karras.isChecked()

        model_killer(keep='sd')

        if self.params.with_inpaint == True: # todo what is this for?
            self.parent.params.advanced = True
        else:
            if self.parent.widgets[self.parent.current_widget].w.mode.currentText() == 'basic':
                self.parent.params.advanced = False
            elif self.parent.widgets[self.parent.current_widget].w.mode.currentText() == 'advanced':
                self.parent.params.advanced = True
                self.parent.render_index = index

        self.set_aesthetics()

        seed = random.randint(0, 2 ** 32 - 1)

        plotting = self.params.plotting

        if self.params.multi_dim_prompt:
            self.multi_dim_loop()

        else:

            if plotting:

                attrib2 = self.params.plotX
                attrib1 = self.params.plotY

                ploty_list_string = self.params.plotXLine
                plotx_list_string = self.params.plotYLine
                plotY = plotx_list_string.split(', ')
                plotX = ploty_list_string.split(', ')
                self.onePercent = 100 / (len(plotX) * len(plotY) * self.params.n_batch * self.params.n_samples * self.params.steps)

            else:
                plotX = [1]
                plotY = [1]
                self.onePercent = 100 / (self.params.n_batch * self.params.n_samples * self.params.steps)
            all_images = []
            for i in plotY:
                for j in plotX:
                    if plotting:
                        self.params.__dict__[attrib1] = i
                        self.params.__dict__[attrib2] = j
                        if attrib1 == 'T': gs.T = int(i)
                        if attrib1 == 'lr': gs.lr = float(i)
                        if attrib2 == 'T': gs.T = int(j)
                        if attrib2 == 'lr': gs.lr = float(j)
                    if self.params.init_image is not None:
                        if os.path.isdir(self.params.init_image) and self.params.animation_mode == 'None':
                            self.params.max_frames = 2

                    self.run_it()
                    if plotting:
                        all_images.append(T.functional.pil_to_tensor(self.parent.image))
            if plotting:
                ver_texts = []
                hor_texts = []
                for i in plotY:
                    ver_texts.append([GridAnnotation(f"{attrib1}: {i}")])
                for j in plotX:
                    hor_texts.append([GridAnnotation(f"{attrib2}: {j}")])
                ##print(hor_texts)
                grid = make_grid(all_images, nrow=len(plotX))
                grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
                filename = f"{time.strftime('%Y%m%d%H%M%S')}_{attrib1}_{attrib2}_grid_{self.params.seed}.png"
                grid_image = Image.fromarray(grid.astype(np.uint8))

                grid_image = draw_grid_annotations(grid_image, grid_image.size[0], grid_image.size[1], hor_texts, ver_texts, self.params.W,
                                                   self.params.H, self.params)
                self.parent.image = grid_image
                self.parent.image_preview_signal(grid_image)
                grid_image.save(os.path.join(self.params.outdir, filename))



    def run_deforum_outpaint(self, params=None, progress_callback=None):
        # self.deforum = DeforumGenerator()
        # self.deforum.signals = Callbacks()
        if params == None:
            params = self.parent.sessionparams.update_params()
            self.parent.params = self.parent.sessionparams.update_params()
        self.progress = 0.0
        self.parent.update = 0
        self.onePercent = 100 / params.steps
        #self.updateRate = self.parent.sizer_count.w.previewSlider.value()
        self.updateRate = 1
        self.parent.currentFrames = []
        self.parent.renderedFrames = 0
        self.parent.sample_number = 1
        if params.n_samples == 1:
            makegrid = False
        else:
            makegrid = self.parent.widgets[self.parent.current_widget].w.make_grid.isChecked()
        #sampler_name = translate_sampler(self.parent.sampler.w.sampler.currentText())
        sampler_name = "ddim"
        init_image = "outpaint.png"
        self.set_aesthetics()

        #gs.aesthetic_embedding_path = os.path.join(gs.system.aesthetic_gradients_dir, self.parent.widgets[self.parent.current_widget].w.aesthetic_embedding.currentText())
        #if params == None:
        params = self.parent.sessionparams.update_params()


        steps = int(params.steps)
        H = int(params.H)
        W = int(params.W)
        seed = int(params.seed) if params.seed != "" else random.randint(0, 44444444)
        prompt = str(params.prompts)
        print(prompt)
        strength = float(params.strength)
        mask_blur = float(params.mask_blur)
        reconstruction_blur = float(params.recons_blur)
        scale = float(params.scale)
        ddim_eta = float(params.ddim_eta)
        with_inpaint = bool(params.with_inpaint)

        #self.parent.sessionparams.params.advanced = True
        self.parent.params.advanced = True
        #print(prompt)
        self.deforum_six.outpaint_txt2img(init_image=init_image,
                                          steps=steps,
                                          H=H,
                                          W=W,
                                          seed=seed,
                                          prompt=prompt,
                                          strength=strength,
                                          mask_blur=mask_blur,
                                          recons_blur=reconstruction_blur,
                                          scale=scale,
                                          ddim_eta=ddim_eta,
                                          image_callback=self.parent.image_preview_signal,
                                          step_callback=self.parent.tensor_preview_signal,
                                          with_inpaint=with_inpaint)

        # self.run_txt2img_lm(init_img=init_image, init_mask='outpaint_mask.png')

        self.signals.reenable_runbutton.emit()

    def deforum_outpaint_thread(self):

        self.parent.sessionparams.params = self.parent.sessionparams.update_params()
        self.choice = "Outpaint"
        worker = Worker(self.run_deforum_outpaint)
        self.parent.threadpool.start(worker)
class GridAnnotation:
    def __init__(self, text='', is_active=True):
        self.text = text
        self.is_active = is_active
        self.size = None

def draw_grid_annotations(im, width, height, hor_texts, ver_texts, W, H, params):
    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def draw_texts(drawing, draw_x, draw_y, lines):
        for i, line in enumerate(lines):
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="center")

            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)

            draw_y += line.size[1] + line_spacing

    fontsize = (W + H) // 100
    line_spacing = fontsize // 2

    try:
        fnt = ImageFont.truetype(Roboto, fontsize)
    except Exception:
        fnt = ImageFont.truetype(Roboto, fontsize)

    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_left = 0 if sum([sum([len(line.text) for line in lines]) for lines in ver_texts]) == 0 else W // 4

    cols = im.width // W
    rows = im.height // H

    ##print(f"DEBUG: {cols}, {rows}, of which at least one should be more then 1...")

    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    calc_img = Image.new("RGB", (1, 1), "white")
    calc_d = ImageDraw.Draw(calc_img)

    for texts, allowed_width in zip(hor_texts + ver_texts, [W] * len(hor_texts) + [pad_left] * len(ver_texts)):
        items = [] + texts
        texts.clear()

        for line in items:
            wrapped = wrap(calc_d, line.text, fnt, allowed_width)
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]

        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=fnt)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in
                        ver_texts]

    pad_top = max(hor_text_heights) + line_spacing * 2

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)
    #p_pad = len(params["prompts"][0]) * 1.75
    #d.multiline_text(((pad_left / 2) + p_pad, pad_top / 2), params["prompts"][0], font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="left")

    for col in range(cols):
        x = pad_left + W * col + W / 2
        y = pad_top / 2 - hor_text_heights[col] / 2

        draw_texts(d, x, y, hor_texts[col])

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + H * row + H / 2 - ver_text_heights[row] / 2

        draw_texts(d, x, y, ver_texts[row])

    return result
