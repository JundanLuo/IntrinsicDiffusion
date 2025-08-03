# ------------------------------------------------------------------------
# This file is part of IntrinsicDiffusion.
# Licensed under the Apache License, Version 2.0.
# See https://github.com/JundanLuo/IntrinsicDiffusion for details.
# If you use this code, please cite the paper listed on the above website.
# ------------------------------------------------------------------------


import math

import torch

from modeling.intrinsic_control_wrapper import IntrinsicControlWrapper
from utils import image_util
from utils import constants as C


class IntrinsicDiffusionPipeline(object):
    gamma = C.GAMMA
    control_joint: IntrinsicControlWrapper or None

    def __init__(self,
                 control_joint: IntrinsicControlWrapper or None = None,
                 base_img_dim=C.BASE_IMG_DIM):
        """
        Wrapper for ControlNet models to estimate intrinsic images (albedo, shading, normal).
        """
        self.control_joint = control_joint
        self.base_img_dim = base_img_dim

    def infer_intrinsic_images(self, _input_img, is_linear_input=False,
                               infer_img_size=None,
                               output_original_size=False,
                               pred_albedo=True, pred_shading=True, pred_normal=True,
                               inference_chunk_batch_size=0):
        """
        Estimate intrinsic components (albedo, shading, normal) from input image.

        :param _input_img: Input image tensor of shape (B, C, H, W).
        :param is_linear_input: Whether the input image is in linear RGB space.
        :param infer_img_size: Target size for inference.
            "ori": original image size,
            int: largest dimension to resize to,
            tuple/list: H X W,
            None: (max_dim, max_dim) where max_dim is the largest dimension of input image.
        :param output_original_size: Whether to resize output components back to original input image size.
        :param pred_albedo: Whether to predict albedo.
        :param pred_shading: Whether to predict shading.
        :param pred_normal: Whether to predict normal.
        :param inference_chunk_batch_size: Batch size for inference. 0 means all in one batch.
        """
        assert pred_albedo or pred_shading or pred_normal, \
            "At least one of albedo, shading and normal should be predicted."

        # let input image be linear
        if not is_linear_input:
            input_img = image_util.srgb_to_rgb(_input_img, gamma=self.gamma)
        else:
            input_img = _input_img

        # determine target input image size
        if infer_img_size is not None:
            if infer_img_size == "ori":  # use original image size
                infer_img_size = max(input_img.shape[2:])
            if isinstance(infer_img_size, int):
                ratio = float(infer_img_size) / max(input_img.shape[2:])
                tg_img_size = [math.ceil(x * ratio / self.base_img_dim) * self.base_img_dim
                               for x in input_img.shape[2:]]
            elif isinstance(infer_img_size, tuple) or isinstance(infer_img_size, list):
                tg_img_size = infer_img_size
            else:
                assert False, f"infer_img_size should be int or tuple/list, but got {type(infer_img_size)}."
            assert tg_img_size[0] % self.base_img_dim == 0 and tg_img_size[1] % self.base_img_dim == 0, \
                f"infer_img_size = {infer_img_size}. Should be divisible by {self.base_img_dim}."
        else:
            max_dim = max(input_img.shape[2:])
            max_dim = self.base_img_dim * math.ceil(float(max_dim) / self.base_img_dim)
            tg_img_size = (max_dim, max_dim)

        # resize input image
        input_img = torch.nn.functional.interpolate(input_img,
                                                    size=tg_img_size,
                                                    mode="bilinear", align_corners=False,
                                                    antialias=True)

        # predict intrinsic images
        if self.control_joint is None:
            assert False, f"No control_joint model loaded."
        else:
            joint_input_img = []
            joint_text_prompt = []
            if pred_albedo:
                joint_input_img.append(input_img)
                joint_text_prompt += [C.albedo] * input_img.shape[0]
            if pred_shading:
                joint_input_img.append(input_img)
                joint_text_prompt += [C.shading] * input_img.shape[0]
            if pred_normal:
                joint_input_img.append(input_img)
                joint_text_prompt += [C.normal] * input_img.shape[0]
            joint_input_img = torch.cat(joint_input_img, dim=0)
            out = self.control_joint(joint_input_img, text_prompt=joint_text_prompt, inference_chunk_batch_size=inference_chunk_batch_size)
            out_albedo = out_shading = out_normal = None
            if pred_albedo:
                out_albedo = out[:input_img.shape[0]]
                out = out[input_img.shape[0]:]
            if pred_shading:
                out_shading = out[:input_img.shape[0]]
                out = out[input_img.shape[0]:]
            if pred_normal:
                out_normal = out[:input_img.shape[0]]
                out = out[input_img.shape[0]:]
            assert out.shape[0] == 0, f"out.shape[0] = {out.shape[0]}"
            del pred_albedo, pred_shading, pred_normal

        # Post process
        if out_albedo is not None:
            out_albedo = out_albedo.clamp(min=0.0)
            device = out_albedo.device
        if out_shading is not None:
            out_shading = out_shading.clamp(min=0.0)
            out_shading = 1.0 - out_shading
            out_shading = 1.0 / out_shading.clamp(min=1e-5) - 1.0  # convert to regular shading
            out_shading = out_shading.clamp(min=0.0)
            device = out_shading.device
        if out_normal is not None:
            # denormalize normal
            out_normal = out_normal * 2.0 - 1.0
            normal_l2 = (out_normal ** 2.0).sum(dim=1, keepdim=True).sqrt()
            out_normal = out_normal / normal_l2.clamp(min=1e-5)
            device = out_normal.device
        else:
            out_normal = torch.ones_like(input_img, device=device)
        if out_albedo is None:
            if out_shading is not None:
                out_albedo = input_img.to(out_shading.device) / out_shading.clamp(min=1e-5)
            else:
                out_albedo = torch.ones_like(input_img, device=device)
        if out_shading is None:
            if out_albedo is not None:
                out_shading = input_img.to(out_albedo.device) / out_albedo.clamp(min=1e-5)
            else:
                out_shading = torch.ones_like(input_img, device=device)

        # Reconstruct input image using predicted components
        out_reconstr_input = (out_albedo * out_shading).clamp(min=0.0)
        out_reconstr_input = out_reconstr_input * input_img.mean(dim=[1, 2, 3], keepdim=True).to(out_reconstr_input.device) \
                                / out_reconstr_input.mean(dim=[1, 2, 3], keepdim=True).clamp(min=1e-5)
        if not is_linear_input:
            out_reconstr_input = image_util.rgb_to_srgb(out_reconstr_input, gamma=self.gamma)

        # resize back to original size
        if output_original_size:
            data_tuple = (out_albedo, out_shading, out_normal, out_reconstr_input)
            data_tuple = (torch.nn.functional.interpolate(x, size=_input_img.shape[2:],
                                                          mode="bilinear", align_corners=False,
                                                          antialias=True)
                          for x in data_tuple)
            out_albedo, out_shading, out_normal, out_reconstr_input = data_tuple
        return out_albedo, out_shading, out_normal, out_reconstr_input


def load_model(model_name, device, args):
    """
    Load the model based on the model name and configuration arguments.
    """
    if model_name == "controlnet":
        # assert args.vae_scaling_factor is not None, "Must specify vae_scaling_factor for controlnet"
        pa_path = args.prompt_adapter_path if args.prompt_adapter_path is not None else "original"
        mw = IntrinsicControlWrapper(
            base_model_path=args.pretrained_model_name_or_path,
            prompt_adapter_model_or_path=pa_path,
            vae_scaling_factor=args.vae_scaling_factor,
            controlnet_path=args.controlnet_model_name_or_path,
            num_inference_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
            agg_num=args.agg_num,
            denormalize=True,  # depends on whether training with normalization
            device=device,
            dtype=torch.float32,
            enable_xformers=args.enable_xformers_memory_efficient_attention
        )
        model = IntrinsicDiffusionPipeline(control_joint=mw)
        return model
    else:
        assert False, f"Not implemented model {model_name}."
