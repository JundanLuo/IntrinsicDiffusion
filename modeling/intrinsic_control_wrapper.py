# ------------------------------------------------------------------------
# This file is part of IntrinsicDiffusion.
# Licensed under the Apache License, Version 2.0.
# See https://github.com/JundanLuo/IntrinsicDiffusion for details.
# If you use this code, please cite the paper listed on the above website.
# ------------------------------------------------------------------------


import warnings
from typing import List, Optional, Union

import torch

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
)
from transformers import AutoTokenizer

from modeling.diffusers.pipeline_controlnet import StableDiffusionControlNetPipeline
from modeling.prompt_adapter import PromptAdapter
from utils import constants as C
IntrinsicControlNet = C.ControlModel


def enable_xformers_if_available(model):
    # Wrapper for the xFormers memory-efficient attention logic
    # (originally implemented in lines 826–838 of
    # https://github.com/huggingface/diffusers/blob/v0.24.0/examples/controlnet/train_controlnet.py)

    # Enables xFormers for memory-efficient attention if compatible and available
    from diffusers.utils.import_utils import is_xformers_available
    from packaging import version

    if is_xformers_available():
        import xformers
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            warnings.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. "
                "If you observe problems during training, please update xFormers to at least 0.0.17. "
                "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        if version.parse("2.0.0") > version.parse(torch.__version__):
            model.enable_xformers_memory_efficient_attention()
            print(f"Using xFormers {xformers.__version__} for memory efficient attention, "
                  f"with PyTorch {torch.__version__}")
        else:
            print(f"Not use xFormers in PyTorch {torch.__version__}, "
                  f"skip enabling memory efficient attention.")
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")


class IntrinsicControlWrapper(object):
    """
    ControlNet wrapper utilities for Stable Diffusion pipelines.
    """
    def __init__(self,
                 base_model_path="ptx0/pseudo-journey-v2",
                 vae=None, text_encoder=None, tokenizer=None,  unet=None,
                 prompt_adapter_model_or_path: PromptAdapter or str = None,
                 vae_scaling_factor=None,
                 controlnet_path=None,
                 controlnet=None,
                 num_inference_steps=50,
                 guidance_scale=1.0,
                 agg_num=1,
                 denormalize=True,
                 device=torch.device("cpu"),
                 dtype=torch.float32,
                 seed: int or None = None,
                 enable_xformers=True):
        """
        Initializes a wrapper around the Stable Diffusion + IntrinsicControlNet pipeline.

        Args:
            base_model_path (str): Path or identifier for the base Stable Diffusion model.
            vae, text_encoder, tokenizer, unet: Optional pretrained components to override defaults.
            prompt_adapter_model_or_path: Either a `PromptAdapter` instance or path to load one.
            vae_scaling_factor (float): Optional scaling factor override for VAE.
            controlnet_path (str): Path to a pretrained ControlNet model.
            controlnet: Directly provided ControlNet instance.
            num_inference_steps (int): Number of denoising steps.
            guidance_scale (float): Guidance scale for ControlNet.
            agg_num (int): Number of samples per prompt to be averaged for the final output.
            denormalize (bool): Whether to (de)normalize the output images. Set according to training setup.
            device (torch.device): Device to run the pipeline on.
            dtype (torch.dtype): Precision for computation (supports only float32).
            seed (int): Random seed for reproducibility.
            enable_xformers (bool): Whether to enable memory-efficient attention.
        """
        # --- check inputs ---
        assert dtype == torch.float32, \
            f"Only support float32, but got {dtype}."
        assert prompt_adapter_model_or_path is not None, \
            "Prompt adapter should be specified."
        assert not (controlnet_path is not None and controlnet is not None), \
            "Only one of controlnet_path and controlnet should be specified."
        assert isinstance(agg_num, int) and agg_num >= 1, \
            f"agg_num should be a positive integer, but got {agg_num}"

        # --- controlnet ---
        self.dtype = dtype
        controlnet = self._load_controlnet(controlnet_path, controlnet, dtype)

        # --- pipeline ---
        self.pipe = self._make_pipeline(
            base_model_path,
            vae,
            text_encoder,
            tokenizer,
            unet,
            vae_scaling_factor,
            controlnet,
            dtype,
        )

        # --- prompt adapter ---
        self.prompt_adapter = self._make_prompt_adapter(
            prompt_adapter_model_or_path,
            tokenizer,
            text_encoder,
            base_model_path,
        )

        # --- scheduler ---
        # speed up diffusion process with faster scheduler and memory optimization
        # self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        if self.pipe.scheduler.config["rescale_betas_zero_snr"]:
            print(f'Use DDIM scheduler with rescale_betas_zero_snr={self.pipe.scheduler.config["rescale_betas_zero_snr"]} '
                  f'and timestep_spacing={self.pipe.scheduler.config["timestep_spacing"]}.')
        else:
            warnings.warn(f"Scheduler is not using rescale_betas_zero_snr=True, which might be wrong.")

        # --- inference setup ---
        self.device = self.num_inference_steps = self.guidance_scale = self.agg_num = self.generator = None
        self.update_inference_params(
            device=device if device is not None else controlnet.device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            agg_num=agg_num,
            seed=seed,
        )
        self.denormalize: bool = denormalize

        # --- inference setup: pipe ---
        self.pipe = self.pipe.to(self.device)
        if enable_xformers:
            enable_xformers_if_available(self.pipe)
        # memory optimization.
        # self.pipe.enable_model_cpu_offload()

        print(
            f"Initialized IntrinsicControlWrapper:\n"
            f"\tsteps {num_inference_steps}, guidance_scale {guidance_scale}, aggregation_num {agg_num}\n"
            f"\tdevice {self.device}"
        )

    def update_inference_params(self,
                                device: Optional[torch.device] = None,
                                num_inference_steps: Optional[int] = None,
                                guidance_scale: Optional[float] = None,
                                agg_num: Optional[int] = None,
                                seed: Optional[int] = None):
        if device is not None:
            self.device = device
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps
        if guidance_scale is not None:
            self.guidance_scale = guidance_scale
        if agg_num is not None:
            self.agg_num = agg_num
        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def __call__(self, input_img: torch.Tensor, text_prompt: Union[str, List[str]],
                 output_type:str ="pt",
                 inference_chunk_batch_size: int = 0,
                 show_progress_bar: bool = True):
        """
        Generates images using the ControlNet pipeline.

        Args:
            input_img (torch.Tensor): Control image tensor [B, C, H, W].
            text_prompt (str | list[str]): Prompt(s) describing the target output.
            output_type (str): Output format, either "pt" (PyTorch tensor) or "latent".
            inference_chunk_batch_size (int): Batch size for inference. 0 means all in one batch.
            show_progress_bar (bool): Whether to display a progress bar during inference.
        """
        # Expand prompts to match batch size
        text_prompt = self._expand_prompts(text_prompt, batch=input_img.shape[0])
        assert len(text_prompt) == input_img.shape[0], \
            f"Prompt length {len(text_prompt)} does not match batch size {input_img.shape[0]}"

        # Move input image (conditioning image) to device
        condition_image = input_img.to(self.device, dtype=self.dtype)
        img_height, img_width = condition_image.shape[2], condition_image.shape[3]

        # Encode text prompts
        prompt_embeds = self.prompt_adapter(text_prompt)

        # Run diffusion
        assert output_type in ["pt", "latent"], \
            f"Invalid output_type: {output_type}."
        assert isinstance(inference_chunk_batch_size, int), \
            f"inference_chunk_batch_size should be an integer, but got {type(inference_chunk_batch_size)}."
        self.pipe.set_progress_bar_config(disable=not show_progress_bar)
        if inference_chunk_batch_size > 0:  # expand batch for multiple runs to save GPU memory
            expand_condition_image = self._repeat_tensor_or_list(condition_image, self.agg_num)
            expand_prompt_embeds = self._repeat_tensor_or_list(prompt_embeds, self.agg_num)
            expand_text_prompt = self._repeat_tensor_or_list(text_prompt, self.agg_num)
            num_images_per_prompt = 1
            actual_batch_size = inference_chunk_batch_size  # process in chunks
        else:
            expand_condition_image, expand_prompt_embeds, expand_text_prompt = \
                condition_image, prompt_embeds, text_prompt
            num_images_per_prompt = self.agg_num
            actual_batch_size = expand_condition_image.shape[0]  # all in one batch
        out_image_chunks = []
        for i in range(0, expand_condition_image.shape[0], actual_batch_size):
            start_idx, end_idx = i, min(i + actual_batch_size, expand_condition_image.shape[0])
            if inference_chunk_batch_size > 0:
                print(f"Run chunk: {start_idx} - {end_idx} (total: {expand_condition_image.shape[0]})")
            out = self.pipe(
                    prompt=None,
                    image=expand_condition_image[start_idx:end_idx],
                    latents=None,
                    prompt_embeds=expand_prompt_embeds[start_idx:end_idx],
                    domain_labels=expand_text_prompt[start_idx:end_idx],
                    height=img_height, width=img_width,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=self.generator,
                    output_type=output_type,
                    do_denormalize=self.denormalize,
                ).images
            out_image_chunks.append(out)
        out_images = torch.cat(out_image_chunks, dim=0)
        del expand_condition_image, expand_prompt_embeds, expand_text_prompt, out_image_chunks

        # Output aggregation
        out_images = self._merge_outputs(out_images, batch_size=condition_image.shape[0],
                                         agg_num=self.agg_num)
        return out_images

    # ----------------- building helpers ----------------- #

    @staticmethod
    def _load_controlnet(
        path: str,
        obj: IntrinsicControlNet or None,
        dtype: torch.dtype,
    ) -> IntrinsicControlNet:
        """
        Load a ControlNet model, either from an existing object or from a path.
        """
        if obj is None:
            obj = IntrinsicControlNet.from_pretrained(path, torch_dtype=dtype)
        print(f"ControlNet's image condition type: {obj.conditioning_model_type}")
        if hasattr(obj.controlnet_cond_embedding, "swin_features_resolution"):
            print(f"\t SwinV2 default ft res.: {obj.controlnet_cond_embedding.swin_features_resolution}\n"
                  f"\t Domains: {obj.controlnet_cond_embedding.domain_transformers.keys()}\n")
        return obj

    @staticmethod
    def _make_pipeline(
        base_model_path,
        vae,
        text_encoder,
        tokenizer,
        unet,
        vae_scaling,
        controlnet,
        dtype,
    ) -> StableDiffusionControlNetPipeline:
        """
        Construct the Stable Diffusion + ControlNet pipeline.
        Allows optional replacement of components or VAE scaling factor.
        """
        Pipeline = StableDiffusionControlNetPipeline

        # Case 1: all components are manually specified
        if any([vae, text_encoder, tokenizer, unet]):
            assert all(
                [vae, text_encoder, tokenizer, unet]
            ), "All pipeline components must be specified if overriding."
            if vae and vae_scaling is not None:
                assert abs(vae.config.scaling_factor - vae_scaling) < 1e-4, "VAE scaling mismatch."
            return Pipeline.from_pretrained(
                base_model_path,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            )

        # Case 2: only VAE scaling factor is overridden
        if vae_scaling:
            vae = AutoencoderKL.from_pretrained(base_model_path, scaling_factor=vae_scaling, subfolder="vae",
                                                revision=None)
            pipe = Pipeline.from_pretrained(
                base_model_path,
                vae=vae,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            )
            print(f"Updated VAE scaling factor: {pipe.vae.config.scaling_factor}")
            return pipe

        # Case 3: default loading
        return Pipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )

    def _make_prompt_adapter(
        self,
        model_or_path,
        tokenizer,
        text_encoder,
        base_model_path,
    ) -> PromptAdapter:
        """
        Build or load a prompt adapter.

        Args:
            model_or_path: "original" to build according to the base model,
                           path to load a pretrained adapter,
                           or an already-initialized adapter.
        """
        if isinstance(model_or_path, str):
            if model_or_path == "original":
                # Build adapter according to base model
                tok = tokenizer if tokenizer is not None else \
                    AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer", revision=None, use_fast=False)
                enc = text_encoder if text_encoder is not None else \
                    self._load_text_encoder(base_model_path)
                print(f"Building 'original' prompt adapter according to {base_model_path}")
                return PromptAdapter(tok, enc, [C.albedo, C.shading, C.normal])
            else:
                # Load adapter from given path
                adapter = PromptAdapter(None, None, [C.albedo, C.shading, C.normal])
                adapter.load_pretrained(model_or_path, strict=True)
                print(f"Loaded prompt adapter from {model_or_path}")
                return adapter

        if isinstance(model_or_path, PromptAdapter):
            print(f"Using provided prompt adapter instance")
            return model_or_path

        raise TypeError(f"Invalid prompt adapter: {type(model_or_path)}")

    @staticmethod
    def _load_text_encoder(base_model_path: str):
        """Helper to construct a text encoder when needed."""
        from train_iid_jointly import import_model_class_from_model_name_or_path
        EncoderClass = import_model_class_from_model_name_or_path(base_model_path,
                                                                  revision=None)
        return EncoderClass.from_pretrained(base_model_path, subfolder="text_encoder",
                                            revision=None)

    # ----------------- utilities ----------------- #
    @staticmethod
    def _expand_prompts(prompt: Union[str, List[str]], batch: int) -> List[str]:
        """
        Expand or validate text prompt list so it matches batch size.
        """
        if isinstance(prompt, str):
            return [prompt] * batch
        if isinstance(prompt, list):
            if len(prompt) == 1:
                return prompt * batch
            assert len(prompt) == batch, f"Prompt list length {len(prompt)} does not match batch size {batch}"
            return prompt
        raise TypeError(f"Invalid prompt type: {type(prompt)}")

    @staticmethod
    def _repeat_tensor_or_list(
            x: torch.Tensor or List,
            repeat: int
    ) -> torch.Tensor or List:
        """
        Repeat each element in a tensor or list consecutively `repeat` times.
        Example:
            list [1,2,3], repeat=3 -> [1,1,1,2,2,2,3,3,3]
            tensor [[a],[b]], repeat=2 -> [[a],[a],[b],[b]]
        """
        if isinstance(x, torch.Tensor):
            return x.repeat_interleave(repeat, dim=0)

        if isinstance(x, list):
            return [elem for elem in x for _ in range(repeat)]

        raise TypeError(f"Invalid type to repeat: {type(x)}")


    @staticmethod
    def _merge_outputs(results: torch.tensor, batch_size: int, agg_num: int) -> torch.tensor:
        """
        Aggregate results.
        """
        _, channels, height, width = results.shape
        merged = results.view(batch_size, agg_num, channels, height, width).mean(dim=1)
        # if output_type == "pt":
        #     merged = merged.clamp(min=0.0)
        return merged