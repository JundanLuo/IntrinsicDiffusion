# ------------------------------------------------------------------------
# This file is part of IntrinsicDiffusion.
# Licensed under the Apache License, Version 2.0.
# See https://github.com/JundanLuo/IntrinsicDiffusion for details.
# If you use this code, please cite the paper listed on the above website.
# ------------------------------------------------------------------------


import random
import os
import argparse
from typing import List
import math
import tempfile
import zipfile
import io

import gradio as gr
import numpy as np
import torch
from torchvision.transforms import functional as F
from accelerate.utils import set_seed

from utils import constants as C
from utils import gradio_util
from utils import image_util
from modeling import intrinsicdiffusion_pipe


# global state
STATE = {
    "model": None,
    "base_model_path": None,  # str
    "controlnet_root_path": None,  # str
    "prev_input_image": None,  # torch.tensor, 3HW, float32, range [0,1]
    "prev_mask": None,  # torch.tensor or None, 3HW, float32, binary {0,1}
    "prev_raw_albedo": None,  # torch.tensor, 3HW, float32, linear space
    "prev_raw_shading": None,  # torch.tensor, 3HW, float32, linear space
    "prev_raw_normal": None,  # torch.tensor, 3HW, float32, range [-1,1]
    "prev_raw_reconstr": None,  # torch.tensor, 3HW, float32, range [0, 1+), following input space
    "prev_albedo_img": None,  # numpy uint8, HWC, range [0,255]
    "prev_shading_img": None,  # numpy uint8, HWC, range [0,255]
    "prev_normal_img": None,  # numpy uint8, HWC, range [0,255]
    "prev_reconstr_img": None,  # numpy uint8, HWC, range [0,255]
}


def randomize_seed():
    """Generate a random seed"""
    return random.randint(0, 999999)


def preprocess_input_images(input_image, mask_image, output_height, output_width):
    """Preprocess input and mask images
    :return
        input_image: torch.Tensor, 1CHW, float32, range [0,1]
        mask_image: torch.Tensor, 1CHW, float32, binary {0,1}
        with_mask: bool, whether a mask was provided
    """
    if mask_image is None:
        mask_image = np.ones_like(input_image, dtype=np.float32)
        with_mask = False
    else:
        with_mask = True
    data_tuple = (input_image, mask_image)
    data_tuple = (d.astype(np.float32) / 255.0 if d.dtype == np.uint8 else d.astype(np.float32)
                  for d in data_tuple)  # range [0,1]
    data_tuple = (d[..., np.newaxis] if d.ndim == 2 else d for d in data_tuple)  # add channel dim if grayscale
    data_tuple = (d[..., :3] if d.shape[2] >= 3 else np.repeat(d, 3, axis=2) for d in data_tuple)  # ensure RGB
    data_tuple = (image_util.numpy_to_tensor(d) for d in data_tuple)  # HWC to CHW
    data_tuple = (F.resize(d, size=[output_height, output_width],
                           interpolation=F.InterpolationMode.BILINEAR, antialias=True)
                  for d in data_tuple)  # resize
    data_tuple = (d[None] for d in data_tuple)  # 1CHW
    input_image, mask_image = data_tuple
    mask_image = (mask_image.min(dim=1, keepdim=True)[0] > 0.5).to(torch.float32).repeat(1, input_image.shape[1], 1, 1)  # binary mask
    return input_image, mask_image, with_mask


def postprocess_output_images(linear_flag):
    """
    Postprocess output images from the model for display and download
    :param linear_flag:
    :return:
    """
    # Get data from global state
    global STATE
    albedo = STATE["prev_raw_albedo"]
    shading = STATE["prev_raw_shading"]
    normal = STATE["prev_raw_normal"]
    reconstr = STATE["prev_raw_reconstr"]
    mask = STATE["prev_mask"]

    # Linear or sRGB conversion
    if albedo is not None:
        albedo *= image_util.get_scale_alpha(albedo, mask if mask is not None else torch.ones_like(albedo),
                                             0.95, 0.85)
        if not linear_flag:
            albedo = image_util.rgb_to_srgb(albedo, gamma=C.GAMMA)
    if shading is not None:
        shading *= image_util.get_scale_alpha(shading, mask if mask is not None else torch.ones_like(shading),
                                              0.95, 0.85)
        if not linear_flag:
            shading = image_util.rgb_to_srgb(shading, gamma=C.GAMMA)
    if normal is not None:
        normal = (normal + 1.0) / 2.0  # range [0,1]
    if reconstr is not None:
        reconstr = reconstr.clip(min=0.0, max=1.0)  # range [0,1]

    # Apply mask if available
    if mask is not None:
        albedo, shading, normal, reconstr = (d * mask if d is not None else None
                                             for d in (albedo, shading, normal, reconstr))

    # Convert to numpy uint8 for Gradio
    albedo, shading, normal, reconstr = (image_util.tensor_to_numpy_uint8(d) if d is not None else None
                                         for d in (albedo, shading, normal, reconstr))
    return albedo, shading, normal, reconstr


def intrinsic_decomposition(
        input_image,
        mask_image,
        base_model_path,
        controlnet_root_path,
        linear_input,
        output_width,
        output_height,
        ddim_steps,
        num_samples,
        infer_batch_size,
        seed,
        guidance_scale,
        linear_output
):
    """
    Main inference function for intrinsic image decomposition
    """

    if input_image is None:
        return None, None, None, "Please upload an input image first."

    global STATE
    info_text = ""

    # --- load the model ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        return None, None, None, "CPU inference is not recommended. Please use a machine with a CUDA-capable GPU."
    if STATE["model"] is None or \
            STATE["base_model_path"] != base_model_path or \
            STATE["controlnet_root_path"] != controlnet_root_path:
        model = intrinsicdiffusion_pipe.IntrinsicControlWrapper(
            base_model_path=base_model_path,
            prompt_adapter_model_or_path=os.path.join(controlnet_root_path, "prompt_adapter"),
            controlnet_path=os.path.join(controlnet_root_path, "controlnet"),
            num_inference_steps=ddim_steps,
            guidance_scale=guidance_scale,
            agg_num=num_samples,
            denormalize=True,  # depends on whether training with normalization
            device=device,
            dtype=torch.float32,
            seed=None,  # config seed at each run
            enable_xformers=False
        )
        model = intrinsicdiffusion_pipe.IntrinsicDiffusionPipeline(control_joint=model)
    else:
        model = STATE["model"]
        model.control_joint.update_inference_params(
            device=device,
            num_inference_steps=ddim_steps,
            guidance_scale=guidance_scale,
            agg_num=num_samples,
            seed=None
        )
    STATE["model"] = model
    STATE["base_model_path"] = base_model_path
    STATE["controlnet_root_path"] = controlnet_root_path

    # --- input data ---
    input_image, mask_image, with_mask = preprocess_input_images(input_image, mask_image, output_height, output_width)

    # --- infer ---
    if seed < 0:
        curr_seed = randomize_seed()
    else:
        curr_seed = int(seed)
    set_seed(curr_seed)
    info_text += f"Seed: {curr_seed}. \n"
    info_text += f"Inference batch size: {'all in one batch' if infer_batch_size <= 0 else infer_batch_size}.\n"

    for key in STATE.keys():
        if "prev" in key:
            STATE[key] = None  # clear previous image state
    with torch.no_grad():
        pred_r, pred_s, pred_n, reconstr_input = model.infer_intrinsic_images(
            _input_img=input_image,
            infer_img_size=(output_height, output_width),
            is_linear_input=linear_input,
            output_original_size=True,
            pred_albedo=True,
            pred_shading=True,
            pred_normal=True,
            inference_chunk_batch_size=infer_batch_size)
        torch.cuda.empty_cache()
        STATE["prev_raw_albedo"] = pred_r[0].to(torch.float32).cpu()  # predictions might be float16
        STATE["prev_raw_shading"] = pred_s[0].to(torch.float32).cpu()  # predictions might be float16
        STATE["prev_raw_normal"] = pred_n[0].to(torch.float32).cpu()  # predictions might be float16
        STATE["prev_raw_reconstr"] = reconstr_input[0].to(torch.float32).cpu()  # predictions might be float16
    STATE["prev_input_image"] = input_image[0].cpu()  # 3HW
    STATE["prev_mask"] = mask_image[0].cpu() if with_mask else None

    # -- conversion --
    albedo_output, shading_output, normal_output, reconstr_output = postprocess_output_images(linear_output)
    info_text += f"Processed with resolution: {output_width}x{output_height}.\n"
    STATE["prev_albedo_img"] = albedo_output
    STATE["prev_shading_img"] = shading_output
    STATE["prev_normal_img"] = normal_output
    STATE["prev_reconstr_img"] = reconstr_output

    return albedo_output, shading_output, normal_output, reconstr_output, info_text


def validate_controlnet_root(path, required_subfolders):
    """Validate the ControlNet root path and surface warnings to the UI.
    inputs = (path, required_subfolders)
    """
    path = (path or "").strip()
    if not path:
        gr.Warning("IntrinsicControlNet Root Path is empty.")
        return

    # 1) Exists
    if not os.path.exists(path):
        gr.Warning(f"Path does not exist: {path}")
        return

    # 2) Is a folder
    if not os.path.isdir(path):
        gr.Warning(f"Path is not a folder: {path}")
        return

    # 3) Contains required subfolders
    missing = [sf for sf in required_subfolders if not os.path.isdir(os.path.join(path, sf))]
    if missing:
        gr.Warning(
            f"Missing subfolder(s):{', '.join(missing)}\n"
            f"in the IntrinsicControlNet Root Path"
        )
        return
    # If everything is correct, we remain silent.


def create_model_config_section():
    """Create the model configuration section"""
    gr.Markdown("## 🔧 Model Configuration")

    with gr.Group():
        base_model_path = gr.Textbox(
            label="Base Model Path",
            value="ptx0/pseudo-journey-v2",
            placeholder="Enter base model path...",
            info="Hugging Face model ID or server path to the base model."
        )

        controlnet_root_path = gr.Textbox(
            label="IntrinsicControlNet Root Path",
            value="trained_models/",
            placeholder="Enter ControlNet root path...",
            info="Server folder with IntrinsicDiffusion models; must include 'prompt_adapter/' and 'controlnet/'.",
            interactive=True
        )

        required_subfolders = C.REQUIRED_subfolders_IntrinsicControlNet
        gradio_util.bind_events(controlnet_root_path,
                                events=["submit", "blur"],
                                fn=validate_controlnet_root,
                                inputs=[controlnet_root_path, gr.State(required_subfolders)],
                                outputs=None)

    with gr.Group():
        # with gr.Row():
        #     joint_inference = gr.Checkbox(
        #         label="Joint Inference",
        #         info="Uncheck for one-by-one inference (A→S→N) to save GPU memory.",
        #         value=True
        #     )
        with gr.Row():
            # Process button
            process_btn = gr.Button(
                "🚀 Generate Decomposition",
                variant="primary",
                size="lg"
            )
    return base_model_path, controlnet_root_path, process_btn


def create_input_section():
    """Create the input section"""
    gr.Markdown("## 📤 Input")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                linear_input = gr.Checkbox(
                    label="Linear Input",
                    value=False,
                    info="Leave unchecked if the input image is in sRGB space (most common, e.g., photos)",
                )

                input_image = gr.Image(
                    label="Input Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )

        with gr.Column(scale=1):
            with gr.Group():
                use_mask = gr.Checkbox(
                    label="Upload a mask",
                    value=False,
                    info="Check to upload a mask image."
                )

                mask_image = gr.Image(
                    label="Mask",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    visible=False,  # start hidden
                )

                # Show/hide (and clear) the mask uploader on demand
                def toggle_mask(show: bool):
                    if show:
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False, value=None)

                use_mask.change(fn=toggle_mask, inputs=use_mask, outputs=mask_image)

    # Preserve the original return signature
    return input_image, mask_image, linear_input


def create_inference_settings_section():
    """Create the inference settings section"""

    gr.Markdown("## ⚙️ Inference Settings")
    with gr.Group():
        # Resolution sliders
        with gr.Group():
            with gr.Row():
                init_output_resolution = gr.Checkbox(
                    label="Reset Resolution on New Input Image",
                    value=True,
                    # info="Update output resolution each time a new image is uploaded."
                )
                match_aspect_ratio = gr.Checkbox(
                    label="Match Input Aspect Ratio",
                    value=True,
                    # info="Adjust output size to keep the same aspect ratio as the input."
                )

            with gr.Row():
                MODEL_RES_BASE = C.BASE_IMG_DIM  # the model's base resolution unit (64)

                output_width = gr.Slider(
                    label="Output Width",
                    minimum=MODEL_RES_BASE,
                    maximum=2048,
                    step=MODEL_RES_BASE,
                    value=1024,
                    info=f"Auto adjusted to a multiple of {MODEL_RES_BASE}."
                )
                output_height = gr.Slider(
                    label="Output Height",
                    minimum=MODEL_RES_BASE,
                    maximum=2048,
                    step=MODEL_RES_BASE,
                    value=768,
                    info=f"Auto adjusted to a multiple of {MODEL_RES_BASE}."
                )

        # Number of samples and inference batch size
        with gr.Row():
            num_samples = gr.Slider(
                label="Samples to Average",
                minimum=1,
                maximum=10,
                step=1,
                value=4,
                info="Averages multiple samples to produce the final intrinsic images.",
            )

            infer_batch_size = gr.Number(
                label="Inference Batch Size",
                value=0,
                precision=0,
                info="0 (default): all sampled images in one batch. "
                     "Set to smaller value (e.g., 1) to save GPU memory.",
            )

        # DDIM steps and seed
        with gr.Row():
            ddim_steps = gr.Slider(
                label="DDIM Steps",
                minimum=1,
                maximum=1000,
                step=1,
                value=20,
                info="range: 1-1000, typically 20-50."
            )
            # guidance_scale = gr.Slider(
            #     label="Guidance Scale",
            #     minimum=0.1,
            #     maximum=1.0,
            #     step=0.01,
            #     value=1.0,
            #     info="1.0 (default): condition as trained."
            # )
            guidance_scale = gr.State(1.0)

            with gr.Row():
                seed = gr.Number(
                    label="Seed",
                    value=999,
                    precision=0,
                    info="If negative, a random seed will be chosen."
                )
                randomize_btn = gr.Button("🎲 Randomize")
                randomize_btn.click(
                    fn=randomize_seed,
                    outputs=[seed]
                )

    return (init_output_resolution, match_aspect_ratio, output_width, output_height, ddim_steps, num_samples,
            seed, randomize_btn, guidance_scale, infer_batch_size)


def create_output_section():
    """Create the output section"""
    gr.Markdown("## 📊 Output")

    with gr.Group():
        info_display = gr.Textbox(
            label="Processing Info",
            interactive=False
        )

        linear_output = gr.Checkbox(
            label="Linear Output",
            value=True,
            info="Applies only to albedo and shading maps."
        )

    with gr.Row():
        albedo_output = gr.Image(
            label="Albedo",
            type="numpy",
            show_download_button=True
        )

        shading_output = gr.Image(
            label="Shading",
            type="numpy",
            show_download_button=True
        )

    with gr.Row():
        normal_output = gr.Image(
            label="Surface Normal",
            type="numpy",
            show_download_button=True
        )

        reconstr_output = gr.Image(
            label="Reconstructed Input",
            type="numpy",
            show_download_button=True,
        )

    gr.Markdown(
            "### Export Intrinsic Images:\n"
            "- .png files: intrinsic images as shown above.\n"
            "- .npy files: raw intrinsic image data (linear space).\n"
        )

    with gr.Row():
        # with gr.Column(scale=0):
        #     download_filename = gr.Textbox(
        #         label="Output File (.zip)",
        #         value="out.zip",
        #         placeholder="Enter output filename...",
        #     )
        with gr.Column(scale=0):
            save_btn = gr.Button(
                "💾 Export",
                variant="primary",
                size="lg")
            save_btn.click(
                fn=export_zip_data,
                inputs=[gr.State("out.zip")],
                outputs=[gr.File()],
            )

    # Link checkbox change event to update output images
    linear_output.change(
        fn=postprocess_output_images,
        inputs=linear_output,
        outputs=[albedo_output, shading_output, normal_output, reconstr_output]
    )

    return linear_output, info_display, albedo_output, shading_output, normal_output, reconstr_output, save_btn


def create_tips_section():
    """Create the tips section"""
    gr.Markdown("## 💡 Tips")
    gr.Markdown(
        """
        - **Albedo**: The intrinsic color/texture of surfaces without lighting effects.
        - **Shading**: How light falls on the surfaces (shadows and illumination).
        - **Surface Normal**: The geometric orientation of surfaces.
        - To save GPU memory:
            - Use lower output resolutions.
            - Reduce the number of samples to average.
            - Set a small batch size for inference (e.g., 1 or 2).
        - For better results, set the output resolution to at least 512 pixels on each side.
        - Ensure both input and output use the correct linear/sRGB settings.  
        """
    )


def export_zip_data(filename):
    """
    Create a downloadable ZIP file from STATE.

    Args:
        filename (str): The desired filename for the ZIP file
    Returns:
        str: Path to the created ZIP file for download
    """
    # Ensure filename has .zip extension
    if not filename.endswith('.zip'):
        filename += '.zip'

    # Create a temporary file for the ZIP
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, filename)

    # Create zip_data dict: .png files and .npy files
    global STATE
    input_npy, mask_npy = STATE["prev_input_image"], STATE["prev_mask"]  # tensor or None
    albedo_npy, shading_npy, normal_npy, reconstr_npy = (STATE["prev_raw_albedo"],
                                                         STATE["prev_raw_shading"],
                                                         STATE["prev_raw_normal"],
                                                         STATE["prev_raw_reconstr"])  # tensor or None
    input_npy, mask_npy, albedo_npy, shading_npy, normal_npy, reconstr_npy = \
        (image_util.tensor_to_numpy(d) if d is not None else None
         for d in (input_npy, mask_npy, albedo_npy, shading_npy, normal_npy, reconstr_npy))  # tensor -> numpy
    if any(d is None for d in [input_npy, albedo_npy, shading_npy, normal_npy, reconstr_npy]):
        gr.Warning("No data available or incomplete data for download. Please run inference first.")
        return None
    albedo_img, shading_img, normal_img, reconstr_img = \
        STATE["prev_albedo_img"], STATE["prev_shading_img"], STATE["prev_normal_img"], STATE["prev_reconstr_img"]  # numpy uint8 or None
    if any(d is None for d in [albedo_img, shading_img, normal_img, reconstr_img]):
        gr.Warning("No processed output images for download.")
        return None
    zip_data = {
        # --- PNG images (np.uint8) ---
        "input.png": (input_npy * 255.0).clip(0, 255).astype(np.uint8),  # convert back to uint8
        "albedo.png": albedo_img,
        "shading.png": shading_img,
        "normal.png": normal_img,
        "reconstructed_input.png": reconstr_img,
        # --- raw .npy files (np.float32) ---
        # "input.npy": input_npy.astype(np.float32),
        "albedo.npy": albedo_npy.astype(np.float32),
        "shading.npy": shading_npy.astype(np.float32),
        "normal.npy": normal_npy.astype(np.float32),
        "reconstructed_input.npy": reconstr_npy.astype(np.float32),
    }
    if mask_npy is not None:
        zip_data["mask.png"] = (mask_npy * 255.0).clip(0, 255).astype(np.uint8)  # convert back to uint8
        # zip_data["mask.npy"] = mask_npy.astype(np.float32)
    for k, item in zip_data.items():
        if ".png" in k:  # convert np.uint8 to PIL image
            zip_data[k] = image_util.array_to_pil(item, img_bit=8)

    # Create the ZIP file
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_name, data in zip_data.items():
                data_buffer = io.BytesIO()  # in-memory buffer
                if file_name.endswith('.png'):  # save PIL image to buffer
                    data.save(data_buffer, format='PNG')
                elif file_name.endswith('.npy'):  # save numpy array to buffer
                    np.save(data_buffer, data)
                else:
                    print(f"Warning: Unsupported file type for {file_name}")
                data_buffer.seek(0)  # Reset buffer pointer
                zipf.writestr(file_name, data_buffer.getvalue())  # Write buffer content to zip
        if not os.path.exists(zip_path):
            gr.Warning(f"Failed to create zip file at {zip_path}")
            return None
        else:
            return zip_path
    except Exception as e:
        gr.Warning(f"Error creating ZIP file: {e}")
        return None


def update_output_resolution(input_image, is_new_input,
                             init_output_resolution, match_aspect_ratio,
                             output_width, output_height, change_width):
    # Input image size
    if input_image is not None:
        input_height, input_width = input_image.shape[:2]
    else:
        input_width, input_height = 1024, 768

    # Handle invalid output resolution
    if not isinstance(output_width, int) or not isinstance(output_height, int):
        output_width, output_height = input_width, input_height

    if is_new_input and init_output_resolution:
        # Listen to input image change
        if input_image is not None:
            output_width, output_height = input_width, input_height
    else:
        # Listen to match_aspect_ratio change or output_width/output_resolution change
        if match_aspect_ratio:
            if input_image is not None:
                ratio = input_width / input_height
                if change_width:  # use output_width as reference
                    output_width = gradio_util.snap_to_base_value(output_width, C.BASE_IMG_DIM, C.BASE_IMG_DIM)
                    output_height = float(output_width) / ratio
                else:  # use output_height as reference
                    output_height = gradio_util.snap_to_base_value(output_height, C.BASE_IMG_DIM, C.BASE_IMG_DIM)
                    output_width = float(output_height) * ratio
            else:
                # No input image, cannot match aspect ratio
                pass
        else:
            # No need to match aspect ratio
            pass
    return gradio_util.snap_to_base_value(output_width, C.BASE_IMG_DIM, C.BASE_IMG_DIM), \
        gradio_util.snap_to_base_value(output_height, C.BASE_IMG_DIM, C.BASE_IMG_DIM)


def setup_event_handlers(input_image, init_output_resolution, match_aspect_ratio, output_width, output_height,
                         randomize_btn, seed,
                         process_btn, base_model_path, controlnet_root_path, mask_image,
                         linear_input, ddim_steps, num_samples, infer_batch_size,
                         guidance_scale, linear_output, albedo_output, shading_output,
                         normal_output, reconstr_output, save_btn, info_display):
    """Set up all event handlers"""
    # Update resolution when image is uploaded
    input_image.change(
        fn=update_output_resolution,
        inputs=[input_image, gr.State(True), init_output_resolution, match_aspect_ratio,
                output_width, output_height, gr.State(True)],
        outputs=[output_width, output_height]
    )

    # Update output resolution when sliders are released or typed in
    gradio_util.bind_events(output_width,
                            events=["release"],
                            fn=update_output_resolution,
                            inputs=[input_image, gr.State(False), init_output_resolution, match_aspect_ratio,
                                    output_width, output_height, gr.State(True)],
                            outputs=[output_width, output_height])
    gradio_util.bind_events(output_height,
                            events=["release"],
                            fn=update_output_resolution,
                            inputs=[input_image, gr.State(False), init_output_resolution, match_aspect_ratio,
                                    output_width, output_height, gr.State(False)],
                            outputs=[output_width, output_height])

    # match_aspect_ratio checkbox change
    match_aspect_ratio.change(
        fn=update_output_resolution,
        inputs=[input_image, gr.State(False), init_output_resolution, match_aspect_ratio,
                output_width, output_height, gr.State(True)],
        outputs=[output_width, output_height]
    )

    # Process button
    process_btn.click(
        fn=intrinsic_decomposition,
        inputs=[
            input_image,
            mask_image,
            base_model_path,
            controlnet_root_path,
            linear_input,
            output_width,
            output_height,
            ddim_steps,
            num_samples,
            infer_batch_size,
            seed,
            guidance_scale,
            linear_output
        ],
        outputs=[
            albedo_output,
            shading_output,
            normal_output,
            reconstr_output,
            info_display
        ]
    )


def create_gradio_interface():
    """Create the complete Gradio interface"""

    with gr.Blocks(title="IntrinsicDiffusion", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 IntrinsicDiffusion Demo

            Upload an image (optionally with a mask) to decompose it into **albedo**, **shading**, and **surface normal** maps.
            """
        )
        gr.Markdown("⚠️ **Note:** This script does not support multiple users simultaneously. "
                    "All sessions share the same global state.")

        input_image, mask_image, linear_input = create_input_section()

        with gr.Row():
            # Left column - Inference Settings
            with gr.Column(scale=3):
                (init_output_resolution, match_aspect_ratio, output_width, output_height, ddim_steps, num_samples,
                 seed, randomize_btn, guidance_scale, infer_batch_size) = create_inference_settings_section()

            # Right column - Model Configuration
            with gr.Column(scale=2):
                base_model_path, controlnet_root_path, process_btn = create_model_config_section()

        # Output section
        (linear_output, info_display, albedo_output,
         shading_output, normal_output, reconstr_output, save_btn) = create_output_section()

        # Tips section
        create_tips_section()

        # Set up all event handlers
        setup_event_handlers(
            input_image, init_output_resolution, match_aspect_ratio, output_width, output_height, randomize_btn, seed,
            process_btn, base_model_path, controlnet_root_path, mask_image,
            linear_input, ddim_steps, num_samples, infer_batch_size,
            guidance_scale, linear_output, albedo_output, shading_output,
            normal_output, reconstr_output, save_btn, info_display
        )

    return demo


def parse_args():
    """Parse command line arguments for app configuration"""
    parser = argparse.ArgumentParser(
        description="Launch IntrinsicDiffusion Gradio App",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-n", "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server name/IP to bind to"
    )

    parser.add_argument(
        "-p", "--server-port",
        type=int,
        default=7860,
        help="Port number to run the server on"
    )

    parser.add_argument(
        "-s", "--share",
        action="store_true",
        help="Create a publicly shareable link"
    )

    parser.add_argument(
        "-we", "--no-show-error",
        action="store_true",
        help="Disable error display in the interface"
    )

    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "-a", "--auth",
        type=str,
        nargs=2,
        metavar=('USERNAME', 'PASSWORD'),
        help="Enable authentication with username and password"
    )

    parser.add_argument(
        "-k", "--ssl-keyfile",
        type=str,
        help="Path to SSL key file for HTTPS"
    )

    parser.add_argument(
        "-c", "--ssl-certfile",
        type=str,
        help="Path to SSL certificate file for HTTPS"
    )

    return parser.parse_args()


def launch_app(server_name="0.0.0.0", server_port=7860, show_error=True,
               share=False, debug=False, auth=None, ssl_keyfile=None, ssl_certfile=None):
    """Launch the Gradio app with specified configuration"""
    demo = create_gradio_interface()

    launch_kwargs = {
        "server_name": server_name,
        "server_port": server_port,
        "show_error": show_error,
        "share": share
    }

    # Only add debug if it's True (Gradio might not support debug=False)
    if debug:
        launch_kwargs["debug"] = debug

    # Add optional parameters only if provided
    if auth:
        launch_kwargs["auth"] = auth
    if ssl_keyfile:
        launch_kwargs["ssl_keyfile"] = ssl_keyfile
    if ssl_certfile:
        launch_kwargs["ssl_certfile"] = ssl_certfile

    demo.launch(**launch_kwargs)


# Launch the app
if __name__ == "__main__":
    args = parse_args()

    # Convert auth tuple if provided
    auth = tuple(args.auth) if args.auth else None

    launch_app(
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=not args.no_show_error,
        share=args.share,
        debug=args.debug,
        auth=auth,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile
    )
