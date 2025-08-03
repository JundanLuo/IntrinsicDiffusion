# ------------------------------------------------------------------------
# This file is part of IntrinsicDiffusion.
# Licensed under the Apache License, Version 2.0.
# See https://github.com/JundanLuo/IntrinsicDiffusion for details.
# If you use this code, please cite the paper listed on the above website.
# ------------------------------------------------------------------------


import os
import time
import argparse
import random

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from accelerate.utils import set_seed

from utils import image_util
from modeling.intrinsicdiffusion_pipe import IntrinsicDiffusionPipeline, load_model
from dataset.dataset_imagefolder import ImageFolder


def infer_data(model: IntrinsicDiffusionPipeline, args,
               visualize_dir=None, visualize_use_mask=True,
               raw_pred_dir=None):
    """
    Infer intrinsic images from the input data using the provided model.
    :param model: IntrinsicDiffusionPipeline, the model to use for inference.
    :param args:
    :param visualize_dir: directory to save visualized results.
    :param visualize_use_mask: whether to use mask for visualization.
    :param raw_pred_dir: directory to save raw predictions as .npy files.
    :return:
    """
    # Output dir
    if visualize_dir is not None:
        os.makedirs(visualize_dir, exist_ok=True)
        split_img_dir = os.path.join(visualize_dir, "split")
        os.makedirs(split_img_dir, exist_ok=True)
    if raw_pred_dir is not None:
        os.makedirs(raw_pred_dir, exist_ok=True)

    # Dataset and dataloader
    print(f"=========================== Infer data ===========================")
    print(f"Resolution: {args.resolution}")
    dataset = ImageFolder(args.test_data_dir,
                          channel_order="CHW",
                          img_size=None)
    print(f"Total {len(dataset)} images in {args.test_data_dir} folder.")
    if args.max_test_samples is not None and args.max_test_samples < len(dataset):
        # randomly sample a subset of the dataset
        random_indices = random.sample(range(len(dataset)), args.max_test_samples)
        dataset = Subset(dataset, random_indices)
        print(f"The first of the random indices are {random_indices[:min(5, len(dataset))]}.")
        print(f"Infer a subset with {len(dataset)} images")
    else:
        print(f"Infer the entire folder.")
    dataloader = DataLoader(dataset,
                            shuffle=False, drop_last=False,
                            num_workers=args.dataloader_num_workers, batch_size=args.test_batch_size)

    # Infer
    is_linear_input = args.linear_input
    print(f"Is linear input: {is_linear_input}")
    for i, data in enumerate(dataloader):
        assert data["srgb_img"].shape == data["mask"].shape, "Inconsistent shape"
        if i % 5 == 0:
            print(f"Infer: {i}th /{len(dataloader)} ......")

        with torch.no_grad():
            pred_r, pred_s, pred_n, reconstr_input = model.infer_intrinsic_images(
                _input_img=data["srgb_img"],
                infer_img_size=args.resolution,
                is_linear_input=is_linear_input,
                output_original_size=args.output_original_size,
                pred_albedo=True,
                pred_shading=True,
                pred_normal=True,
                inference_chunk_batch_size=args.inf_chunk_batch_size,)
            torch.cuda.empty_cache()
            pred_r, pred_s, pred_n, reconstr_input = (d.to(torch.float32).cpu() # predictions might be float16
                                                      for d in (pred_r, pred_s, pred_n, reconstr_input))

        # Visualize predicted images
        if visualize_dir is not None:
            for idx in range(pred_r.size(0)):
                input_img = data["srgb_img"][idx].to(pred_r.device)
                mask_input = data["mask"][idx].to(pred_r.device) if visualize_use_mask \
                    else torch.ones_like(pred_r[idx])
                vis_pred_s = (pred_s[idx] * mask_input).clamp(min=0.0, max=10.0)
                vis_pred_r = pred_r[idx] * mask_input
                vis_pred_s *= image_util.get_scale_alpha(vis_pred_s, mask_input, 0.95, 0.85) * mask_input
                vis_pred_r *= image_util.get_scale_alpha(vis_pred_r, mask_input, 0.95, 0.85) * mask_input
                vis_pred_n = (pred_n[idx] + 1.0)/2.0 * mask_input
                vis_reconstr = reconstr_input[idx] * mask_input
                vis_reconstr *= image_util.get_scale_alpha(vis_reconstr, mask_input, 0.99, 0.85) * mask_input
                plt = image_util.plot_images([input_img, mask_input, vis_reconstr,
                                              vis_pred_r, vis_pred_s, vis_pred_n],
                                             titles=["input_img", "mask", "reconstr",
                                                     "pred_r", "pred_s", "pred_n"],
                                             columns=3,
                                             figsize_base=8,
                                             HW_ratio=vis_pred_r.shape[1]/vis_pred_r.shape[2],
                                             show=False)
                plt.savefig(os.path.join(visualize_dir, f"{data['filename'][idx]}_result.jpg"))
                plt.close()
                if True:
                    vis_dict = {
                        "input": input_img,
                        "r": vis_pred_r,
                        "s": vis_pred_s,
                        "n": vis_pred_n,
                        "reconstr": vis_reconstr,
                        "mask_input": mask_input,
                        }
                    for key, item in vis_dict.items():
                        image_util.save_image(item, os.path.join(split_img_dir,
                                                                 f"{data['filename'][idx]}_{key}.png"))
        # Save predictions to .npy files
        if raw_pred_dir is not None:
            for idx in range(pred_r.size(0)):
                filename = data['filename'][idx]
                saving_dict = {
                    "r": pred_r[idx].permute(1, 2, 0).cpu().numpy(),  # HWC
                    "s": pred_s[idx].permute(1, 2, 0).cpu().numpy(),  # HWC
                    "n": pred_n[idx].permute(1, 2, 0).cpu().numpy(),  # HWC
                }
                for k in saving_dict.keys():
                    np.save(os.path.join(raw_pred_dir, f"{filename}_{k}.npy"), saving_dict[k])


def parse_args(input_args=None):
    """ Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_test_epochs",
                        type=int,
                        default=1,
                        help="Test with `--num_test_epochs` different seeds.")
    parser.add_argument("--model_name",
                        type=str,
                        default="controlnet",
                        help="Model name.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    # parser.add_argument(
    #     "--tokenizer_name",
    #     type=str,
    #     default=None,
    #     help="Pretrained tokenizer name or path if not the same as model_name",
    # )
    parser.add_argument(
        "--vae_scaling_factor",
        type=float,
        default=None,
        help=("The scaling factor for the VAE."),
    )
    parser.add_argument(
        "--prompt_adapter_path",
        type=str,
        default=None,
        help="Path to a pretrained prompt adapter.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/evaluation",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        default=False,
        help="Whether to add a timestamp to the output directory.",
    )
    parser.add_argument("--seed", type=int, default=999, help="A seed for reproducible test.")

    def int_or_tuple(value):
        if value == 'ori':
            return 'ori'
        try:
            # Try to convert to a single integer
            return int(value)
        except ValueError:
            # If that fails, try to convert to a tuple of integers
            return tuple(map(int, value.strip('()').split(',')))

    parser.add_argument(
        '--inf_seq',
        type=str,
        nargs='+',
        default=[],
        help="List defining the inference sequence of the intrinsic image decomposition model. "
             "Examples: ['AS', 'N'], ['ASN'], ['A', 'S', 'N']."
    )
    parser.add_argument(
        '--inf_chunk_batch_size',
        type=int,
        default=0,
        help="Chunk batch size for inference. 0 means infer all in one batch. "
             "Setting small positive values can help reduce memory usage."
    )
    parser.add_argument(
        "--resolution",
        type=int_or_tuple,
        default=None,
        help="Specifies the resolution of the inferring data as either a single integer or a tuple of integers."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="Batch size for the test dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # parser.add_argument(
    #     "--mixed_precision",
    #     type=str,
    #     default=None,
    #     choices=["no", "fp16", "bf16"],
    #     help=(
    #         "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
    #         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
    #         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    #     ),
    # )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        default=False,
        help="Whether or not to use xformers."
    )
    # parser.add_argument(
    #     "--set_grads_to_none",
    #     action="store_true",
    #     help=(
    #         "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
    #         " behaviors, so disable this argument if it causes any problems. More info:"
    #         " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
    #     ),
    # )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the test data."
        ),
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument("--linear_input", action="store_true", default=False,
                        help="Whether the input data is in linear space.")
    parser.add_argument("--output_original_size", action="store_true", default=False,
                        help="Whether to output the original size.")
    parser.add_argument('--ddim_steps', type=int, default=20,
                        help='Number of DDIM steps.')
    parser.add_argument('--agg_num', type=int, default=1,
                        help='Number of aggregated samples.')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Guidance scale.')
    # parser.add_argument(
    #     "--num_generated_images",
    #     type=int,
    #     default=1,
    #     help="Number of images to be generated for each `--test_image`, `--test_prompt` pair",
    # )
    parser.add_argument(
        "--vis_wo_mask", action="store_true",
        default=False,
        help="Whether to visualize predictions use mask."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Check arguments
    assert args.test_data_dir is not None, "Specify `--test_data_dir`"
    assert args.guidance_scale == 1.0, "Guidance scale must be 1.0, modeling/diffusers/control_model.py hasn't been updated to support it yet."

    return args


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    print(args)
    # Output directory
    base_dir = args.output_dir
    if args.timestamp:  # add timestamp
        base_dir = os.path.join(base_dir,
                                f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())}')
    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_name, device, args)
    print(f"Using device {device}.")

    # Infer
    for i in range(args.num_test_epochs):  # test with different seeds if args.num_test_epochs > 1
        print(f"\nTest epoch {i+1}/{args.num_test_epochs}:")
        # Seed
        if i > 0 or args.seed == -1:
            args.seed = random.randint(0, 65535)
        set_seed(args.seed)
        print(f"Using seed {args.seed}")
        # Output dir
        output_dir = base_dir
        if args.num_test_epochs > 1:
            output_dir = os.path.join(output_dir, f"epoch_{i}_seed_{args.seed}")
        os.makedirs(output_dir, exist_ok=True)
        # Infer
        infer_data(model, args,
                   visualize_dir=output_dir,
                   visualize_use_mask=not args.vis_wo_mask,
                   raw_pred_dir=os.path.join(output_dir, "raw_pred"))
