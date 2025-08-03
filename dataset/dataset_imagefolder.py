# This file was initialized by ChatGPT.

import os
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from utils import image_util
from utils import constants as C


class ImageFolder(Dataset):
    """
    Dataset class for loading sRGB images from a specified folder.
    """
    dataset_name: str = "folder"
    GAMMA: float = C.GAMMA
    FILL_VALUE: float = 0.5  # value used for invalid pixels (depending on the training data processing)

    def __init__(self, image_dir: str,
                 channel_order: str = "CHW",
                 img_size: Optional[Tuple[int, int]] = (512, 512)) -> None:
        """
        Initialize the ImageFolder dataset.
        The channel_order (image channel) can be specified as either 'CHW' or 'HWC'.
        """
        super(ImageFolder, self).__init__()
        self.image_dir = image_dir
        self.channel_order = channel_order.lower()  # 'chw' or 'hwc'
        self.img_size = img_size

        # Gather and sort image paths, excluding explicit mask files.
        image_list = self.list_images(self.image_dir)  # list all images in the folder
        self.image_list: List[str] = [p for p in image_list if "mask" not in os.path.splitext(os.path.basename(p))[0]]

        # Image reader
        self.read_image = image_util.read_image

    def __len__(self) -> int:
        return len(self.image_list)

    def list_images(self, folder_path: str, exts: Sequence[str] = ('.jpg', '.jpeg', '.png')) -> List[str]:
        """Return sorted list of image paths with the given extensions."""
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in exts]
        image_files.sort()
        return image_files

    def apply_mask(self, img, mask, fill_value: float = 0.0):
        """
        Apply a binary mask to an image. Invalid regions are filled with a constant value.
        """
        img = img * mask + (1.0 - mask).expand_as(img) * fill_value
        return img

    def find_mask_path(self, image_path: str) -> Optional[str]:
        """Find a mask file with the naming pattern "<stem>_mask<ext>".
        """
        stem, _ = os.path.splitext(image_path)
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = f"{stem}_mask{ext}"
            if os.path.exists(candidate):
                return candidate
        return None

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve the image sample and associated data at the given index.
        """
        path = self.image_list[idx]

        # path
        filename = os.path.splitext(os.path.basename(path))[0]
        srgb_img_path = path
        mask_path = self.find_mask_path(srgb_img_path)

        # load image
        srgb_img = self.read_image(srgb_img_path, "tensor")  # CHW
        if mask_path is None:
            mask = torch.ones_like(srgb_img)
        else:
            mask = self.read_image(mask_path, "tensor")  # CHW
            mask = F.resize(mask, srgb_img.shape[1:], interpolation=F.InterpolationMode.BILINEAR,
                            antialias=True)  # CHW
            mask = (mask.min(dim=0, keepdim=True)[0] > 0.99).to(torch.float32)  # CHW

        # resize (if requested)
        data_tuple = (srgb_img, mask)
        if self.img_size is not None:
            data_tuple = (F.resize(d, size=self.img_size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
                          for d in data_tuple)

        # permute
        if self.channel_order == "chw":
            pass
        elif self.channel_order == "hwc":
            data_tuple = tuple([x.permute(1, 2, 0) for x in data_tuple])
        else:
            assert False, f"channel_order {self.channel_order} not supported."
        srgb_img, mask = data_tuple
        mask = (mask > 0.99).to(torch.float32).expand_as(srgb_img)
        # srgb_img = self.apply_mask(srgb_img, mask, fill_value=self.FILL_IN_VALUE)
        # rgb_img = image_util.srgb_to_rgb(srgb_img, self.GAMMA)

        out_dict = dict(text="",
                        srgb_img=srgb_img,
                        # rgb_img=rgb_img,
                        mask=mask,
                        filename=filename, channel_order=self.channel_order,
                        dataset_name=self.dataset_name)
        return out_dict


if __name__ == "__main__":
    img_size = (512, 512)
    data_dir = "../examples"
    dataset = ImageFolder(data_dir, channel_order="CHW",
                          img_size=img_size)
    print(f"Length of dataset: {len(dataset)}")
    for idx, data in enumerate(dataset):
        srgb_img = data["srgb_img"]
        mask = data["mask"]
        print(f"idx: {idx}, srgb_img: {srgb_img.shape}, mask: {mask.shape}")
        image_util.plot_images([srgb_img, mask], ["srgb_img", "mask"], columns=2, cmap="gray",
                               show=True)
