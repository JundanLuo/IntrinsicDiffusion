IIW = "iiw"
SAW = "saw"
Hypersim = "hypersim"
DIODE = "diode"
InteriorVerse = "interiorverse"

IIW_dataset_dir = "./data/CGIntrinsics/IIW/"
SAW_dataset_dir = "./data/CGIntrinsics/SAW/"
Hypersim_dataset_dir = "./data/hypersim"
DIODE_dataset_dir = "./data/DIODE"
InteriorVerse_dataset_dir = "./data/interiorverse"

GAMMA = 1.0/2.2

albedo = "albedo"
shading = "shading"
normal = "surface normal"

# vae_scaling_factors = {
#     reflectance: 0.1806,
#     shading: 0.2049,
#     normal: None
# }

from modeling.diffusers.control_model import ControlNetModel
ControlModel = ControlNetModel  # define the used ControlModel
BASE_IMG_DIM = 64

REQUIRED_subfolders_IntrinsicControlNet = ["prompt_adapter", "controlnet"]
