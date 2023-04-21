# imports
import numpy as np
import nibabel as nib
from skimage.util import view_as_blocks
import os

# function to load nifti files
def load_nifti(path):
    img = nib.load(path)
    return img

# function to split nifti files in 64x64x64 patches
def split_nifti(img, patch_size):
    img_data = img.get_fdata()
    img_data = img_data.astype(np.float32)
    img_data = img_data / 255
    img_data = img_data.transpose(2, 1, 0)
    img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], img_data.shape[2], 1)
    img_patches = view_as_blocks(img_data, block_shape=(patch_size, patch_size, patch_size, 1))
    return img_patches

# function to get all files in a directory
def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.nii' in file:
                files.append(os.path.join(r, file))
    return files

    