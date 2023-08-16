# imports
import numpy as np
import nibabel as nib
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
import re


# simple binary search
def find_next_lower_index(numbers, target):
    left = 0
    right = len(numbers) - 1
    while left <= right:
        mid = (left + right) // 2
        if numbers[mid][1] > target:
            right = mid - 1
        else:
            left = mid + 1
    return left


# Dataset Class
class PatchedFlairDataset(Dataset):
    def __init__(self, flair_file_list, seg_file_list, index_file_list, file_index_array):
        self.flair_files = flair_file_list
        self.seg_files = seg_file_list
        self.index_files = index_file_list
        self.file_index_array = file_index_array

    def __getitem__(self, index):
        array_index = find_next_lower_index(self.file_index_array, index)
        # choose file
        flair_file = np.load(self.flair_files[array_index])
        seg_file = np.load(self.seg_files[array_index])
        index_file = np.load(self.index_files[array_index])

        segment_index = index - self.file_index_array[array_index][1]
        patch_index = index_file[segment_index]

        flair_patch = flair_file[patch_index[0], patch_index[1], patch_index[2]]
        seg_patch = seg_file[patch_index[0], patch_index[1], patch_index[2]]

        return flair_patch, seg_patch

    def __len__(self):
        return self.file_index_array[-1][1]


# function to load nifti files
def load_nifti(path):
    img = nib.load(path)
    return img


# write a function to get patches from a nifti file
def get_patches(img):
    # Get the image data as a numpy array
    data = img.get_fdata()

    # Calculate the number of chunks in each dimension
    chunks_x = data.shape[0] // 32
    chunks_y = data.shape[1] // 32
    chunks_z = data.shape[2] // 32

    # Use the as_strided function to create a view of the data as a sequence of 32x32x32 chunks
    chunk_shape = (32, 32, 32)
    chunk_strides = data.strides
    chunks = np.lib.stride_tricks.as_strided(data, shape=(chunks_x, chunks_y, chunks_z, *chunk_shape), strides=(
    32 * chunk_strides[0], 32 * chunk_strides[1], 32 * chunk_strides[2], *chunk_strides))

    return chunks


# function to get all files in a directory
def get_files(path, filter=None, type=".nii", include_label_files=False):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if type in file:
                if not include_label_files and str(file).__contains__('_labels'):
                    pass
                else:
                    files.append(os.path.join(r, file))
    if filter is not None:
        files = [f for f in files if filter in f]
    return files


def save_nifti(patches, path):
    img = nib.Nifti1Image(np.array(patches), np.eye(4))
    nib.save(img, path)


def show_image(img):
    plt.imshow(img.get_fdata()[:, :, 100])
    plt.show()


def data_post_processing(directory_path):
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            print('processing ' + str(file))
            dest = os.path.join(directory_path, os.path.splitext(file)[0] + '_labels')
            chunks = np.load(os.path.join(directory_path, file))
            label_arr = np.any(chunks == 1, axis=(3, 4, 5)).astype(int)
            np.save(dest, label_arr)


def data_post_processing_generator(seg_path, dest_path):
    for file in os.listdir(seg_path):
        if os.path.isfile(os.path.join(seg_path, file)):
            if '_labels' in os.path.splitext(file)[0]:
                print('processing ' + str(file))
                labeled_seg = np.load(os.path.join(seg_path, file))
                amount_healthy = np.count_nonzero(labeled_seg)
                amount_sick = labeled_seg.size - amount_healthy
                print('amount sick: ' + str(amount_sick))
                print('amount healthy: ' + str(amount_healthy))
                indices = []
                new_labeled_seg = []
                for i in range(6):
                    for j in range(7):
                        for k in range(6):
                            if labeled_seg[i, j, k] == 1 and amount_sick > 0:
                                indices.append((i, j, k))
                                amount_sick -= 1
                                new_labeled_seg.append(1)
                            if labeled_seg[i, j, k] == 0 and amount_healthy > 0:
                                indices.append((i, j, k))
                                amount_healthy -= 1
                                new_labeled_seg.append(0)
                            if amount_healthy == 0 and amount_sick == 0:
                                break
                np.save(os.path.join(dest_path, 'relevant_indices_' + file.split("_")[1]), indices)


def rename_file_with_padded_number(directory):
    pattern = r'(\d+)'
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Check if the file is a regular file
        if os.path.isfile(filepath):

            # Extract the number from the filename using regex
            match = re.search(pattern, filename)
            if match:
                number = match.group(1)  # Extract the matched number
                padded_number = number.zfill(5)  # Pad the number with zeros

                # Construct the new filename with the padded number
                new_filename = re.sub(pattern, padded_number, filename)
                new_filepath = os.path.join(directory, new_filename)

                # Rename the file
                os.rename(filepath, new_filepath)
                print(f"Renamed '{filename}' to '{new_filename}'")


if __name__ == "__main__":
    # data_post_processing_generator(os.path.join("..", "data/seg"),
    #                               os.path.join("..", "data/indices"))
    rename_file_with_padded_number(os.path.join("..", "data/seg"))
