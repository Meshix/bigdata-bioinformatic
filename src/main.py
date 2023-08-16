# conda environment: terminal in diesem ordner und dann conda activate bigbio
import os
import numpy as np
import data_processing as dp
# set data path ../BRATS/
data_path = os.path.join("..", "BRATS")
nii_path = os.path.join("..", "data")

if __name__ == "__main__":
    # load flair files
    files = dp.get_files(data_path, "flair")
    print("Loading FLAIR Files")
    flair_files = [dp.load_nifti(f) for f in files]
    for i, f in enumerate(files):
        # chunk file
        patches = dp.get_patches(flair_files[i])
        # save patches as numpy array
        np.save(os.path.join(nii_path, "flair", f"flair_{i}_chunked.npy"), patches)
        # print progress in same line
        print(f"Saved {i+1} of {len(flair_files)} images")
    # load seg files
    # seg = dp.get_files(data_path, "seg")
    # seg_files = [dp.load_nifti(f) for f in seg]

    # # q: what is the shortcut to comment out a block of code on windows?

    
    # for i, image in enumerate(seg_files):
    #     # chunk file
    #     print(f"Chunking Image {i+1} of {len(seg_files)} images")
    #     patches = dp.get_patches(image)
    #     for patch in patches:
    #         # everything greater than 0 is tumor, so set to 1
    #         patch[patch > 0] = 1
    #         patch[patch < 0] = 0
    #     # save patches as numpy array
    #     np.save(os.path.join(nii_path, "seg", f"seg_{i}_chunked.npy"), patches)
    #     print(f"Saved {i+1} of {len(seg_files)} images")


    
    
