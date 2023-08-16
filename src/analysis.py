import nibabel as nib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sklearn.metrics as metrics
from sklearn.cluster import KMeans

import matplotlib.colors as colors

def plot_3d(image_data):
    # Visualize the 3D image using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    x, y, z = image_data.nonzero()

    for i in range(0, image_data.shape[0], 64):
        for j in range(0, image_data.shape[1], 64):
            ax.plot([i, i], [j, j], [0, image_data.shape[2]], color='blue', linestyle='dashed', zorder=5)

    for i in range(0, image_data.shape[0], 64):
        for k in range(0, image_data.shape[2], 64):
            ax.plot([i, i], [0, image_data.shape[1]], [k, k], color='red', linestyle='dashed', zorder=5)

    for j in range(0, image_data.shape[1], 64):
        for k in range(0, image_data.shape[2], 64):
            ax.plot([0, image_data.shape[0]], [j, j], [k, k], color='green', linestyle='dashed', zorder=10)

    ax.scatter(x, y, z, zdir="z", c=image_data[x, y, z], cmap="gray", zorder=2)
    # save the image
    plt.savefig('../figures/3dplot_chunked_v3.png')


def plot_3d_cube(image_data):
    # Visualize the 3D image using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    x, y, z = image_data.nonzero()

    # set cube size
    cube_size = 32
    grid_rows = 6
    grid_cols = 7
    grid_slices = 6

    # Center of the cube within the full image
    center_x = image_data.shape[0] // 2
    center_y = image_data.shape[1] // 2
    center_z = image_data.shape[2] // 2

    # Calculate cube boundaries
    x_start = center_x - cube_size // 2
    x_end = x_start + cube_size
    y_start = center_y - cube_size // 2
    y_end = y_start + cube_size
    z_start = center_z - cube_size // 2
    z_end = z_start + cube_size

    # Calculate grid spacing
    grid_spacing_x = (x_end - x_start) // (grid_rows - 1)
    grid_spacing_y = (y_end - y_start) // (grid_cols - 1)
    grid_spacing_z = (z_end - z_start) // (grid_slices - 1)

    # Visualize the 3D image using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    # x, y, z = image_data.nonzero()
    x, y, z = image_data[x_start:x_end, y_start:y_end, z_start:z_end].nonzero()

    # Plot the grid (dashed lines)
    for i in range(x_start, x_end + 1, grid_spacing_x):
        ax.plot([i, i], [y_start, y_end], [z_start, z_start], color='blue', linestyle='dashed', zorder=5)
        ax.plot([i, i], [y_start, y_end], [z_end, z_end], color='blue', linestyle='dashed', zorder=5)
        ax.plot([i, i], [y_start, y_start], [z_start, z_end], color='blue', linestyle='dashed', zorder=5)
        ax.plot([i, i], [y_end, y_end], [z_start, z_end], color='blue', linestyle='dashed', zorder=5)

    for j in range(y_start, y_end + 1, grid_spacing_y):
        ax.plot([x_start, x_end], [j, j], [z_start, z_start], color='red', linestyle='dashed', zorder=5)
        ax.plot([x_start, x_end], [j, j], [z_end, z_end], color='red', linestyle='dashed', zorder=5)
        ax.plot([x_start, x_start], [j, j], [z_start, z_end], color='red', linestyle='dashed', zorder=5)
        ax.plot([x_end, x_end], [j, j], [z_start, z_end], color='red', linestyle='dashed', zorder=5)

    for k in range(z_start, z_end + 1, grid_spacing_z):
        ax.plot([x_start, x_end], [y_start, y_start], [k, k], color='green', linestyle='dashed', zorder=10)
        ax.plot([x_start, x_end], [y_end, y_end], [k, k], color='green', linestyle='dashed', zorder=10)
        ax.plot([x_start, x_start], [y_start, y_end], [k, k], color='green', linestyle='dashed', zorder=10)
        ax.plot([x_end, x_end], [y_start, y_end], [k, k], color='green', linestyle='dashed', zorder=10)

    # Plot the cube data

    ax.scatter(x + x_start, y + y_start, z + z_start, zdir="z",
               c=image_data[x_start:x_end, y_start:y_end, z_start:z_end][x, y, z], cmap="gray", zorder=2)

    # ax.scatter(x, y, z, zdir="z", c=image_data[x, y, z], cmap="gray", zorder=2)
    # save the image
    plt.savefig('../figures/3dplot_chunked_cube_v3.png')

def plot_only_chunk(image_data):
    # Define cube dimensions
    cube_size = 32
    grid_rows = 6
    grid_cols = 7
    grid_slices = 6

    # Center of the cube within the full image
    center_x = image_data.shape[0] // 2
    center_y = image_data.shape[1] // 2
    center_z = image_data.shape[2] // 2

    # Calculate cube boundaries
    x_start = center_x - cube_size // 2
    x_end = x_start + cube_size
    y_start = center_y - cube_size // 2
    y_end = y_start + cube_size
    z_start = center_z - cube_size // 2
    z_end = z_start + cube_size

    # Calculate grid spacing
    grid_spacing_x = (x_end - x_start) // (grid_rows - 1)
    grid_spacing_y = (y_end - y_start) // (grid_cols - 1)
    grid_spacing_z = (z_end - z_start) // (grid_slices - 1)

    # Visualize the 3D image using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # Plot the cube data
    x, y, z = image_data[x_start:x_end, y_start:y_end, z_start:z_end].nonzero()
    ax.scatter(x + x_start, y + y_start, z + z_start, zdir="z", c=image_data[x_start:x_end, y_start:y_end, z_start:z_end][x, y, z], cmap="gray", zorder=2)

    # Hide axes
    ax.set_axis_off()

    # Save the image
    plt.savefig('../figures/3dplot_chunked_cube_only.png', bbox_inches='tight', pad_inches=0)

def plot_cm_and_roc():
    with open("saved_dictionary.pkl", "rb") as fp:
        preds = pickle.load(fp)
    predictions_list = []
    labels_list = []

    for dict_item in preds:
        predictions_list.append(dict_item['predictions'].cpu().item())
        labels_list.append(dict_item['labels'].cpu().item())

    # Convert lists to numpy arrays
    predictions_np = np.array(predictions_list)
    y_pred = np.where(predictions_np >= 1, 1, 0)
    labels_np = np.array(labels_list)

    fpr, tpr, threshold = metrics.roc_curve(labels_np, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    np.savetxt("roc_data.csv", [fpr, tpr], delimiter=",")

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}
    plt.rc('font', **font)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("../figures/roc.png")

    cm = metrics.confusion_matrix(labels_np, y_pred)

    disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot()
    plt.savefig("../figures/cm.png")

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn+fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("Specificity", specificity, "Precision", precision, "Recall", recall)
    print("F-Score: ", 2*(precision*recall)/(precision + recall))

def plot_3d_only_brain(image_data):
        # Visualize the 3D image using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    x, y, z = image_data.nonzero()

    ax.scatter(x, y, z, zdir="z", c=image_data[x, y, z], cmap="gray", zorder=2)
    # save the image
    plt.savefig('../figures/3dplot_only_brain.png')
# create main function where functions are called
def main():
    # Load the Nifti file
    #nifti_file = nib.load('../BRATS/BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_flair.nii.gz')
    # Get the image data
    #image_data = nifti_file.get_fdata()
    # call the functions
    # plot_3d(image_data) # full brain
    #plot_3d_cube(image_data) # chunk of the brain
    #plot_only_chunk(image_data)
    #plot_3d_only_brain(image_data)
    #plot_cm_and_roc

# call the main function
if __name__ == "__main__":
    main()
