import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os

class ChunkDataSet(Dataset):
    def __init__(self, data_files, labels):
        self.data_files = data_files
        self.labels = labels

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data = np.load(file_path)
        label = self.labels[idx]

        # Randomly select a chunk and its corresponding label
        chunk_index = np.random.randint(0, data.shape[0])
        chunk = data[chunk_index]
        chunk_label = label[chunk_index]

        return torch.from_numpy(chunk), torch.tensor(chunk_label)

def get_data_loader(data_files, labels, batch_size=16):
    dataset = ChunkDataSet(data_files, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# print working directory
print(os.getcwd())
# get all filepaths in directory ../data/flair 
data_files = glob.glob('../data/flair/*.npy')

# get all files with label in their name in directory ../data/seg
labels = glob.glob('../data/seg/*labels*.npy')

print(len(data_files), len(labels))
dataloader = get_data_loader(data_files, labels, batch_size=16)

for batch in dataloader:
    inputs, targets = batch
    # Perform your training/validation/testing operations here using the inputs and targets
