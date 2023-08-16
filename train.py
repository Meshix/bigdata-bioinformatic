from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import src.data_processing
import src.model
import os
import numpy as np
from torch.utils.data import DataLoader
import glob
from pytorch_lightning.loggers import TensorBoardLogger



def generate_file_index_array(file_list, output_path, new=True):
    if not new:
        return np.load(output_path)
    file_index_array = np.empty(len(file_list), dtype=object)
    previous_size = 0
    for i in range(len(file_list)):
        # This part allows us to calculate the size of the array
        # by just looking in the file header and not needing to load the whole array
        # via np.load
        with open(file_list[i], 'rb') as f:
            # Check if the file starts with the correct magic bytes
            if f.read(6) != b'\x93NUMPY':
                raise ValueError('Invalid NPY file')
            major_version = ord(f.read(1))
            minor_version = ord(f.read(1))
            header_length = np.frombuffer(f.read(2), dtype=np.uint16)[0]
            header = f.read(header_length)
            shape_start = header.index(b'shape') + 9
            shape_end = header.index(b')', shape_start)
            shape_data = header[shape_start:shape_end].decode()
            shape = tuple(map(int, shape_data.split(',')))
            file_index_array[i] = (file_list[i], (previous_size + shape[0]))
            previous_size += shape[0]
    np.save(output_path, file_index_array)
    return file_index_array


if __name__ == "__main__":
    print("Starting Training Process! Loading Data...")
    print(os.getcwd())
    # get file paths of flair files
    flair_files = sorted(glob.glob('../data/flair/*.npy'))
    print(f"Total Data: {len(flair_files)}")
    # flair_files = flair_files[:20]

    # load 80% of flair files as training data
    train_flair = flair_files[:int(len(flair_files) * 0.8)]
    print(f"Training Data: {len(train_flair)}")
    # loaded_train_flair = [np.load(f) for f in train_flair]
    print("FLAIR Test Loaded")

    # load 20% of flair files as validation data
    val_flair = flair_files[int(len(flair_files) * 0.8):]
    print(f"Validation Data: {len(val_flair)}")
    # loaded_val_flair = [np.load(f) for f in val_flair]
    print("FLAIR Val Loaded")

    # get file paths of seg files
    seg_files = sorted(glob.glob('../data/seg/*labels*.npy'))
    # seg_files = seg_files[:20]
    # load 80% of seg files as training data
    train_seg = seg_files[:int(len(seg_files) * 0.8)]
    # loaded_train_seg = [np.load(f) for f in train_seg]
    print("SEG Test Loaded")

    # load 20% of seg files as validation data
    val_seg = seg_files[int(len(seg_files) * 0.8):]
    # loaded_val_seg = [np.load(f) for f in val_seg]
    print("SEG Val Loaded")

    # get file paths of index files
    index_files = sorted(glob.glob('../data/indices/*indices*.npy'))
    # load 80% of index files as training data
    train_index = index_files[:int(len(index_files) * 0.8)]
    print("Index Test Loaded")

    # load 20% of index files as validation data
    val_index = index_files[int(len(index_files) * 0.8):]
    print("Index Val Loaded")

    print("Starting to calculate file dictionary ..")
    print("For test data")
    test_file_index_array = generate_file_index_array(train_index, '../data/testdata_array.npy')
    print("For val data")
    val_file_index_array = generate_file_index_array(val_index, '../data/valdata_array.npy')

    # torch.cuda.set_per_process_memory_fraction(0.8)

    # create data set
    print("Creating data sets")
    train_data = data_processing.PatchedFlairDataset(train_flair, train_seg, train_index, test_file_index_array)
    val_data = data_processing.PatchedFlairDataset(val_flair, val_seg, val_index, val_file_index_array)

    # create data loader
    # num workers must be 0 for debugging in train_loader
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=0)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=True)

    logger = TensorBoardLogger('../tb_logs', name='test1')

    model = model.PatchedFlairModule(train_flair, train_seg, val_flair, val_seg, 64, 0.0001)
    trainer = Trainer(default_root_dir='/prodi/bioinfdata/user/bdprakti/bigdata_ss_23/bigdata-bioinfo/checkpoints',
                      devices=[3], accelerator="gpu", max_epochs=100, num_sanity_val_steps=1, callbacks=[early_stopping], logger=logger)
    trainer.fit(model, train_loader, val_loader)

    # test the model
    # trainer.test(model, dataloaders=DataLoader(test_data)) # TODO add test set
    # trainer.test(model, test_loader)
