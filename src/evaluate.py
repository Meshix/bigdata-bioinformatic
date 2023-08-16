import glob
import os

from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

import data_processing
import model
import train

import pickle

if __name__ == "__main__":
    #TODO load 20% of data and not only 1%
    print(os.getcwd())
    # get file paths of flair files
    flair_files = sorted(glob.glob('../data/flair/*.npy'))
    print(f"Total Data: {len(flair_files)}")

    # load 20% of flair files as validation data
    val_flair = flair_files[int(len(flair_files) * 0.8):]
    print(f"Validation Data: {len(val_flair)}")
    # loaded_val_flair = [np.load(f) for f in val_flair]
    print("FLAIR Val Loaded")

    # get file paths of seg files
    seg_files = sorted(glob.glob('../data/seg/*labels*.npy'))

    # get file paths of index files
    index_files = sorted(glob.glob('../data/indices/*indices*.npy'))
    # load 20% of seg files as validation data
    val_seg = seg_files[int(len(seg_files) * 0.8):]

    # load 20% of index files as validation data
    val_index = index_files[int(len(index_files) * 0.8):]
    print("Index Val Loaded")

    val_index = index_files[int(len(index_files) * 0.8):]
    print("Index Val Loaded")
    print("Starting to calculate file dictionary ..")
    print("For val data")
    val_file_index_array = train.generate_file_index_array(val_index, '../data/valdata_array.npy')
    val_data = data_processing.PatchedFlairDataset(val_flair, val_seg, val_index, val_file_index_array)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=0)
    e_model = model.PatchedFlairModule.load_from_checkpoint('tb_logs/test1/version_1/checkpoints/epoch=12-step=10075.ckpt', )
    trainer = Trainer(default_root_dir='/prodi/bioinfdata/user/bdprakti/bigdata_ss_23/bigdata-bioinfo/tb_logs/test1',
                      devices=[3], accelerator="gpu", max_epochs=1, num_sanity_val_steps=1)
    outputs = trainer.predict(model=e_model, dataloaders=val_loader)
    
    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(outputs, f)
