import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

class PatchedFlairModule(pl.LightningModule):
    def __init__(self, train_flair_path, train_seg_path, val_flair_path, val_seg_path, batch_size, learning_rate):
        super().__init__()
        self.f1 = torchmetrics.F1Score(task='binary')
        self.auroc = torchmetrics.AUROC(task='binary')
        self.accuracy = torchmetrics.Accuracy(task='binary', average='macro')
        self.conv1 = nn.Sequential(nn.Conv3d(1, 64, kernel_size=5, stride=1, padding=0, bias=False),
                                   nn.SELU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=5, stride=1, padding=0, bias=False),
                                   nn.SELU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                   nn.SELU(inplace=True),
                                   nn.BatchNorm3d(256))
        self.lin2 = nn.Linear(256 * 20 * 20 * 20, 1)
        self.sigmoid = nn.Sigmoid()

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_flair_path = train_flair_path
        self.train_seg_path = train_seg_path
        self.val_flair_path = val_flair_path
        self.val_seg_path = val_seg_path
        self.save_conv = []  # new numpy array with tuples (shape (0,) )

        self.save_hyperparameters()
        
    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        conv = x
        x = x.view(x.shape[0], -1)
        x = self.lin2(x)
        return x, conv

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        y_hat = y_hat.squeeze()
        loss = self.loss_fn(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        y = y.unsqueeze(dim=1)
        loss = self.loss_fn(y_hat, y.float())
        self.log('val_loss', loss)
        self.f1.update(y_hat, y)
        self.auroc.update(y_hat, y)
        self.accuracy.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self):
        val_accuracy = self.accuracy.compute()
        val_f1 = self.f1.compute()
        val_auroc = self.auroc.compute()
        self.log("val_accuracy", val_accuracy)
        self.log("val_f1", val_f1)
        self.log("val_auroc", val_auroc)
        self.accuracy.reset()
        self.f1.reset()
        self.auroc.reset()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat, conv = self(x)
        y = y.unsqueeze(dim=1)
        loss = self.loss_fn(y_hat, y.float())

        #self.log('test_loss', loss)
        self.f1.update(y_hat, y)
        self.auroc.update(y_hat, y)
        self.accuracy.update(y_hat, y)

        conv_np = conv.detach().cpu().numpy() # convert conv to npy arr
        conv_np = conv_np.flatten()
        y_hat_np = y_hat.detach().cpu().numpy() # convert y_hat to npy arr
        y_hat_np = y_hat_np.flatten()
        # conv_np = conv.detach().cpu().numpy().flatten()  # Save conv as a numpy array
        # y_hat_np = y_hat.detach().cpu().numpy().flatten()  # Save y_hat as a numpy array
        self.save_conv.append((y_hat_np, conv_np))
        #self.conv_label = np.append(self.conv_label, batch)

        return {'predictions': y_hat, 'labels': y}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
