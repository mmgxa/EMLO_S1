
# Imports

import sys
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets ,models,transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


import tableprint as tp
import torchmetrics
from torchsummary import summary


try:
    shutil.rmtree('csv_logs')
except:
    pass



def scorepro(targets, predictions):
        
    TP = torch.sum(  targets[targets==1] == predictions[targets==1] )
    TN = torch.sum(  targets[targets==0] == predictions[targets==0] )
    FP = torch.sum(  targets[targets==1] == predictions[targets==0] )
    FN = torch.sum(  targets[targets==0] == predictions[targets==1] )

    c1_acc = TP / (TP + FN)
    c0_acc = TN / (TN + FP)

    return c0_acc.item()*100, c1_acc.item()*100


top_model_weights_path = 'model.h5'



# Dataset and DataLoader
TRAIN_PATH = './data/train'
TEST_PATH = './data/validation'

img_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(TRAIN_PATH, transform=img_transforms)
val_data = datasets.ImageFolder(TEST_PATH, transform=img_transforms)



num_workers = 4
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,  num_workers=num_workers)





# Model
pretrained_vgg16 = models.vgg16(pretrained=True)

for name, param in pretrained_vgg16.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

num_features = pretrained_vgg16.classifier[6].in_features
pretrained_vgg16.classifier[6] = nn.Linear(in_features=num_features, out_features=2)



class Model(pl.LightningModule):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.avg_train_loss = 0.
        self.avg_valid_loss = 0.
        self.table_context = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.start_time = 0
        self.end_time = 0
        self.epoch_mins = 0
        self.epoch_secs = 0
        self.table_context = None
        self.train_accm = torchmetrics.Accuracy()
        self.valid_accm = torchmetrics.Accuracy()
        self.train_acc = 0.
        self.valid_acc = 0.
        self.c0 = 0.
        self.c1 = 0.
        

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optim


    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        _, predictions = torch.max(output, 1)
        acc_train = self.train_accm(predictions, target)
        loss = self.loss_fn(output, target)
        return {"loss": loss, "p": predictions, "y": target}
    
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        _, predictions = torch.max(output, 1)
        acc_train = self.valid_accm(predictions, target)
        loss_valid = self.loss_fn(output, target)
        return {"loss": loss_valid, "p": predictions, "y": target}


    def on_train_epoch_start(self) :
        self.start_time = time.time()


    def validation_epoch_end(self, outputs):
        if self.trainer.sanity_checking:
          return
        
        self.avg_valid_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.valid_acc = (self.valid_accm.compute() * 100).item()
        self.valid_accm.reset()
        self.log("epoch_num", int(self.current_epoch+1), on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.log("val_loss", self.avg_valid_loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        
        if self.current_epoch == self.trainer.max_epochs - 1:
          y = torch.cat([x['y'] for x in outputs])
          p = torch.cat([x['p'] for x in outputs])
          self.c0, self.c1 = scorepro(y, p)
          

    def training_epoch_end(self, outputs):
        self.avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.train_acc = (self.train_accm.compute() * 100).item()
        self.train_accm.reset()

    def on_train_epoch_end(self):
        self.end_time = time.time()
        self.epoch_mins, self.epoch_secs = self.epoch_time(self.start_time, self.end_time)
        time_int = f'{self.epoch_mins}m {self.epoch_secs}s'
    
        metrics = {'epoch': self.current_epoch+1, 'Train Acc': self.train_acc, 'Train Loss': self.avg_train_loss,  'Valid Acc': self.valid_acc, 'Valid Loss': self.avg_valid_loss}
        if self.table_context is None:
          self.table_context = tp.TableContext(headers=['epoch', 'Train Acc', 'Train Loss', 'Valid Acc', 'Valid Loss', 'Time'])
          self.table_context.__enter__()
        self.table_context([self.current_epoch+1, self.train_acc, self.avg_train_loss, self.valid_acc, self.avg_valid_loss, time_int])
        self.logger.log_metrics(metrics)

        if self.current_epoch == self.trainer.max_epochs - 1:
          self.table_context.__exit__()
          print(self.c0, self.c1)
          file = open("metrics.csv", "w")
          file.write("CatAcc=" + repr(int(self.c0)) + "\n")
          file.write("DogAcc=" + repr(int(self.c1)) + "\n")
          file.write("ValAcc=" + repr(int(self.valid_acc)) + "\n")


    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    


model = Model(pretrained_vgg16)

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath='./',
    filename='model',
    mode='max'
)


csvlogger = CSVLogger('csv_logs', name='EMLO_S1', version=0)
trainer = pl.Trainer(max_epochs=1, num_sanity_val_steps=0, logger=csvlogger, gpus=0, callbacks=[checkpoint_callback], log_every_n_steps=1)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


