import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml
import argparse
from dataset import FaceDataset,collate_fn
from easydict import EasyDict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ModelFace import FaceModel
from model import backbone_timm,build_model
from torchsampler import ImbalancedDatasetSampler
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    return args

def prepare_data(cfg, args):
    train_dataset = FaceDataset( cfg.data.train_anno,cfg.data.img_folder)
    val_dataset = FaceDataset( cfg.data.val_anno,cfg.data.img_folder)
    print("Total Train Dataset:", len(train_dataset))
    print("Total Val Dataset:", len(val_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True,collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    
    args = get_args()    
    with open(args.cfg_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = EasyDict(cfg)
    train_dataloader, val_dataloader = prepare_data(cfg, args)
    logger = TensorBoardLogger("mode_all", name=cfg.model.model_name)
    early_stop = EarlyStopping(monitor="val_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    backbone,embed_size=backbone_timm(cfg.model.model_name)

    model=FaceModel(backbone,embed_size,cfg)
    trainer = pl.Trainer(
        max_epochs=200,
        logger=logger,
        accelerator = 'gpu',
        check_val_every_n_epoch=1,
        callbacks=[early_stop, checkpoint_callback],
        )
    trainer.fit(model=model,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
    