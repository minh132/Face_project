import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score,MulticlassAccuracy,MulticlassAveragePrecision

class FaceModel(pl.LightningModule):
    def __init__(self, backbone,embed_size ,cfg ):
        super().__init__()
        self.cfg=cfg
        self.backbone=backbone
        # self.age_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_age))
        self.emo_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256,cfg.class_Infor.num_emo ))
        # self.skin_head= nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_skin))
        # self.mask_head=nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_mask))
        # self.gender_head=nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_gender))
        # self.race_head=nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_race))
        self.accuracy=MulticlassAccuracy(num_classes=cfg.class_Infor.num_emo)
        self.f1=MulticlassF1Score(num_classes=cfg.class_Infor.num_emo)

        
    def forward(self,x):

        x=self.backbone(x)
        # x_age=self.age_head(x)
        x_emo=self.emo_head(x)
        # x_skin=self.skin_head(x)
        # x_mask=self.mask_head(x)
        # x_gender=self.gender_head(x)
        # x_race=self.race_head(x)
        # return x_age,x_emo,x_skin,x_mask,x_gender,x_race
        return x_emo
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-6, weight_decay=1e-4
        )
    
    def training_step(self, batch, batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        
        out_emo=self(x)
        # loss_age=F.cross_entropy(out_age,target_age)
        loss_emo=F.cross_entropy(out_emo,target_emo)
        # loss_skin=F.cross_entropy(out_skin,target_skin)
        # loss_mask=F.cross_entropy(out_mask,target_mask)
        # loss_gender=F.cross_entropy(out_gender,target_gender)
        # loss_race=F.cross_entropy(out_race,target_race)
        # loss=loss_age*0.2+loss_emo*0.2+loss_skin*0.2+loss_mask*0.1+loss_gender*0.1+loss_race*0.2
        self.log("train_loss", loss_emo, prog_bar=True, on_step=True)
        return loss_emo
    def validation_step(self,batch,batch_idx):
        x,target_emo,target_race,target_mask,target_skin,target_emo,target_gender=batch
        out_emo=self(x)
        loss_emo=F.cross_entropy(out_emo,target_emo)
        # loss_emo=F.cross_entropy(out_emo,target_emo)
        # loss_skin=F.cross_entropy(out_skin,target_skin)
        # loss_mask=F.cross_entropy(out_mask,target_mask)
        # loss_gender=F.cross_entropy(out_gender,target_gender)
        # loss_race=F.cross_entropy(out_race,target_race)
        # loss=loss_emo*0.2+loss_emo*0.2+loss_skin*0.2+loss_mask*0.1+loss_gender*0.1+loss_race*0.2
        self.log("val_loss", loss_emo, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        acc=self.accuracy(out_emo,target_emo)
        f1_score=self.f1(out_emo,target_emo)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("val_f1",f1_score,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        
        return loss_emo


class AgeModel(pl.LightningModule):
    def __init__(self, backbone,embed_size ,cfg ):
        super().__init__()
        self.cfg=cfg
        self.backbone=backbone
        self.age_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_age))   
        self.accuracy=MulticlassAccuracy(num_classes=cfg.class_Infor.num_age)
        self.f1=MulticlassF1Score(num_classes=cfg.class_Infor.num_age)

        
    def forward(self,x):

        x=self.backbone(x)

        x_age=self.age_head(x)
        return x_age
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-4, weight_decay=1e-4
        )
    
    def training_step(self, batch, batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch      
        out_age=self(x)
        loss_age=F.cross_entropy(out_age,target_age)     
        self.log("train_loss", loss_age, prog_bar=True, on_step=True)
        return loss_age
    
    def validation_step(self,batch,batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        out_age=self(x)
        loss_age=F.cross_entropy(out_age,target_age)
        self.log("val_loss", loss_age, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        acc=self.accuracy(out_age,target_age)
        f1_score=self.f1(out_age,target_age)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("val_f1",f1_score,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        return loss_age


class EmoModel(pl.LightningModule):
    def __init__(self, backbone,embed_size ,cfg ):
        super().__init__()
        self.cfg=cfg
        self.backbone=backbone
        self.emo_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_emo))   
        self.accuracy=MulticlassAccuracy(num_classes=cfg.class_Infor.num_emo)
        self.f1=MulticlassF1Score(num_classes=cfg.class_Infor.num_emo)

        
    def forward(self,x):
        x=self.backbone(x)
        x_emo=self.emo_head(x)
        return x_emo
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-6, weight_decay=1e-4
        )
    
    def training_step(self, batch, batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch      
        out_emo=self(x)
        loss_emo=F.cross_entropy(out_emo,target_emo)     
        self.log("train_loss", loss_emo, prog_bar=True, on_step=True)
        return loss_emo
    
    def validation_step(self,batch,batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        out_emo=self(x)
        loss_emo=F.cross_entropy(out_emo,target_emo)
        self.log("val_loss", loss_emo, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        acc=self.accuracy(out_emo,target_emo)
        f1_score=self.f1(out_emo,target_emo)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("val_f1",f1_score,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        return loss_emo
    
 
class GenderModel(pl.LightningModule):
    def __init__(self, backbone,embed_size ,cfg ):
        super().__init__()
        self.cfg=cfg
        self.backbone=backbone
        self.gender_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_gender))   
        self.accuracy=MulticlassAccuracy(num_classes=cfg.class_Infor.num_gender)
        self.f1=MulticlassF1Score(num_classes=cfg.class_Infor.num_gender)

        
    def forward(self,x):
        x=self.backbone(x)
        x_gender=self.gender_head(x)
        return x_gender
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-6, weight_decay=1e-4
        )
    
    def training_step(self, batch, batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch      
        out_gender=self(x)
        loss_gender=F.cross_entropy(out_gender,target_gender)     
        self.log("train_loss", loss_gender, prog_bar=True, on_step=True)
        return loss_gender
    
    def validation_step(self,batch,batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        out_gender=self(x)
        loss_gender=F.cross_entropy(out_gender,target_gender)
        self.log("val_loss", loss_gender, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        acc=self.accuracy(out_gender,target_gender)
        f1_score=self.f1(out_gender,target_gender)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("val_f1",f1_score,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        return loss_gender

class RaceModel(pl.LightningModule):
    def __init__(self, backbone,embed_size ,cfg ):
        super().__init__()
        self.cfg=cfg
        self.backbone=backbone
        self.race_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_race))   
        self.accuracy=MulticlassAccuracy(num_classes=cfg.class_Infor.num_race)
        self.f1=MulticlassF1Score(num_classes=cfg.class_Infor.num_race)

        
    def forward(self,x):
        x=self.backbone(x)
        x_race=self.race_head(x)
        return x_race
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-6, weight_decay=1e-4
        )
    
    def training_step(self, batch, batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch      
        out_race=self(x)
        loss_race=F.cross_entropy(out_race,target_race)     
        self.log("train_loss", loss_race, prog_bar=True, on_step=True)
        return loss_race
    
    def validation_step(self,batch,batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        out_race=self(x)
        loss_race=F.cross_entropy(out_race,target_race)
        self.log("val_loss", loss_race, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        acc=self.accuracy(out_race,target_race)
        f1_score=self.f1(out_race,target_race)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("val_f1",f1_score,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        return loss_race

class MaskModel(pl.LightningModule):
    def __init__(self, backbone,embed_size ,cfg ):
        super().__init__()
        self.cfg=cfg
        self.backbone=backbone
        self.mask_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_mask))   
        self.accuracy=MulticlassAccuracy(num_classes=cfg.class_Infor.num_mask)
        self.f1=MulticlassF1Score(num_classes=cfg.class_Infor.num_mask)

        
    def forward(self,x):
        x=self.backbone(x)
        x_mask=self.mask_head(x)
        return x_mask
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-6, weight_decay=1e-4
        )
    
    def training_step(self, batch, batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch      
        out_mask=self(x)
        loss_mask=F.cross_entropy(out_mask,target_mask)     
        self.log("train_loss", loss_mask, prog_bar=True, on_step=True)
        return loss_mask
    
    def validation_step(self,batch,batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        out_mask=self(x)
        loss_mask=F.cross_entropy(out_mask,target_mask)
        self.log("val_loss", loss_mask, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        acc=self.accuracy(out_mask,target_mask)
        f1_score=self.f1(out_mask,target_mask)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("val_f1",f1_score,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        return loss_mask

class SkinModel(pl.LightningModule):
    def __init__(self, backbone,embed_size ,cfg ):
        super().__init__()
        self.cfg=cfg
        self.backbone=backbone
        self.skin_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_skin))   
        self.accuracy=MulticlassAccuracy(num_classes=cfg.class_Infor.num_skin)
        self.f1=MulticlassF1Score(num_classes=cfg.class_Infor.num_skin)
 
    def forward(self,x):
        x=self.backbone(x)
        x_skin=self.skin_head(x)
        return x_skin
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-6, weight_decay=1e-4
        )
    
    def training_step(self, batch, batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch      
        out_skin=self(x)
        loss_skin=F.cross_entropy(out_skin,target_skin)     
        self.log("train_loss", loss_skin, prog_bar=True, on_step=True)
        return loss_skin
    
    def validation_step(self,batch,batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        out_skin=self(x)
        loss_skin=F.cross_entropy(out_skin,target_skin)
        self.log("val_loss", loss_skin, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        acc=self.accuracy(out_skin,target_skin)
        f1_score=self.f1(out_skin,target_skin)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("val_f1",f1_score,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        return loss_skin