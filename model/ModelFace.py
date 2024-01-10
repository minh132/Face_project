import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceModel(pl.LightningModule):
    def __init__(self, backbone,embed_size ,cfg ):
        super().__init__()
        self.cfg=cfg
        self.backbone=backbone
        self.age_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_age))
        self.emo_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256,cfg.class_Infor.num_emo ))
        self.skin_head= nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_skin))
        self.mask_head=nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_skin))
        self.gender_head=nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_gender))
        self.race_head=nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_race))

        
    def forward(self,x):
        x=self.backbone(x)
        x_age=self.age_head(x)
        x_emo=self.emo_head(x)
        x_skin=self.skin_head(x)
        x_mask=self.mask_head(x)
        x_gender=self.gender_head(x)
        x_race=self.race_head(x)
        return x_age,x_emo,x_skin,x_mask,x_gender,x_race
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-6, weight_decay=1e-4
        )
    
    def training_step(self, batch, batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        
        out_age,out_emo,out_skin,out_mask,out_gender,out_race=self(x)
        loss_age=F.cross_entropy(out_age,target_age)
        loss_emo=F.cross_entropy(out_emo,target_emo)
        loss_skin=F.cross_entropy(out_skin,target_skin)
        loss_mask=F.cross_entropy(out_mask,target_mask)
        loss_gender=F.cross_entropy(out_gender,target_gender)
        loss_race=F.cross_entropy(out_race,target_race)
        loss=loss_age*0.2+loss_emo*0.2+loss_skin*0.2+loss_mask*0.1+loss_gender*0.1+loss_race*0.1
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss
    def validation_step(self,batch,batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        out_age,out_emo,out_skin,out_mask,out_gender,out_race=self(x)
        loss_age=F.cross_entropy(out_age,target_age)
        loss_emo=F.cross_entropy(out_emo,target_emo)
        loss_skin=F.cross_entropy(out_skin,target_skin)
        loss_mask=F.cross_entropy(out_mask,target_mask)
        loss_gender=F.cross_entropy(out_gender,target_gender)
        loss_race=F.cross_entropy(out_race,target_race)
        loss=loss_age*0.2+loss_emo*0.2+loss_skin*0.2+loss_mask*0.1+loss_gender*0.1+loss_race*0.1
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        return loss