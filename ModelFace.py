import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score,MulticlassAccuracy,MulticlassAveragePrecision
from loss import *
from optimizer import SAM 
def calculate_samples_per_class_tensor(batch_target_tensor, num_classes):
    unique_classes, counts = torch.unique(batch_target_tensor, return_counts=True)
    
    samples_per_class = torch.zeros(num_classes, dtype=torch.int, device=batch_target_tensor.device)
    samples_per_class[unique_classes] = counts.int()  
    
    return samples_per_class.tolist()

class FaceModel(pl.LightningModule):
    def __init__(self, backbone,embed_size ,cfg ):
        super().__init__()
        self.cfg=cfg
        self.backbone=backbone
        self.age_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_age))
        self.emo_head = nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256,cfg.class_Infor.num_emo ))
        self.skin_head= nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_skin))
        self.mask_head=nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_mask))
        self.gender_head=nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_gender))
        self.race_head=nn.Sequential(nn.Linear(embed_size, 256), nn.ReLU(), nn.Linear(256, cfg.class_Infor.num_race))
        self.accuracy=MulticlassAccuracy(num_classes=cfg.class_Infor.num_emo)
        self.f1=MulticlassF1Score(num_classes=cfg.class_Infor.num_emo)

        
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
        SGD= torch.optim.SGD(
            self.parameters(), lr=1e-6, weight_decay=1e-4
        )
        optim = SAM(self.parameters(), SGD, lr=1e-6, momentum=0.9,weight_decay=1e-4)
        return optim
    
    def training_step(self, batch, batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        samples_age_per_cls = calculate_samples_per_class_tensor(target_age,self.cfg.class_Infor.num_age)
        samples_emo_per_cls = calculate_samples_per_class_tensor(target_emo,self.cfg.class_Infor.num_emo)
        samples_skin_per_cls = calculate_samples_per_class_tensor(target_skin,self.cfg.class_Infor.num_skin)
        samples_mask_per_cls = calculate_samples_per_class_tensor(target_mask,self.cfg.class_Infor.num_mask)
        samples_gender_per_cls = calculate_samples_per_class_tensor(target_gender,self.cfg.class_Infor.num_gender)
        samples_race_per_cls=calculate_samples_per_class_tensor(target_race,self.cfg.class_Infor.num_race)
        out_age,out_emo,out_skin,out_mask,out_gender,out_race=self(x)
        ce_loss_age=F.cross_entropy(out_age,target_age)
        cb_loss_age = CB_loss(target_age, out_age, samples_age_per_cls,self.cfg.class_Infor.num_age, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_age=ce_loss_age+cb_loss_age
        ce_loss_emo=F.cross_entropy(out_emo,target_emo)
        cb_loss_emo = CB_loss(target_emo, out_emo, samples_emo_per_cls,self.cfg.class_Infor.num_emo, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_emo=ce_loss_emo+cb_loss_emo
        ce_loss_skin=F.cross_entropy(out_skin,target_skin)
        cb_loss_skin = CB_loss(target_skin, out_skin, samples_skin_per_cls,self.cfg.class_Infor.num_skin, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_skin=ce_loss_skin+cb_loss_skin
        ce_loss_mask=F.cross_entropy(out_mask,target_mask)
        cb_loss_mask = CB_loss(target_mask, out_mask, samples_mask_per_cls,self.cfg.class_Infor.num_mask, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_mask=ce_loss_mask+cb_loss_mask
        ce_loss_gender=F.cross_entropy(out_gender,target_gender)
        cb_loss_gender=CB_loss(target_gender, out_gender, samples_gender_per_cls,self.cfg.class_Infor.num_gender, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_gender=ce_loss_gender+cb_loss_gender
        ce_loss_race=F.cross_entropy(out_race,target_race)
        cb_loss_race=CB_loss(target_race, out_race, samples_race_per_cls,self.cfg.class_Infor.num_race, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_race=ce_loss_race+cb_loss_race
        
        loss=loss_age*0.2+loss_emo*0.2+loss_skin*0.2+loss_mask*0.1+loss_gender*0.1+loss_race*0.2
        self.log("train_loss", loss_emo, prog_bar=True, on_step=True)
        return loss
    def validation_step(self,batch,batch_idx):
        x,target_age,target_race,target_mask,target_skin,target_emo,target_gender=batch
        samples_age_per_cls = calculate_samples_per_class_tensor(target_age,self.cfg.class_Infor.num_age)
        samples_emo_per_cls = calculate_samples_per_class_tensor(target_emo,self.cfg.class_Infor.num_emo)
        samples_skin_per_cls = calculate_samples_per_class_tensor(target_skin,self.cfg.class_Infor.num_skin)
        samples_mask_per_cls = calculate_samples_per_class_tensor(target_mask,self.cfg.class_Infor.num_mask)
        samples_gender_per_cls = calculate_samples_per_class_tensor(target_gender,self.cfg.class_Infor.num_gender)
        samples_race_per_cls=calculate_samples_per_class_tensor(target_race,self.cfg.class_Infor.num_race)
        out_age,out_emo,out_skin,out_mask,out_gender,out_race=self(x)
        ce_loss_age=F.cross_entropy(out_age,target_age)
        cb_loss_age = CB_loss(target_age, out_age, samples_age_per_cls,self.cfg.class_Infor.num_age, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_age=ce_loss_age+cb_loss_age
        ce_loss_emo=F.cross_entropy(out_emo,target_emo)
        cb_loss_emo = CB_loss(target_emo, out_emo, samples_emo_per_cls,self.cfg.class_Infor.num_emo, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_emo=ce_loss_emo+cb_loss_emo
        ce_loss_skin=F.cross_entropy(out_skin,target_skin)
        cb_loss_skin = CB_loss(target_skin, out_skin, samples_skin_per_cls,self.cfg.class_Infor.num_skin, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_skin=ce_loss_skin+cb_loss_skin
        ce_loss_mask=F.cross_entropy(out_mask,target_mask)
        cb_loss_mask = CB_loss(target_mask, out_mask, samples_mask_per_cls,self.cfg.class_Infor.num_mask, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_mask=ce_loss_mask+cb_loss_mask
        ce_loss_gender=F.cross_entropy(out_gender,target_gender)
        cb_loss_gender=CB_loss(target_gender, out_gender, samples_gender_per_cls,self.cfg.class_Infor.num_gender, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_gender=ce_loss_gender+cb_loss_gender
        ce_loss_race=F.cross_entropy(out_race,target_race)
        cb_loss_race=CB_loss(target_race, out_race, samples_race_per_cls,self.cfg.class_Infor.num_race, loss_type="focal", beta=0.9999, gamma=2.0)
        loss_race=ce_loss_race+cb_loss_race
        
        loss=loss_age*0.2+loss_emo*0.2+loss_skin*0.2+loss_mask*0.1+loss_gender*0.1+loss_race*0.2
        self.log("val_loss", loss_emo, prog_bar=True, on_step=True)
        return loss

