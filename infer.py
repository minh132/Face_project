import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml
import argparse
from dataset import FaceDataset
from easydict import EasyDict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms as T
from model import Openclip,FaceModel,backbone_timm,AgeModel,GenderModel,MaskModel,RaceModel,SkinModel,EmoModel
import pandas as pd
import cv2 
import os
import json
from retinaface import RetinaFace
import numpy as np
from PIL import Image
AGE= {
        0 :'20-30s',
        1 :'40-50s',
        2 : 'Kid',
        3 : 'Senior',
        4 :'Baby',
        5 : 'Teenager'
}

SKIN=  {
    0 :'mid-light',
    1 :'light',
    2 :'mid-dark',
    3 :'dark'
}
MASK= {
    0 :'unmasked',
    1 : 'masked'
}
GENDER =  {
    0 :'Male',
    1 :'Female'
}
RACE= {
    0 :'Caucasian',
    1 :'Mongoloid',
    2 :'Negroid'
} 
EMOTION = {
        0 :'Neutral',
        1 :'Happiness',
        2 :'Anger',
        3 : 'Surprise',
        4 : 'Fear',
        5 : 'Sadness',
        6 : 'Disgust'
}

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
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    return train_dataloader, val_dataloader

if __name__ == '__main__':  
    args = get_args()
    json_file_path = "/root/code/exp/faceany/public_test_and_submission_guidelines/file_name_to_image_id.json"
    with open(json_file_path, "r") as json_file:
        map_id = json.load(json_file)  
    with open(args.cfg_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = EasyDict(cfg)
    transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])

        ])
    logger = TensorBoardLogger("lightning_logs", name=cfg.model.model_name)
    early_stop = EarlyStopping(monitor="val_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    # backbone,embed_size=Openclip(name=cfg.model.model_name,pretrained=cfg.model.pretrained_name)
    backbone,embed_size=backbone_timm(cfg.model.model_name)
    backbone_age,embed_size_age=backbone_timm(cfg.model.model_age)
    backbone_gender,embed_size_gender=backbone_timm(cfg.model.model_gender)
    model_age=AgeModel(backbone_age,embed_size_age,cfg)
    model_emo=EmoModel(backbone,embed_size,cfg)
    model_gender=GenderModel(backbone_gender,embed_size_gender,cfg)
    model_mask=MaskModel(backbone,embed_size,cfg)
    model_race=RaceModel(backbone,embed_size,cfg)
    model_skin=SkinModel(backbone,embed_size,cfg)

    ckpt_age='model_age/convnextv2_base.fcmae_ft_in22k_in1k/version_0/checkpoints/epoch=14-step=4080.ckpt'
    ckpt_emo='model_emo/convnextv2_tiny.fcmae_ft_in22k_in1k/version_0/checkpoints/epoch=23-step=4896.ckpt'
    ckpt_gender='model_age/convnextv2_base.fcmae_ft_in22k_in1k/version_1/checkpoints/epoch=6-step=2849.ckpt'
    ckpt_mask='model_mask/convnextv2_tiny.fcmae_ft_in22k_in1k/version_0/checkpoints/epoch=15-step=3264.ckpt'
    ckpt_race='model_race/convnextv2_tiny.fcmae_ft_in22k_in1k/version_0/checkpoints/epoch=21-step=4488.ckpt'
    ckpt_skin='model_skin/convnextv2_tiny.fcmae_ft_in22k_in1k/version_0/checkpoints/epoch=27-step=5712.ckpt'
    model_age.load_state_dict(torch.load(ckpt_age)['state_dict'])
    model_mask.load_state_dict(torch.load(ckpt_mask)['state_dict'])
    model_emo.load_state_dict(torch.load(ckpt_emo)['state_dict'])
    model_race.load_state_dict(torch.load(ckpt_race)['state_dict'])
    model_gender.load_state_dict(torch.load(ckpt_gender)['state_dict'])
    model_skin.load_state_dict(torch.load(ckpt_skin)['state_dict'])
    # inference
    # model=FaceModel.load_from_checkpoint('/root/code/exp/faceany/lightning_logs/ViT-B-16/version_0/checkpoints/epoch=199-step=20400.ckpt',backbone,embed_size,cfg)


    output_df_path = "answer_submit.csv"
    # create pandas with output_df_path
    output_df = pd.DataFrame(columns=['file_name','bbox','image_id', 'race', 'age', 'emotion', 'gender', 'skintone', 'masked'])

    img_root = '/root/code/exp/faceany/public_test'
    test_df = pd.read_csv('/root/code/exp/faceany/public_test_and_submission_guidelines/answer.csv')

    for file in os.listdir(img_root):
      
        img_path = os.path.join(img_root, file)
        img = Image.open(img_path)
        faces = RetinaFace.detect_faces(img_path)
        # max_area = 0
        # max_face = None
        # for (x,y,w,h) in faces:
        #     ## thử với face có diện tích lớn nhất
        #     ## do detect ra nhiều face nên chọn face có diện tích lớn nhất
        #     current_area = w * h
        #     if current_area > max_area:
        #         max_area = current_area
        #         max_face = (x, y, w, h)

        if isinstance(faces, dict): 
            try:
                for face,face_info in faces.items():
                    area = face_info['facial_area']

                    x, y, x2, y2 = area
                    # img = image[y:y2,x:x2]
                    
                    
                    cropped_image = img.crop((x,y,x2,y2))
                    
                    img = transform(cropped_image)
                    img = torch.unsqueeze(img, 0)
                    # output
                    with torch.no_grad():
                        age = model_age(img)
                        mask = model_mask(img)
                        gender = model_gender(img)
                        skin = model_skin(img)
                        race = model_race(img)
                        emo = model_emo(img)
                    print(GENDER[int(torch.argmax(gender))])
                    print(AGE[int(torch.argmax(age))])
                    output_df = output_df._append({
                    'file_name': file,
                    'bbox': f"[{x}, {y}, {x2-x}, {y2-y}]",
                    'image_id': map_id[file],
                    
                    'race': RACE[int(torch.argmax(race))],
                    'age': AGE[int(torch.argmax(age))],
                    'emotion': EMOTION[int(torch.argmax(emo))],
                    'gender': GENDER[int(torch.argmax(gender))],
                    'skintone': SKIN[int(torch.argmax(skin))],
                    'masked': MASK[int(torch.argmax(mask))]
                }, ignore_index=True)
            except Exception as e:
                print(f"An error occurred: {e}")
        # break

    # Save output_df to CSV
    output_df.to_csv(output_df_path, index=False)