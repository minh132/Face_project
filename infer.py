import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml
import argparse
from dataset import FaceDataset
from easydict import EasyDict

from torchvision import transforms as T
from ModelFace import AgeModel,GenderModel,MaskModel,RaceModel,SkinModel,EmoModel,AgeGenderModel,FaceModel
from model import backbone_timm,build_model
import pandas as pd
import cv2 
import os
import json
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import time 
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



if __name__ == '__main__':  
    args = get_args()
    json_file_path = "/root/code/exp/faceany/file_name_to_image_id.json"
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
    
    backbone,embed_size=backbone_timm(cfg.model.model_name)
    model=FaceModel(backbone,embed_size,cfg)
    ckpt_model='/root/code/exp/faceany/mode_all/convnextv2_base.fcmae_ft_in22k_in1k/version_0/checkpoints/epoch=16-step=8449.ckpt'

    model.load_state_dict(torch.load(ckpt_model)['state_dict'])
    # inference

    output_df_path = "answer_submit.csv"
    # create pandas with output_df_path
    output_df = pd.DataFrame(columns=['file_name','bbox','image_id', 'race', 'age', 'emotion', 'gender', 'skintone', 'masked'])

    img_root = '/root/code/exp/faceany/test/private_test_data'
    test_df = pd.read_csv('/root/code/exp/faceany/public_test_and_submission_guidelines/answer.csv')
    app = FaceAnalysis(name= 'buffalo_l',allowed_modules=['detection'],providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) # enable detection model only
    app.prepare(ctx_id=0, det_size=(512, 512))
    for file in os.listdir(img_root):
        
        img_path = os.path.join(img_root, file)
        img = cv2.imread(img_path)
        img = img[:,:,::-1]
        time1=time.time()
        faces = app.get(img)
        image = Image.open(img_path)



        
        try:
            for idx, face in enumerate(faces):
                x,y,x2,y2 = face.bbox
                # img = image[y:y2,x:x2]
                
                
                cropped_image = image.crop((x,y,x2,y2))
                
                img = transform(cropped_image)
                img = torch.unsqueeze(img, 0)
                # output
                with torch.no_grad():
                    age = model(img)
                    mask = model(img)
                    gender = model(img)
                    skin = model(img)
                    race = model(img)
                    emo = model(img)

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
            print(img_path)


        
    # Save output_df to CSV
    output_df.to_csv(output_df_path, index=False)