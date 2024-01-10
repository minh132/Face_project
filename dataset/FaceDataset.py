from torch.utils.data import Dataset
import torch
from torchvision import transforms as T
import pandas as pd
import os
from PIL import Image
import json 
import numpy as np 

AGE= {
        '20-30s':0,
        '40-50s':1,
        'Kid':2,
        'Senior':3,
        'Baby':4,
        'Teenager':5
}

SKIN=  {
    'mid-light':0,
    'light':1,
    'mid-dark':2,
    'dark':3
}
MASK= {
    'unmasked':0,
    'masked':1
}
GENDER =  {
    'Male':0,
    'Female':1
}
RACE= {
    'Caucasian':0,
    'Mongoloid':1,
    'Negroid':2
} 
EMOTION = {
        'Neutral':0,
        'Happiness':1,
        'Anger':2,
        'Surprise':3,
        'Fear':4,
        'Sadness':5,
        'Disgust':6
}

class FaceDataset(Dataset):
    def __init__(self, csv_anno,img_dir):
        super(FaceDataset, self).__init__()
        self.anno=pd.read_csv(csv_anno)
        self.img_dir=img_dir
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.transforms.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
    def __getitem__(self,index):
        anno=self.anno.iloc[index]
        img_path=os.path.join(self.img_dir,anno['file_name'])
        img=Image.open(img_path).convert('RGB')
        x,y,w,h=map(int,json.loads(anno['bbox']))
        image_array = np.array(img)
        cropped_image = image_array[y:y+h, x:x+w]
        cropped_image = Image.fromarray(cropped_image)
        cropped_image = self.transform(cropped_image)
        age=AGE[anno['age']]
        race=RACE[anno['race']]
        masked=MASK[anno['masked']]
        skin=SKIN[anno['skintone']]
        emotion=EMOTION[anno['emotion']]
        gender=GENDER[anno['gender']]
        return cropped_image,age,race,masked,skin,emotion,gender
    def __len__(self):
        return len(self.anno)

if __name__=="__main__":
    ds=FaceDataset(csv_anno='/root/code/exp/faceany/label/train_data.csv',img_dir='/root/code/exp/faceany/data')
    print(ds[1])


    