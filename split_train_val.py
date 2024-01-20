import pandas as pd
from sklearn.model_selection import train_test_split


csv_file_path = '/root/code/exp/faceany/label/merge_label.csv'
df = pd.read_csv(csv_file_path)
stratify_columns = ['skintone','gender']


# Chia tập train và tập validation với giữ nguyên phân phối của các cột
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df[stratify_columns])




train_df.to_csv('/root/code/exp/faceany/label/train_data.csv', index=False)
val_df.to_csv('/root/code/exp/faceany/label/val_data.csv', index=False)
