import pandas as pd
from sklearn.model_selection import train_test_split


csv_file_path = '/root/code/exp/faceany/labels.csv'
df = pd.read_csv(csv_file_path)


train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)


train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
