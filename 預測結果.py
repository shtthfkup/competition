import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model, load_model
#working root
my_working_root = "/Users/zhengzhiyun/Desktop/code/py/competition"

# 讀取資料表
df = pd.read_csv(my_working_root+"/test_datalist_public+private.csv")
# PPD 
df['PPD'] = df['PPD'].fillna(-1)
print("'PPD' na number =", df['PPD'].isna().sum())

# Voice handicap index - 10
print(df['Voice handicap index - 10'].describe())
df.loc[df['Voice handicap index - 10'].isna(), 'Voice handicap index - 10'] = 23  # 缺值補23(中位數)
print("'Voice handicap index - 10' na number =", df['Voice handicap index - 10'].isna().sum())

# one-hot
categorical_columns = ['Sex', 'Smoking', 'Diurnal pattern', 'Onset of dysphonia ', 'Occupational vocal demand']
df_trans = pd.get_dummies(df, columns=categorical_columns)

# Normalization
df_trans['Age'] = df['Age'] / 100
df_trans['PPD'] = df['PPD'] / 2
df_trans['Drinking'] = df['Drinking'] / 2
df_trans['frequency'] = df['frequency'] / 3
df_trans['Noise at work'] = (df['Noise at work'] - 1) / 2
df_trans['Occupational vocal demand'] = (4 - df['Occupational vocal demand']) / 3
df_trans['Voice handicap index - 10'] = df['Voice handicap index - 10'] / 40
df_trans.describe()
#add
df_trans['Lumping*Occupational vocal demand'] = df['Lumping']*df['Occupational vocal demand']
df_trans['Fatigue+Occupational vocal demand'] = (df['Occupational vocal demand']+df['Fatigue'])/2
df_trans['Smoking*PPD'] = df['Smoking']*df['PPD']
df_trans['Drinking*Frequency'] = df['Drinking']*df['frequency']


# 特徵欄位
feature_columns = ['Age', 'Narrow pitch range', 'Decreased volume', 'Fatigue',
      'Dryness', 'Lumping', 'heartburn', 'Choking', 'Eye dryness', 'PND',
      'PPD', 'Drinking', 'frequency', 'Noise at work', 'Diabetes',
      'Hypertension', 'CAD', 'Head and Neck Cancer', 'Head injury', 'CVA',
      'Voice handicap index - 10', 'Sex_1', 'Sex_2', 'Smoking_0', 'Smoking_1',
      'Smoking_2', 'Smoking_3', 'Diurnal pattern_1', 'Diurnal pattern_2',
      'Diurnal pattern_3', 'Diurnal pattern_4', 'Onset of dysphonia _1',
      'Onset of dysphonia _2', 'Onset of dysphonia _3',
      'Onset of dysphonia _4', 'Onset of dysphonia _5',
      'Occupational vocal demand_1', 'Occupational vocal demand_2',
      'Occupational vocal demand_3', 'Occupational vocal demand_4',
      'Occupational vocal demand','Lumping*Occupational vocal demand',
      'Fatigue+Occupational vocal demand','Smoking*PPD','Drinking*Frequency']
# 目標欄位
target_columns = ['Disease category_1', 'Disease category_2', 'Disease category_3', 'Disease category_4', 'Disease category_5']


# 讀取模型存檔
h5_file_path = my_working_root+"/model/feat45_date_time/epoch_end.h5"  # <-- 確認模型檔案存放的路徑
model = load_model(h5_file_path, compile=False)
# 取得預測結果
y_pred = model.predict(df_trans.loc[:, feature_columns])
# 轉換為 疾病類別(1-5)
disease_category = np.argmax(y_pred, axis=1) + 1 
#submission
df_submit = pd.DataFrame()
df_submit['ID'] = df['ID']
df_submit['Category'] = disease_category

# 儲存為 .csv
csv_path = my_working_root
os.makedirs(csv_path, exist_ok=True)
df_submit.to_csv(csv_path+'/output.csv', header=False, index=False, encoding='utf-8')  # 提交的csv格式不含 index 和 欄位!!