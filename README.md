# 多模態病理嗓音競賽
## 資料處理+訓練流程.py  
1.用於訓練模型  
2.輸入為training datalist.csv  
3.輸出為模型epoch_end.h5

## 預測結果.py  
1.用於預測結果  
2.輸入為預測集test_datalist_public+private.csv  
3.輸出預測結果output.csv

## 安裝配置環境

```console
pip install numpy, pandas, tensorflow, matplotlib, scikit-learn, matplotlib
```

## 重要模塊輸出/輸入
### 資料處理+訓練流程.py
輸入 working root  
```python
my_working_root = ""
```

模型設計  
```python
def build_model(feature_num, num_classes=5):
    inputs = layers.Input(shape=(feature_num, ), name="feats")

    x = layers.Dense(64, activation="leaky_relu")(inputs)
    x = layers.Dense(32, activation="sigmoid")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
     
    logits = layers.Dense(num_classes, name="logits")(x)

    model = Model(inputs=inputs, outputs=logits, name=f"feat{feature_num}_date_time")  # <-- 為模型命名(存檔資料夾名稱)
    return model
```

資料前處理  
```python
# PPD 
df['PPD'] = df['PPD'].fillna(-1)  

# Voice handicap index - 10  (VHI-10嗓音障礙指標) 0(healthy)-40(devastating)
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

df_trans = pd.get_dummies(df_trans, columns=['Disease category'])
```

調整batch size  
```python
batch_size = 64
```

調整epoch次數  
```python
EPOCHS = 50
```

區分測試集和驗證集比例  
```python
# train/valid   80/20 ratio
train_index, valid_index = train_test_split(df_trans.index, train_size=0.8, random_state=333, stratify=df['Disease category'])
print('train_index shape =', train_index.shape)
print('valid_index shape =', valid_index.shape)
```

confusion matrix  
```python
class_names = CLASSES
plt.rcParams.update({'font.size': 8})

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
```

classification report  
```python

report = classification_report(y_true.argmax(axis=1), y_pred.argmax(axis=1), target_names=CLASSES, digits=4)
print(report)
```

UAR (Unweighted Average Recall)
```python
uar = recall_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
print("Validation UAR (Unweighted Average Recall) :", uar)
```
### 預測結果.py

讀取模型存檔  
```python
h5_file_path = my_working_root+"/model/feat45_date_time/epoch_end.h5"  # <-- 確認模型檔案存放的路徑
model = load_model(h5_file_path, compile=False)
```

取得預測結果  
```python
y_pred = model.predict(df_trans.loc[:, feature_columns])
```

儲存為 .csv  
```python
csv_path = my_working_root
os.makedirs(csv_path, exist_ok=True)
df_submit.to_csv(csv_path+'/output.csv', header=False, index=False, encoding='utf-8')
```
