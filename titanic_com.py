import pandas as pd
import torch as tch
import torch.nn as nn
from torch.utils.data import *


# ============================= 1.建立数据集 ================================

print("1. 建立数据集")

class Titanic_Dataset(Dataset) :

    def __init__(self , filepath):
        self.filepath = filepath
        self.data , self.mean , self.std , self.base_feature = self._load_data()
        self.feature_size = len(self.data.columns) - 1


    def _load_data(self) :
        df = pd.read_csv(self.filepath)
        # 不用指定r , 默认
        # 清洗数据时编写者有义务知道列的含义
        df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
        df.dropna(subset=["Age"])
        df = pd.get_dummies(df , columns=["Sex" , "Embarked"] , dtype=tch.int32)
        mean = std = {}
        base_feature = []
        # 尝试自动生成标准差和均值
        for name , col in df.items() :
            mean_val = col.mean()
            std_val = col.std()
            mean[name] = mean_val
            std[name] = std_val
            base_feature.append(name)

        #标注化

        for i in range(0 , len(base_feature)) :
            df[base_feature[i]] = (df[base_feature[i]] - mean[base_feature[i]]) / std[base_feature[i]]
        
        return df , mean , std , base_feature
    
    def __len__(self) :
        return len(self.base_feature)
    
    def __getitem__(self, index):
        lable = self.data["Survived" , index]
        features = self.data.drop(columns=["Survived"]).iloc[index].values

        return tch.tensor(features , dtype=tch.float32) , tch.tensor(lable , dtype=tch.float32)


# =========================== 2.建立数据集实例和loader =================================

print("2.建立数据集实例和loader")

full_dataset = Titanic_Dataset("C:\\Users\\wang\\Desktop\\学习文件\\泰坦尼克\\titanic\\train.csv")

total_size = len(full_dataset)
# 这里实例不是字典,读出来的是行数,不是列数(键值对个数)
train_size = int(total_size * 0.8)
validation_size = total_size - train_size
train_dataset , validation_dataset = random_split(full_dataset , [train_size , validation_size])

train_loader = DataLoader(train_dataset , batch_size = 32 , shuffle=True)
validation_loader = DataLoader(validation_dataset , batch_size=32 , shuffle=False)


# ============================= 3.建立sigmoid模型 ===================================

print("建立sigmoid模型")

class LogisticRegressionModel(nn.Module) :

    def __init__(self , dim):
        super(LogisticRegressionModel , self).__init__()
        self.linear = nn.Linear(dim , 1)

    def forward(self , x) :
        return tch.sigmoid(self.linear(x))

device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
model = LogisticRegressionModel(train_dataset.feature_size)
model.to(device = device)
model.train()

optimizer = tch.optim.SGD(model.parameters(), lr=0.1)


# =============================== 4.开始迭代 =====================================

print("4.开始迭代")

epoch = 1000

for i in range(0 , epoch) :

    correct = total_loss = step = 0
    #这里的MGD是每次随机取256条一条一条处理
    for features , lables in DataLoader(train_dataset , batch_size=256 , shuffle=True) :
        step += 1 
        features = features.to(device)
        lables = lables.to(device)
        optimizer.zero_grad()
        # 模型的forward在model(features)时调用 forward里的x就是feature
        # linear(x)的输出为二维张量(batch_size , prec预测值),所以此处使用要压缩掉0维度
        output = model(features).squeeze()

        #正确个数(acc的公式)
        correct += tch.sum((output >= 0.5) == lables)
        loss = tch.nn.functional.binary_cross_entropy(output , lables)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        #以前的w' = w - lc * grad都交给了optimizer

    #print(f'Epoch {epoch + 1}, Loss: {total_loss/step:.4f}')
    #print(f'Training Accuracy: {correct / len(train_dataset)}')


# =============================== 5.开始验证 =====================================

print("5.开始验证")

model.eval() #evaluate
with tch.no_grad() :
    correct = 0
    for features, labels in DataLoader(validation_dataset, batch_size=256):
        features = features.to("cuda")
        labels = labels.to("cuda")
        outputs = model(features).squeeze()
        correct += tch.sum(((outputs >= 0.5) == labels))
    print(f'Validation Accuracy: {correct / len(validation_dataset)}')