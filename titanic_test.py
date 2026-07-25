#逻辑回归是
'''   分类    '''
#问题

#其实说是回归也可以，除非限定离散的一定叫分类

#处理数据和独热编码
'''
独热编码：
有几种备选可能状态就把编码的位数设置为几,对于每条实际数据,只有符合数据情况的那一位设计成1'''

import pandas as pd
import torch as tch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#重要代码

#1. pandas
'''
1.打印所有列 pd.set_options('display.max_colums' , None)
2.打开文件,只读 df = pd.read_cvs(r"<path>")
3.删除列 df.drop(column = [……])
4.删除带有空缺的列行(na = Not Avaliable) df.dropna(subset = [……]) subset指定列
5.独热编码(get_dummies是函数,不是方法,所以是传参数的)
    df = pd.get_dummies(df , column = [] , dtype = ?)
6.head(num) 返回前num行

'''


#2. Dataset
'''对数据集的抽象，需要处理的数据都放在Dataset实例里，要自己实现__len__和__getitem__'''

#实现
class Titanic_Dataset(Dataset) :

    def __init__(self , filepath):
        self.filepath = filepath
        self.mean = {
            "Pclass": 2.236695,
            "Age": 29.699118,
            "SibSp": 0.512605,
            "Parch": 0.431373,
            "Fare": 34.694514,
            "Sex_female": 0.365546,
            "Sex_male": 0.634454,
            "Embarked_C": 0.182073,
            "Embarked_Q": 0.039216,
            "Embarked_S": 0.775910
        }
        self.std = {
            "Pclass": 0.838250,
            "Age": 14.526497,
            "SibSp": 0.929783,
            "Parch": 0.853289,
            "Fare": 52.918930,
            "Sex_female": 0.481921,
            "Sex_male": 0.481921,
            "Embarked_C": 0.386175,
            "Embarked_Q": 0.194244,
            "Embarked_S": 0.417274
        }

        self.data = self._load_data()
        self.data_size = len(self.data.columns) - 1 #为什么要减一?

    def _load_data(self) :

        df = pd.read_csv(self.filepath)
        #df本质上是一个以列名为key的字典
        df.drop(columns = ["PassengerId", "Name", "Ticket", "Cabin"])
        df.dropna(subset=["Age"])
        df = pd.get_dummies(df , columns=["Sex" , "Embarked"] , dtype=int)

        #标准化: 先减去均值再除以标准差得到均值为0,标准差为1
        #目的是将权重调整到同一个水准上,方便学习率调节
        base_feature = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        for i in range(0 , len(base_feature)) :
            df[base_feature[i]] = (df[base_feature[i]] - self.mean[base_feature[i]]) / self.std[base_feature[i]]

        return df
        
    def __len__(self) :
        return len(self.data)

    #Py的类就是用纯C写类的那种架构

    #DataFrame 即 df , 是一个一个以列名为键，以 Series 为值的字典，
    # 并且这些 Series 共享同一个行索引（Index）Series 本身是一个带有索引的一维数组。
    # 这样一个结构的索引方式很多可以用PY风格的[key][idx]索引, 可以用两次iloc[r , c]定位
    #也可以用loc[key , col]定位 , 也可以切片

    #自己想: __getitem__(idx) 需要能根据样本的index返回具体的样本。

    def __getitem__(self, index):
        '''t_res = tch.tensor((1 , self.__len__()) , dtype = float)

        for val in self.data.values() :
            好吧……'''
        
        features = self.data.drop(columns=["Survived"]).iloc[index].values
        lable = self.data["Survived"].iloc[index]

        return tch.tensor(features , dtype=tch.float32) , tch.tensor(lable , dtype=tch.float32)
            

#3. DataLoader
dataset = Titanic_Dataset("C:\\Users\\wang\\Desktop\\学习文件\\泰坦尼克\\titanic\\train.csv")

dataloader = DataLoader(dataset=dataset , batch_size= 32 , shuffle=True)
#数据集   2幂的取样   均匀打乱选取(否则按序选取)
#返回一个(input , lable)的元组   这里input lable 的形状分别是(32 , 10) (32 , (1))



#4. nn.Moudule
'''对模块进行抽象,如果一个系统由多个模块组成,全部继承这个类,可以极大的方便管理'''

#要实现的方法  __init__ forward

# 这里提一下PY的继承,py是子类中同一套init的参数列表适应继承链上所有的init,
# 继承链上的顺序取决于继承顺序,如果遇到一个节点参数对不上就会TypeError
# 因此建议"不合群"的使用组合,或者把参数打包


class LogisticRegressionModel(tch.nn.Module) :

    #计算线性部分
    def __init__(self, dim) :
        super(LogisticRegressionModel , self).__init__()
        self.linear = tch.nn.Linear(dim , 1)
        # feature 的维数     输出值个数
        # 比如在 sigmoid(w_1*x + w_2*y + b)的线性部分:
        # dim = 2(b 会自动管理?) , 输出为1表示拟合(预测)值(只有一个值,一维)
        # 用的还是features 向量

    # 向前传播,变成逻辑
    def forward(self , x) :
        return tch.sigmoid(self.linear(x))
        # 返回01逻辑值
    
        #这里linear作为一个变量为什么可以接受参数?实际上PY的类可以设计成可调用的(什么意思?)
        # linear 作为nn.Linear的实例,内部定义了一个__call__方法,实际上执行那个的是:
        # linear.__call__(x), 在这个函数内部调用nn.Linear的forward进行计算
        # 由于我们自己定义的forward所在的类为nn.Linear的子类,所以不会递归
        # 至于PY的动态绑定,PY完全没有声明类型,用哪个重写版本取决于变量到底是谁


#总结整个处理流程:
# 1.创建数据集(pandas处理)
# 求出平均/标准差(一定有专门负责的方法) 导入数据(pandas) 清洗数据 独热编码 标准化 
# 实现__len__,__getitem__ 
# 2.建立MGD实例DataLoader , 用数据集 , 大小 , 是否洗牌初始化
# 3.建立模型,实现计算,一个模型要包含线性部分和激活函数部分,对应__init__和forward

#DataSet —————— DataLoader —————— Module
#对应
#手动生成数据    一次性处理全部    计算outputs = inputs @ w + b

# 怎么用在哪里用BCE返回倒数和梯度下降?哪里使用学习率?
# 在过程部分


# 5.优化器Optimizer

'''
伪代码:
## 定义优化器，传入模型的参数，并且设置固定的学习率
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(epochs):
    for features,labels in DataLoader(myDataset, batch_size=256):
        optimizer.zero_grad() ##优化器清理管理参数的梯度值。
        loss = forward_and_compute_loss(features, labels) ##前向传播并计算loss
        loss.backward() ##loss反向传播,每个参数计算梯度
        optimizer.step() ##优化器进行参数更新
'''








    



