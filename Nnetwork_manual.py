import pandas as pd
import torch as tch
from torch.utils.data import *
import torch.nn as nn
import statistics as stcs


# ============================= 1.建立数据集 ================================

print("1. 建立数据集")

class MNistDataset(Dataset) :

    def __init__(self , filepath):
        self.filepath = filepath
        self.images , self.lables , self.mean , self.std = self._load_data()


    def _load_data(self) :
        #df = pd.read_csv(self.filepath)
        #可以考虑一下pandas的版本
        #但是和上次不同,这里不需要整理数据,也可以避免上次的关于提前知道列名的问题
        images = lables = mean = std = []
        with open(self.filepath , 'r') as f :
            next(f)
            for rows in f :
                rows = rows.rstrip("\n") # 消除换行符
                rows = rows.split(",")
                # Python也需要区分可变对象和不可变对象
                # 不可变对象有: str , tumple , int , float
                # 可变对象有 list , dict , set , tensor , 类实例的可变部分
                # 区分的意义是: 不可变对象不能原地修改,需要先复制再赋值给新变量
                # open打开法每行以","分割的字符串呈现,而str不可变,所以要row = row.func
                # 而pandas是列表打开的
                images.append([float(x) for x in rows[1:]])
                lables.append(int(rows[0]))
            for col_id in range(0 , len(images[[0]])) :
                mean.append(stcs.mean(images[:col_id]))
                std.append(stcs.pstdev(images[:col_id]))
    
        return images , lables , mean , std
    

    def __getitem__(self, index):
        image = self.images[index]
        lable = self.lables[index]
        # 示范用的是全局的均值和标准差,但我决定应该用列的均值和标准差进行
        # df = pd.read_csv(self.filepath)

        # 归一化和标准化可以用张量直接对列表处理
        image = tch.tensor(image , dtype=tch.float32)
        image /= 255.0
        image = (image - self.mean) / self.std

        lable = tch.tensor(lable)

        return image , lable


    def __len__(self) :
        return len(self.images)
    

# =========================== 2.建立数据集实例和loader =================================
batch_size = 64
device = tch.device("cpu")
train_dataset = MNistDataset(r"C:\\Users\\wang\\Desktop\\学习文件\\handwriting\\mnist\\mnist_train.csv")
train_dataloader = DataLoader(train_dataset , batch_size = 64 , shuffle = True)

test_dataset = MNistDataset(r"C:\\Users\\wang\\Desktop\\学习文件\\handwriting\\mnist\\mnist_test.csv")
test_dataloader = DataLoader(test_dataset , batch_size = 32 , shuffle = True)


# =============================== 3.手动建立神经网络 ==================================

layer_size = [28 * 28 , 128 , 128 , 64 , 10]
weights = [] 
bias = []

layer_size = zip(layer_size[:-1] , zip[1:])
for input_size , output_size in layer_size :
    wei = tch.randn(input_size , output_size , device=device , requires_grad=True) / tch.sqrt(2 / input_size)
    b = tch.zeros(1 , output_size , device = device , requires_grad=True)

    weights.append(wei)
    bias.append(b)

# 注意一层只有一套参数和一套输入,自然使用即可,这些是设定好的

# 提前生成好权重                


# ============================== 4.手动建立激活函数 ==================================

# clamp是阶段张量的方法,把数值都限制在min以上,这里等价于relu
def relu(t) :
    return tch.clamp(t , min=0)

def grad_relu(t) :
    return t > 0
 

def softmax(t) :
    e_t = tch.exp(t - tch.max(t , dim = 1 , keepdim=True))
    return e_t / e_t.sum(dim = 1 , keepdim = True)

'''注意这里softmax的实现有个小的技巧,为了防止输入的值过大,比如1000,e的1000次方就超过float的表示范围了。
解决办法是给softmax函数的分子分母同时除以本类最大值,所以dim = 1
 '''

def cross_entropy(pred, labels):
    N = pred.shape[0]
    one_hot = tch.zeros_like(pred)
    one_hot[tch.arange(N), labels] = 1  # 生成label的one-hot编码
    loss = - (one_hot * tch.log(pred + 1e-8)).sum() / N  # 计算平均loss
    return loss, one_hot


# ================================ 5.训练迭代 ==================================

epoch = 1000 
lr = 0.001
for i in range(0 , epoch) :

    total_loss = 0 
    # 这里提一下loader是如何保证结构正确的(也就是如何防止lables和image弄混):
    # loader不识别结构,在取用数据时完全按照Data_set.__getitem__()识别的结构提取
    # 但是,如果调用那么返回值的顺序就很重要,因而实现函数时返回值的顺序并一定会影响使用
    # 所以,硬编码规定是必须的,先image在lable,和下面遍历loader的顺序一致

    ### 向前传播

    # 隐藏层传播
    for images , lables in train_dataloader:

        x = images.to(device)
        y = lables.to(device)
        N = x.shape[0]
        # py 判断维度小技巧:
        # 一个[]一个维度,维数就是括号里的元素有几个
        # 纸面分析的时候维度是成对出现的,一个大括号里面的行-列是一对双维度
        # 对数随嵌套数增加而增加

        activations = [x]   # 负责记录当前输入值
        pre_activations = []

        '''<for w , b in weights , bias :> 不能这样写!'''

        for w , b in zip(weights[:-1] , bias[:-1]) :
            z = activations[-1] @ w + b
            pre_activations.append(z)
            a = relu(z)
            activations.append(a)

    
    # 输出层传播
    z_out = activations[-1] @ w[-1] + b[-1]
    pre_activations.append(z_out)
    y_pred = a_out = softmax(z_out)
    '''activations.append(a_out) 不需要这一句,方便''' 

    # 计算损失 CE
    loss , one_hot = cross_entropy(y_pred , lables)
    total_loss += loss.item() # item是tensor专属的,把单元素张量提取成浮点数


    ### 反向传播

    '''loss.backward()'''  # 不知道这样行不行?我实在不想推导了
    grads_w = [None] * len(weights)
    grads_b = [None] * len(bias)

    # 输出层
    q = (y_pred - one_hot) / N 
    # pytorch的绝大多数书写运算都提供了函数式和方法式调用
    grads_w[-1] = activations[-1].t() @ q
    grads_b[-1] = q.sum(dim = 0)

    # 隐藏层
    for i in range(len(weights) - 2 , -1 , -1) : # 注意负步长,终点-1
        q = q @ w[i + 1].t() * grad_relu(pre_activations[i])
        grads_w[i] = activations[i - 1].t() @ q
        grads_b[i] = q.sum(dim = 0)

    
    # 更新
    with tch.no_grad() :
        for i in range(0 , len(weights)) :
            weights[i] = weights[i] * lr
            bias[i] = bias[i] * lr

avg_loss = total_loss / len(train_dataloader)
print(f"Epoch {epoch+1}/{epoch}, Loss: {avg_loss:.4f}")




