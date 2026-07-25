import pandas as pd
import torch as tch
from torch.utils.data import *
import torch.nn as nn
import statistics as stcs
import torch.optim as opt

# ============================== 0.建立模型 ================================

class NeuralNetwork(nn.Module) :

    def __init__(self):
        super(NeuralNetwork , self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28 * 28 , 128 , bias=True , device="cpu"),
            # 不想要梯度,在实例后面加.requires_grad_(False)
            nn.ReLU() ,

            nn.Linear(128 , 128, bias=True , device="cpu"),
            nn.ReLU() ,

            nn.Linear(128 , 64, bias=True , device="cpu") ,
            nn.ReLU() ,

            nn.Linear(64 , 10, bias=True , device="cpu")

        )
        # nn.Model可以自动管理参数,参数是模型自带的,而且默认require_grad


    def forward(self , x) :
        return self.model(x)


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
        images = []
        lables = []
        mean = []
        std = []
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
            for col_id in range(0 , len(images[0])) :
                col = [row[col_id] for row in images]
                '''mean.append(stcs.mean(images[:col_id]))
                std.append(stcs.pstdev(images[:col_id]))'''
                #这种写法是前col_id个元素不是行
                mean.append(stcs.mean(col))
                std.append(stcs.pstdev(col))
    
        return images , lables , mean , std
    

    def __getitem__(self, index):
        image = self.images[index]  # image为list
        lable = self.lables[index]
        # 示范用的是全局的均值和标准差,但我决定应该用列的均值和标准差进行
        # df = pd.read_csv(self.filepath)
        '''tensor 和 list不可相互计算: 
        TypeError: unsupported operand type(s) for -: 'Tensor' and 'list'''

        # 归一化和标准化可以用张量直接对列表处理
        image = tch.tensor(image , dtype=tch.float32) # image为tensor
        mean = tch.tensor(self.mean , dtype=tch.float32)
        std = tch.tensor(self.std , dtype=tch.float32)

        image /= 255.0
        #image = (image - mean) / std + 1e-8
        image = (image - 0.1307) / 0.3081  # 标准化上面那种单列的不行
        '''
        为什么不能逐元素标准化归一化?
        比如背景板和数字归一化和标准化的目的是使数据趋于稳定,统一不同种数据的量级
        现在来看逐元素统计:
        假如有一个冷门的背景板,在所有情况下都是白的
        一个热门像素,几乎所有情况都是黑的
        一个正常的随机的像素,又黑又白
        这导致了前两种情况的mean几乎和具体数值没有区别,std趋于0
        这种情况下不应该用数学极限去看待,而是分母直接归0
        即使正确,在归一化的相同尺度下,也会发生数值过大引发的爆炸
        所以按照元素种类一定是错误的,归一的原则是减少种类差异,而逐元素的方式反对这种差异

        但是后面的神经网络优化里的批量归一化则相反,是要照顾种类的,因为目的不一样,对数据处理是为了
        统一种类间,而批量归一是为了稳定种类内
        '''

        lable = tch.tensor(lable)

        return image , lable


    def __len__(self) :
        return len(self.images)
    

# =========================== 2.建立数据集实例和loader =================================
print("2. 建立数据集实例和loader")

batch_size = 64
learning_rate = 0.1
epoch = 13
device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
train_dataset = MNistDataset(r"C:\\Users\\wang\\Desktop\\学习文件\\handwriting\\mnist\\mnist_train_csv\\mnist_train.csv")
train_dataloader = DataLoader(train_dataset , batch_size = 64 , shuffle = True)

test_dataset = MNistDataset(r"C:\\Users\\wang\\Desktop\\学习文件\\handwriting\\mnist\\mnist_test_csv\\mnist_test.csv")
test_dataloader = DataLoader(test_dataset , batch_size = 32 , shuffle = True)


# =========================== 3.模型化建立神经网络 ==================================

print("3. 模型化建立神经网络")
model = NeuralNetwork().to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = opt.SGD(model.parameters() , lr = learning_rate)             


# =============================== 4.迭代训练 =====================================

print("4. 迭代训练")

model.train()
for iter in range(0 , epoch) :
    total_loss = 0 

    for images , lables in train_dataloader :
        optimizer.zero_grad()
        outputs = model(images).to(device)
        loss = criterion(outputs , lables).to(device)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    ava_loss = total_loss / len(train_dataloader)
    print(f"Epoch {iter+1}, Avg Loss: {total_loss/len(train_dataloader):.4f}")
# ================================ 5.测试模型 ==================================

print("5. 测试模型")

model.eval()
correct = 0 
total_test = 0

for images_t , lables_t in test_dataloader:
    images_t = images_t.to(device)
    lables_t = lables_t.to(device)
    outputs = model(images_t)

    pred = tch.argmax(outputs , dim = 1 , keepdim=False)
    # tch.max返回的是命名元组(最大值 , 索引)   tch.argmax 返回的是一维的索引张量
    # 由于索引和数字是一一对应的,所以索引就是预测结果,
    correct += (pred == lables_t).sum().item()   
    '''不是.items() 没有s'''
    # dataloader不会独热编码,提取结果是张量
    # pred , lables_t 都是张量，比较采取逐个比较的策略,所以返回的是一个true ,false的张量
    # 方法: .sum() 返回true的数量  .any()有一个true就返回true   .all()所有为true才返回true

    total_test += lables_t.size(0)  # 累加当前 batch 的实际样本数

acc = correct / total_test


print(f"accuracy: {acc}")