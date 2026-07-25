import torch as tch
import numpy as np

#tensor 张量
#向量是特殊的张量
#0维张量是标量 1维是向量 2维是矩阵

t0 = tch.tensor(3.14)
t1 = tch.tensor([2.3 , 4.5])
t2 = tch.tensor([[2.4 , 4.8 , 0.9] , [1.2 , 1.3 , 4.5]]) #最外层要括起来
#可以用numpy创建
arr = np.array([1.2,3])
t_np = tch.tensor(arr)

'''整数型 torch.uint8、torch.int32、torch.int64。其中torch.int64为默认的整数类型。

浮点型 torch.float16、torch.bfloat16、 torch.float32、torch.float64，其中torch.float32为默认的浮点数据类型。

布尔型 torch.bool

在PyTorch里使用最广泛的就是浮点型tensor。其中torch.float32称为全精度，torch.float16/torch.bfloat16称为半精度。一般情况下模型的训练是在全精度下进行的。如果采用混合精度训练的话，会在某些计算过程中采用半精度计算。混合精度计算会节省显存占用以及提升训练速度。

PyTorch里没有字符串类型，因为Tensor主要关注于数值计算，并不需要支持字符串类型。'''

#支持位运算
ss = ~(2 & 4 ^ 8 | 32)

#注意原生组的布尔索引仍然是按照数值型处理的，这里只有tensor可以这么用

t_boolidx = tch.tensor([2,3,4,5,6,7])

mask = t_boolidx > 2
print(mask)


t_boolidx3 = t_boolidx[mask]

print(t_boolidx3)

'''PyTorch里的数据类型，主要为：

整数型 torch.uint8、torch.int32、torch.int64。其中torch.int64为默认的整数类型。

浮点型 torch.float16、torch.bfloat16、 torch.float32、torch.float64，其中torch.float32为默认的浮点数据类型。

布尔型 torch.bool'''

t_s = tch.tensor([2.2 , 2.2] , dtype = int)
t_s2 = tch.tensor([2.2 , 2.2] , dtype = tch.int32)

print(t_s)

#tensor的属性shape(几乘几) device dtype

#复习
shape = (2 , 3)
t_rand = tch.rand(shape) #随机生成一个2 * 3张量,取值在0到1的闭区间内

rt0 = tch.tensor(6.28)
rt1 = tch.tensor([2.2 , 4.4])
rt2 = tch.tensor([[4.5 , 9.0] , [3.3 , 0.9]])
arr2 = np.array([0.9,0.8])
rt_arr = tch.tensor(arr2)

mask2 =  rt2 > 0.5
t_boolidx4 = rt2[mask2]

shape2 = (1 , 2)
t_rd2 = tch.rand(shape2)


print(t_rand)


'''shape = (2,3)
rand_tensor = torch.rand(shape) # 生成一个从[0,1]均匀抽样的tensor。
randn_tensor = torch.randn(shape) # 生成一个从标准正态分布抽样的tensor。
ones_tensor = torch.ones(shape) #生成一个值全为1的tensor。
zeros_tensor = torch.zeros(shape) # 生成一个值全为0的tensor。
twos_tensor = torch.full(shape, 2) #  生成一个值全为2的tensor。'''

#转换(改变形状)和转置(交换维度)
shape3 = (4 , 4)

t_old = tch.randn(shape3)
t_reshaped = t_old.reshape(2 , 8)
t_permuted = t_old.permute(1 , 0) #注意顺序,注意维度数量和参数一致

print(f"old: {t_old} ,\n reshaped: {t_reshaped} ,\n permuted: {t_permuted}")


#维度扩展
'''
x = torch.tensor([[1,2,3],[4,5,6]])   2 * 3
#扩展第0维                           1 * 2 * 3
x_0 = x.unsqueeze(0)
print(x_0.shape,x_0)
#扩展第1维                           2 * 1 * 3
x_1 = x.unsqueeze(1)
print(x_1.shape,x_1)
#扩展第2维                           2 * 3 * 1
x_2 = x.unsqueeze(2)
print(x_2.shape,x_2)

数字代表数字对应的括号级数可以放下几个元素
扩展就是开新维度设成1'''


t_squeeze = tch.tensor([[1,2,3] , [4,5,6]])
t_unsqu_0 = t_squeeze.unsqueeze(1)
print(t_unsqu_0)
t_unsqu_0 = t_unsqu_0.permute(2 , 1 , 0)
print(t_unsqu_0)

#维度压缩：只压缩维度数为1的
t_one = tch.ones(1,1,3)
t_squed1 = t_one.squeeze(dim = 0)#dim: dimention
t_squed2 = t_one.squeeze()

print(f"{t_squed1} , {t_squed2}")

#运算：符合逐项四则运算 @为矩阵乘法

#统计函数
'''
一个tensor中包含多个元素，对这些元素可以进行统计操作。
比如通过tensor.sum()求和，
通过tensor.mean()求均值，
通过tensor.std()求标准差，
通过tensor.min()求最小值等。

'''
#求均值 并指定维度意味着消灭维度,结果是该维度内所有数据对应相加除以维度数
print("/////////////////////////////////")
t_unmeaned = tch.tensor([[1,2,3] , [2,4,6]] , dtype = float)
t_mean = t_unmeaned.mean(dim = 0)
t_kmean = t_unmeaned.mean(dim = 0 , keepdim = True)
print(f"{t_unmeaned} ,\n {t_mean} ,\n {t_kmean}")

#切片索引功能和py类似
print("/////////////////////////////////")
t_ori = tch.tensor([[1,2,3] , [2,4,6]] , dtype = float)
print(f"{t_ori[0,1]} ,\n {t_ori[: , :]} ,\n {t_ori[: , 1]} ,\n {t_ori[0 , :]} ,\n {t_ori[: , -1]}")



#广播机制
'''理论上来说，技术按需要形状完全一致，即维数和维度数一致'''
'''假如我们有一个tensor:t1。t1的shape为(3,2),
我们想给t1的每个元素都加上1。
此时我们不必构造一个shape为(3,2),元素全为1的tensor再进行相加。
我们可以直接写 t1 +1,PyTorch内部会虚拟扩展出一个形状为(3,2)的tensor,再和t1相加。
这种机制，就是广播机制'''
print("/////////////////////////////////")
t1 = tch.full((3,2) , 2)
t2 = tch.ones(2)

t3 = t1 + t2 
print(f"{t1} ,\n {t2} ,\n {t3}")

'''
1.先检查两个tensor的形状，如果它们的维度个数不同，在短的那个前边补1，使它们的维度个数相同.
2.在维度值为1的维度上，通过虚拟复制，让两个tensor的维度值相等。 对于上一步维度对齐后的例子分别
3.按位计算'''

t4 = tch.full((3,1) , 2)
t5 = tch.full((1,3) , 2)

print(f"{t4 + t5}")
'''
结果是复制扩展：
[2,2,2]    [[2,2,2],[2,2,2],[2,2,2]]'''

#GPU加速
device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


'''import torch
import time

# 确保 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 生成随机矩阵
size = 10000  # 矩阵大小
A_cpu = torch.rand(size, size) # 默认在CPU上创建tensor
B_cpu = torch.rand(size, size)

start_cpu = time.time()
C_cpu = torch.mm(A_cpu, B_cpu)  # 矩阵乘法
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# 在 GPU 上计算
A_gpu = A_cpu.to(device) # 将tensor转移到GPU上
B_gpu = B_cpu.to(device)

start_gpu = time.time()
C_gpu = torch.mm(A_gpu, B_gpu)
torch.cuda.synchronize()  # 确保GPU计算完成
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

print(f"CPU time: {cpu_time:.6f} sec")
if torch.cuda.is_available():
    print(f"GPU time: {gpu_time:.6f} sec")
else:
    print("GPU not available, skipping GPU test.")'''

#目前我的电脑没有cuda



#自动算梯度

x = tch.tensor(1.0 , requires_grad = True)
y = tch.tensor(2.0 , requires_grad = False)

v = 3 * x + 4 * y
u = tch.square(v)
z = tch.log(u)

z.backward()

print(f"{x.grad} , {y.grad}")
#0.5454545021057129 , None




