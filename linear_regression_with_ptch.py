import torch as tch
from torch.utils.tensorboard import SummaryWriter

lr = 0.03
num_iteration = 10000
device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
#只用cpu

#注意这里要用参数向量表示权重w

inputs = tch.rand(100 , 3)
wei = tch.tensor([[1.1] , [2.2] , [3.3]]) #内层的中括号不能去掉,这才是(3,1)维度数
bias = tch.tensor(4.4)  #可以设成标量
target = inputs @ wei + bias + 0.1 * tch.rand(100 , 1) #听说加了误差好?
#注意这里是target,生成的是数据,不是loss/MSE

writer = SummaryWriter(log_dir = "logs")

'''1.误以为 outputs - targets 会先把 outputs 的所有元素求和，
再减去 targets 的总和。
实际上 PyTorch 的 - 运算符对张量是逐元素操作，所以这里是：
outputs - targets = [o1-t1, o2-t2, ..., o10-t10]
同样square遵循广播原则, mean符合降维规则
2.误以为wei,bia 和 w,b设置两次是多此一举,实际上前者是用来造数据的,没有就变成无监督学习了
3.张量是对实例进行追踪的,也就是说如果创建了新的实例将不会提供grad所以####处必须原地改变'''

#这里要再设一次
w = tch.rand((3 , 1) , device = device , requires_grad = True)
b = tch.rand((1,) , device = device , requires_grad = True)

for i in range(0 , num_iteration) :

    output = inputs @ w + b
    loss = tch.mean(tch.square(output - target))
    #通过矩阵运算一次性处理一组数据
    #print(f"loss: {loss.item()} \n")
    #m * n @ n * p ===>  m * p

    writer.add_scalar("loss/train", loss.item(), i)
    #                  名字/标签       损失值     步数

    #回溯得到结果，关闭跟踪后将无法获得梯度
    loss.backward()
    
    with tch.no_grad() :
        w -= lr * w.grad
        b -= lr * b.grad#####不可以w' = w ……    但是也不要忘记更新(-=)

    w.grad.zero_()
    b.grad.zero_()

    #pytorch里多次回溯，后算出来的结果会叠加在之前的结果上，所以必须清理


print(f"w的结果: {w} , b的结果: {b}")


#潜在问题，不同权重的影响力不同，有的梯度非常大，有的非常小，导致共用学习率会导致：
'''
1.lr大了，梯度大的会偏离
2.lr小了，梯度小的训练慢
'''

#实验归一化,顺带复习

#import torch as tch

dev = tch.device("cuda" if tch.cuda.is_available() else "cpu")
new_inputs = tch.tensor([[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]] , dtype=float , device=dev)
lables = tch.tensor([[19], [31], [14], [15], [43]] , dtype=float , device=dev)

new_w = tch.rand((2 , 1) , requires_grad=True , dtype=float,device = dev)
new_b = tch.rand((1 , ) , requires_grad=True ,dtype=float, device = dev)

epoch = 1000
new_lr = 0.5

new_inputs /= tch.tensor([4,3000] , dtype=float ,device=dev)

#for (int i = 0 ; i <= epoch ; ++i) {} 魔怔了
for i in range(0 , epoch) :

    new_output = new_inputs @ new_w + new_b
    new_loss = tch.mean(tch.square(new_output - lables))
    #print(new_loss)

    new_loss.backward() 

    with tch.no_grad() :
        new_w -= new_lr * new_w.grad
        new_b -= new_lr * new_b.grad
        new_w.grad.zero_()
        new_b.grad.zero_()

print(f"w: {new_w}  b: {new_b} \n")

    