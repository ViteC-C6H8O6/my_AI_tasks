import torch as tch
from torch.utils.data import *
import torch.nn as nn 
import torch.functional as F

class LeNet(nn.Module) :

    def __init__(self):
        super(LeNet , self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels = 6 , kernel_size=5 , device="cpu")
        # 激活
        self.pool1 = nn.AvgPool2d(kernel_size=2 , stride=2)

        self.conv2 = nn.Conv2d(in_channels=6 , out_channels=16 , kernel_size=5 , device="cpu")
        # 激活
        self.pool2 = nn.AvgPool2d(kernel_size=2 , stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=16 , out_channels=120 , kernel_size=5 , device="cpu")
        # 激活, 摊平
        self.fc1 = nn.Linear(in_features=120 , out_features=84)
        self.fc2 = nn.Linear(in_features=84 , out_features=10) 

        '''一般来说,非常不建议吧softmax等函数放在模型内部'''
        '''dataloader的循环看起来是逐条处理的,但实际上是在batch_size这个维度上批量处理的
        对于损失函数和和模型采取的都是这样的设计'''
        '''误区: 在批量计算里,batch_size替换的是0维, 而不是新开维度'''

    def forward(self , x) :
        
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)

        x = F.tanh(self.conv2(x))
        x = self.pool2(x)

        x = F.tanh(self.conv3(x))
        x = tch.flatten(x , 1)

        x = F.tanh(self.fc1(x))
        x = self.fc2(x)

        return x 
    
# critetion = softmax(LeNet.forward(x))
# loss = nn.CrossEntropyLoss()
    

class AlexNet(nn.Module) :

    def __init__(self , num_classes = 1000):
        super(AlexNet , self).__init__()
        # 卷积层定义
        self.features = nn.Sequential(
            nn.Conv2d(3 , 96 , kernel_size=11 , stride=4 , padding=2),
            nn.ReLU(inplace=False),   # inplace为False, 小于0的部分会被设置成一个表示max(x , 0)的输出张量
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0) ,#nn.BatchNorm2d(96)
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96 , 256 , kernel_size=5 , stride=1 , padding=2),
            nn.ReLU(inplace=False),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3 , stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 输出: 384x13x13
            nn.ReLU(inplace=False),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2)

        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6 , 4096) ,
            nn.ReLU(inplace=False),

            nn.Dropout(p=0.5),
            nn.Linear(4096 , 4096) ,
            nn.ReLU(inplace=False),

            nn.Linear(4096 , num_classes)
        )

    def forward(self , x) :
        x = self.features(x) 
        x = tch.flatten(x ,1)
        x = self.classifier(x)
        return x
    


class BasicBlock(nn.Module):

    # Standard ResNet BasicBlock (v1).
    # 两个 3x3 卷积，每个卷积后跟 BN 和 ReLU。若输入输出维度不一致，则在捷径路径使用 1x1 卷积。

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个Residual Block的卷积层可能输入和输出通道数不一致：
        # 对于Conv2，第一个Residual Block的卷积层输入和输出通道数一致。
        # 对于Conv3-5的第一个Residual Block：输入输出通道数不一致，则stride设置为2，达到同时减半高宽，翻倍通道数。
        # 第二个Residual Block的卷积层输入和输出通道数一致，stride设置为1，保持特征图尺寸一致。
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层的输入和输出通道数都是out_channels,stride=1,保证输入和输出特征图尺寸一致。
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入和输出特征图尺寸不一致，需要调用stride=2的1×1卷积进行特征图尺寸的调节。
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out





class Residual_Block(nn.Module) :

    def __init__(self , in_channels , mid_channels , mid_stride = 1 , expansion = 4 ,downsample = None):
        super(Residual_Block , self).__init__()

        out_channels = mid_channels * expansion

        self.conv1 = nn.Conv2d(in_channels , mid_channels , kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels , mid_channels , kernel_size=3 , stride=mid_stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels , out_channels , kernel_size=1)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        '''注意,expension是一个特定的属性,因为bottleNeck设定上就是先由输入端压缩通道,在最后一层扩展'''
        '''多个residual_block组成一个bottle_neck其中只有一个bottle_neck才触发expasion'''
    def forward(self , x) :
        identity = x

        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None :
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module) :

    def __init__(self , block , layer_sizes , expansion , num_classes):
        super().__init__() 
        self.layer_sizes = layer_sizes
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3 , 64 , kernel_size=7 , padding=3 , stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.mxpool1 = nn.MaxPool2d(kernel_size=3 , stride=2)

        self.layer1 = self._make_layer(block , 64 , 256 , layer_size=layer_sizes[0] , expasion=expansion)
        self.layer2 = self._make_layer(block , 256 , 512 , layer_size=layer_sizes[1] , stride=2 , expasion=expansion)
        self.layer3 = self._make_layer(block , 512 , 1024 , layer_size=layer_sizes[2] , stride=2 , expasion=expansion)
        self.layer4 = self._make_layer(block , 1024 , 2048 , layer_size=layer_sizes[3] , stride=2 , expasion=expansion)

        self.pool2 = nn.AdaptiveAvgPool2d((1 , 1))
        self.fc = nn.Linear(2048 , num_classes)



    def _make_layer(self , block , in_channels , out_channels , layer_size , stride = 1 , expasion = 4):
        # layer就是bottle_neck, 包含多个residual_block

        mid_channels = out_channels / expasion
        down_sample = None
        if stride != 1 or in_channels != out_channels :
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels , out_channels , kernel_size=1 , stride=stride) ,
                nn.BatchNorm2d(out_channels)
            )

        blocks = [block(in_channels , mid_channels , stride , expasion , down_sample)]

        for _ in range(1 , layer_size) :
            blocks.append(block(out_channels , mid_channels))

        return nn.Sequential(*blocks)
        

    def forward(self , x) :

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool2(x)
        x = tch.flatten(x , 1)
        x = self.fc(x)

        # x = softmax(x)
        

    
