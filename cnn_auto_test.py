import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch as tch
from torch.utils.data import *
import torch.nn as nn
from torchvision import transforms as tran

import kagglehub
from PIL import Image    # 这才是正确写法!!!

FOLDER_DIR = "C:\\Users\\wang\\Desktop\\学习文件"

# 改变下载路径
os.environ["KAGGLEHUB_CACHE"] = FOLDER_DIR



# ================================ 1. 数据准备 ==================================


def verify_image(root_dir) :
    classes = ["Cat" , "Dog"] # 文件夹名称
    samples = []
    cls_idx = {"Cat" : 0 , "Dog" : 1}
    real_dir = os.path.join(root_dir , "PetImages")
    '''kaggle 的下载直接返回根目录, 可以直接使用
    注意,路径字符串反斜杠开头算绝对路径 '''
    for cls_name in classes :
        cls_dir = os.path.join(real_dir , cls_name)
        # path是一个模块,包含了各种处理路径的函数
        # 不建议用原生.join处理路径,实在不行pathlib
        for fname in os.listdir(cls_dir) :
            
            if not fname.lower().endswith((".jpg" , ".jpeg" , ".png")) :
                continue
            path = os.path.join(cls_dir , fname)

            try:
                with Image.open(path) as image:
                    image.verify()
                samples.append((path , cls_idx[cls_name]))
            except Exception:
                print(f"Traceback {path}: 图片无法打开")
    return samples

# ============================== 2. 数据集,模型建立 ==================================


class DandC_Dataset(Dataset) :

    def __init__(self , samples , transform = None):
        # 直接用samples
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        path , lable = self.samples[idx]
        with Image.open(path) as img:
            # Image.open 和 open 的区别

            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        
        return img , lable
    
    def __len__(self) :
        return len(self.samples)


class CNN(nn.Module):

    def __init__(self):
        super(CNN , self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=16 , kernel_size=3 , padding=1 , device="cpu" , bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2) ,

            nn.Conv2d(in_channels = 16 , out_channels=32 , kernel_size=3 , padding=1 , device="cpu" , bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128 , out_channels=1 , kernel_size=1 , device="cpu" , bias=False),
            #nn.AvgPool2d(1 , 1) ,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten() ,
            nn.Sigmoid()

        )

    def forward(self , x):
        return self.model(x)

# 检验函数(不是优化函数)

def evaluate(model , data_loader) :
    total_test = 0
    correct_test = 0

    with tch.no_grad():

        for image , lable in data_loader:
            image = image.to(DEVICE)
            lable = lable.float().unsqueeze(1).to(DEVICE)

            output = model(image)

            pred = (output > 0.5).float()
            total_test += lable.size(0)
            correct_test += (pred == lable).sum().item()
    
    return correct_test / total_test



# =========================== 3. 划分数据集,loader初始化 ==================================


if __name__ == "__main__" :
    '''由于WINDOWS没有fork机制,在使用多进程时会导入主模块,如果没有主进程保护,会引发无限递归'''

    print("1. 数据准备")

    # Download latest version
    path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")

    print("Path to dataset files(cats VS dogs):", path , kagglehub.__version__)

    DEVICE = tch.device("cuda" if tch.cuda.is_available() else "cpu")
    LR = 0.001
    EPOCH_NUM = 10 
    PRINT_STEP = 100
    BATCH_SIZE = 64
    IMG_SIZE = 128

    print("2. 数据集,模型建立")

    data_transform = tran.Compose([
        tran.Resize((IMG_SIZE , IMG_SIZE)) ,                      # 设置好尺寸
        tran.ToTensor() ,
        tran.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("3. 划分数据集,loader初始化")

    all_sample = verify_image(path)

    all_dataset = DandC_Dataset(all_sample , data_transform)
    train_size = int(len(all_sample) * 0.8)
    validation_size = len(all_sample) - train_size

    train_dataset, validation_dataset = random_split(all_dataset , [train_size , validation_size])
    train_loader = DataLoader(train_dataset , BATCH_SIZE , shuffle=True , num_workers=4)
    validation_loader = DataLoader(validation_dataset , BATCH_SIZE , shuffle=False , num_workers=4)
    #num_worker是多进程,通过多核CPU预处理减少主线等待

    model = CNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = tch.optim.Adam(model.parameters(), lr=LR)


# ================================ 4. 训练迭代 ==================================

    print("4. 训练迭代")

    for epoch in range(0 , EPOCH_NUM) :

        print(f"\nEpoch {epoch + 1}/{EPOCH_NUM}")
        running_loss = 0.0

        for step , (input , lable) in enumerate(train_loader) :

            input = input.to(DEVICE)
            lable = lable.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output , lable)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % PRINT_STEP == 0:
                avg_loss = running_loss / PRINT_STEP
                print(f"  Step [{step + 1}] - Loss: {avg_loss:.4f}")
                running_loss = 0.0

        val_acc = evaluate(model, validation_loader)
        print(f"Validation Accuracy after epoch {epoch + 1}: {val_acc:.4f}")


