import pandas as pd
import glob
import tqdm  # tqdm是一个可以显示进度条的模块
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt



import random
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.models as models
import torchvision.transforms as transforms


data_dir = {
    'train_path' : 'train.csv',
    'test_path' : 'test.csv',
    'image_path':'images/',
    'submit_path' : 'sample_submission.csv'
}

#看文件的信息

train_data = pd.read_csv(data_dir['train_path'], header=None)
                                                 # header=None是去掉表头部分.   索引需要减一
                                                 # 索引: [第n列][第n行]  ,[:][0] = image,label
test_data = pd.read_csv(data_dir['test_path'], header=None)

# str(label) to number

label_sort = sorted(list(set(train_data[1][1:])))  # 除去标签'label', 避免t >= 0 && t < n_classes failed
n_classes = len(label_sort)
class_to_num = dict(zip(label_sort, range(n_classes)))
# number to str
num_to_class = {v : k for k, v in class_to_num.items()}

## 0.切分数据集
                # train :0.8 , val :0.2
index=[i for i in range(len(train_data[1:]))]
# random.shuffle(index)   # 打乱
Train_Set = np.asarray(train_data[1:].iloc[index[:int(len(index)*0.8)]])
Train_Set_img = Train_Set[:,0]
Train_Set_label = Train_Set[:,1]
Val_Set = np.asarray(train_data[1:].iloc[index[int(len(index)*0.8):]])
Val_Set_img = Val_Set[:,0]
Val_Set_label = Val_Set[:,1]
Test_Set_img = np.asarray(test_data.iloc[1:,0])
Test_Set_label = torch.tensor([1]*len(Test_Set_img))   # TypeError: unhashable type: 'list'
    # 数据框DataFrame的索引方式：.iloc[index,:],其中index是索引位置
# imshow()


## 1.定义读取数据集
class ClassDataset(Dataset):

    def __init__(self, mode, data_path, data_label, transform=None):
        self.mode = mode
        self.img_path = data_path
        self.img_label = data_label

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None


    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.mode == 'test':
            return img, self.img_label

        else:
            # get img_str_label
            label_idx = self.img_label[index]
            # get img_str_label -> num_label
            label = class_to_num[label_idx]

            return img, label

    def __len__(self):
        return len(self.img_path)


## 2.自定义读取数据dataloader

train_loader = torch.utils.data.DataLoader(
    ClassDataset('train', Train_Set_img, Train_Set_label,
                 transforms.Compose([
                     transforms.Resize((224,224)),
                     transforms.RandomHorizontalFlip(p=0.5),   #随机水平翻转 选择一个概率
                     # transforms.RandomCrop((60, 120)), # 随机剪裁
                     # transforms.ColorJitter(0.3, 0.3, 0.2), # 修改亮度、对比度和饱和度
                     transforms.RandomRotation(5), # 依degrees 随机旋转一定角度
                     transforms.ToTensor(),
                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      # Normalize(mean, std)按通道进行标准化，即先减均值，再除以标准差std
                 ])),
    batch_size=32,
    shuffle=False
)

val_loader = torch.utils.data.DataLoader(
    ClassDataset('val', Val_Set_img, Val_Set_label,
                 transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
    batch_size=32,
    shuffle=False,
)

test_loader = torch.utils.data.DataLoader(
    ClassDataset('test', Test_Set_img, Test_Set_label,
                 transforms.Compose([
                     transforms.Resize((224,224)),
                     # transforms.RandomCrop((60, 120)), # 随机剪裁
                     # transforms.ColorJitter(0.3, 0.3, 0.2), # 修改亮度、对比度和饱和度
                     # transforms.RandomRotation(10), # 依degrees 随机旋转一定角度
                     transforms.CenterCrop(size=224),
                     transforms.ToTensor(),
                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            # Normalize(mean, std)按通道进行标准化，即先减均值，再除以标准差std
                 ])),
    batch_size=32,
    shuffle=False,
)



## 3. 定义分类模型
   ## 3.1 ResNet18_model

class ResNet18_model(nn.Module):
    def __init__(self, num_class):
        super(ResNet18_model,self).__init__()
        # 继承父类的所有属性和方法, 并用父类的方法进行初始化
        model_conv = models.resnet18(pretrained=True)
            # 使用模型结构 ResNet18,并且 use预训练参数
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
            # 自适应平均池化，仅指定输出大小，特征数目不变
        model_conv = nn.Sequential(*list(model_conv.children())[:-1]) # 去除最后一个fc layer
            # 复制model_conv 的模型层次，仅使用预训练模型的一部分
        self.cnn = model_conv
        self.fc = nn.Linear(512, num_class)
            # full connect layer output has err: 11

    def forward(self, img):
        feat = self.cnn(img)
            # img进行cnn 网络提取featrue
        feat = feat.view(feat.shape[0], -1)
        feat = self.fc(feat)

        return feat

class ResNet50_model(nn.Module):
    def __init__(self, num_class):
        super(ResNet50_model,self).__init__()
        model_conv = models.resnet50(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        self.hd_fc = nn.Linear(2048, num_class)
        # self.dropout = nn.Dropout(0.2)
        # self.fc = nn.Linear(256, num_class)


    def forward(self,img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)

        feat = self.hd_fc(feat)
        # feat = self.dropout(feat)
        # out = self.fc(feat)

        return feat

class Mobilenet_model(nn.Module):
    def __init__(self, num_class):
        super(Mobilenet_model,self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(1280)
        # 使得每一层神经网络的输入保持相同分布的。
        self.fc1 = nn.Linear(1280, num_class)
        # self.dropout = nn.Dropout(0.2)self.fc = nn.Linear(256, )

    def forward(self,img):
        feat = self.cnn(img)
        feat = self.avgpool(feat)
        feat = feat.view(feat.shape[0], -1)

        feat = self.bn(feat)
        out = self.fc1(feat)
        # feat = self.dropout(feat)out = self.fc(feat)

        return out
# 设置训练

def train(train_loader, model, criterion, optimizer, plot_train_acc):
    model.train()
    train_loss = []
    train_accs = []

    for _, (input, target) in enumerate(tqdm.tqdm(train_loader)):

        if use_cuda:
            input = input.cuda()
            target = target.cuda()
                 # AttributeError: 'tuple'object has no attribute'cuda'

        prediction = model(input)
        loss = criterion(prediction, target)
                                 # 定义损失函数：criterion，
                                 # 然后通过计算网络真实输出和真实标签之间的误差，
                                 # 得到网络的损失值：loss
        optimizer.zero_grad()    # 梯度清零
        loss.backward()          # 计算loss梯度
        optimizer.step()
        acc = (prediction.argmax(dim=-1) == target).float().mean()
        train_accs.append(acc)
        train_loss.append(loss.item())
                                 # append() 方法用于在列表末尾添加新的对象
                                 # .item()方法 是得到一个元素张量里面的元素值
    train_acc = sum(train_accs) / len(train_accs)
    train_loss = np.mean(train_loss)
    plot_train_acc.append(train_acc)
    # print(f"Train_acc:{train_acc*100}")
    return train_loss, train_acc



def val(val_loader, model, criterion, plot_val_acc):
    model.eval()
    val_loss = []
    val_accs = []
    #仅正向传播
    with torch.no_grad():
        for _, (input, target) in enumerate(tqdm.tqdm(val_loader)):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            prediction = model(input)
            loss = criterion(prediction, target)
            acc = (prediction.argmax(dim=-1) == target).float().mean()
            val_accs.append(acc)
            val_loss.append(loss.item())
    val_acc = sum(val_accs) / len(val_accs)
    val_loss = np.mean(val_loss)
    plot_val_acc.append(val_acc)
    # print(f'val_acc:{val_acc*100}')
    return val_loss, val_acc


def predict(test_loader, model, tta=1):
    model.eval()
    test_pre_tta = None

    # TTA
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for _, (input, target) in enumerate(test_loader):

                if use_cuda:
                    input = input.cuda()

                prediction = model(input)

                if use_cuda:
                    output = prediction.data.cpu().numpy()
                else:
                    output = prediction.data.numpu()

                test_pred.append(output)

        test_pred = np.vstack(test_pred)
             # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
        if test_pre_tta is None:
            test_pre_tta = test_pred
        else:
            test_pre_tta += test_pred

    return test_pre_tta

###


# train and val
##########################################################################

#######################################
            ### config ###
lr = 0.001
model = Mobinet_model(n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr)
epochs = 160
TTA = 1
use_cuda = True
load_loss_pth = 'baseline_model.pth'
load_acc_pth = 'baseline_model.pth'
pth_path = 'pth/'
csv_name = 'submit_2021060601.csv'

#######################################

##########################################################################
# GPU or no

if use_cuda:
    model = model.cuda()
best_loss = 1000.0
best_acc = 0.0
plot_train_acc = []
plot_val_acc = []
plot_train_loss = []
plot_val_loss = []

for epoch in range(epochs):
    print(f"epoch:{epoch + 1}")
    (train_loss, train_acc) = train(train_loader, model, criterion, optimizer, plot_train_acc)
    (val_loss, val_acc) = val(val_loader, model, criterion, plot_val_acc)
    plot_train_loss.append(train_loss)
    plot_val_loss.append(val_loss)
    print(f'train_loss:{train_loss}, train_acc:{train_acc}')
    print(f'val_loss:{val_loss}, val_acc:{val_acc}')


    # # Val_ACC Calculation
    #     # 1. true_label
    #     # 2. pre_label
    #     # 3. judgment
    # val_true_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
    #     # . join()： 指定的字符(分隔符)连接生成字符串数组。
    #     #map(function, iterable)  function 应用于 iterable 中每一项并输出其结果的迭代器
    #
    # # str to num
    # val_true_label_num = []
    # for i in val_true_label:
    #     val_true_label_num.append(class_to_num[i])
    #
    # pre_label = predict(val_loader, model, TTA)
    # pre_label = pre_label.argmax(1)
    # val_pre_label = list(pre_label)
    # # val_pre_label = [''.join(map(str, x)) for x in pre_label]
    #
    # Val_Acc = np.mean(np.array(val_pre_label) == np.array(val_true_label_num))
    # print(f'train_loss:{train_loss}, val_loss:{val_loss}, pre_Val_Acc:{Val_Acc*100}')

    # record best value : min(loss)
    if val_loss < best_loss:
        best_loss = val_loss
        Saved_loss = pth_path + 'ResNet_1fc-Loss-%.2f.pth' % (val_loss)  # 'Resnet50-%d-Loss-%.2f-Acc-%.2f.pth' % (epoch + 1, val_loss, val_acc*100)
        torch.save(model.state_dict(), Saved_loss)
        print(f"Saved_loss:{Saved_loss}")
        load_loss_pth = Saved_loss
    if val_acc > best_acc:
        best_acc = val_acc
        Saved_acc = pth_path + 'ResNet_1fc-Acc-%.2f.pth' % (val_acc * 100)
        torch.save(model.state_dict(), Saved_acc)
        print(f"Saved_acc:{Saved_acc}")
        load_acc_pth = Saved_acc

                        # 模型保存与调用方式一：
                        # 保存：torch.save(model.state_dict(), mymodel.pth)#只保存模型权重参数，不保存模型结构
                        #
                        # 调用：model = My_model(*args, **kwargs)  #这里需要重新模型结构，My_model
                        #       model.load_state_dict(torch.load(mymodel.pth))#这里根据模型结构，调用存储的模型参数
                        #       model.eval()
                        #
                        # 模型保存与调用方式2：
                        # 保存：torch.save(model, mymodel.pth)#保存整个model的状态
                        #
                        # 调用：model=torch.load(mymodel.pth)#这里已经不需要重构模型结构了，直接load就可以
                        #      model.eval()
                        # 原文链接：https://blog.csdn.net/weixin_38145317/article/details/103582549
    if epoch % 10 == 0:
        # plot acc and loss
        plt.plot(plot_train_acc, ":", label="train_acc")
        plt.plot(plot_val_acc, ":", label="val_acc")
        plt.plot(plot_train_loss, label="train_loss")
        plt.plot(plot_val_loss, label="val_loss")
        plt.legend()
        plt.show()

# 5. predicted
    # load
model.load_state_dict(torch.load(load_loss_pth))

test_pre_label = predict(test_loader, model, TTA)
print('test_pre_label.shape:', test_pre_label.shape)

test_pre_label = np.vstack([test_pre_label.argmax(1)]).T
pre_label = []
for x in test_pre_label:
    pre_label.append(''.join(map(str, x)))  # 'numpy.int64' object is not iterable

# num to str
pre_label_submit = []
for i in pre_label:
    pre_label_submit.append(num_to_class[int(i)])

df_submit = pd.read_csv(data_dir['submit_path'])
df_submit['label'] = pd.Series(pre_label_submit)
df_submit.to_csv(csv_name, index=None)
print('Successful...')