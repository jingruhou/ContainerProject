# coding=utf-8
import os
import random

# 标注文件路径 /home/caffe/data/VOCdevkit/TV_LOGO_Dataset
xmlfilepath = r"/home/data/VOCdevkit/TV_LOGO_Dataset/Annotations"
saveBasePath = r"/home/data/VOCdevkit/TV_LOGO_Dataset/ImageSets/Main"

# 数据拆分比例
trainval_percent = 0.8
train_percent = 0.8
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
# 训练集和验证集数量
tv = int(num * trainval_percent)
# 训练集数量
tr = int(tv * train_percent)

trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("train size", tr)

ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')  # 训练集和验证集
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')  # 测试集
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')  # 训练集
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')  # 验证集

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:  # 写入训练集和验证集
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:  # 写入测试集
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
