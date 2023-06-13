import os
import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None):
        self.phase = phase 
        self.transform = transform 
        self.raf_path = raf_path
        
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        
        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]    
        else:
            self.data = df[df['name'].str.startswith('test')]
   
        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        _, self.sample_counts = np.unique(self.label, return_counts=True)

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned/', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)#返回文件地址个数，即训练或测试集个数

    def __getitem__(self, idx):
        path = self.file_paths[idx] #idx索引的文件地址 
        image = Image.open(path).convert('RGB')#以RGB的方式，读取文件
        label = self.label[idx] #idx索引的表情标签

        if self.transform is not None:
            image = self.transform(image) #转换图片(旋转、平移、、、)
        
        return image, label #返回图像和标签

class AffDataSet(data.Dataset):
    def __init__(self, aff_path, phase, transform = None):
        self.phase = phase #判断训练或测试
        self.transform = transform #数据增强
        self.aff_path = aff_path #读取地址
       
        df = pd.read_csv(os.path.join(self.aff_path, 'EmoLabel/affectnet_new_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        
        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]    
        else:
            self.data = df[df['name'].str.startswith('test')]
   
        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1  

        _, self.sample_counts = np.unique(self.label, return_counts=True)

        self.file_paths = []
        for f in file_names:
            path = os.path.join(self.aff_path, 'Image/', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)#返回文件地址个数，即训练或测试集个数

    def __getitem__(self, idx):
        path = self.file_paths[idx] #idx索引的文件地址 
        image = Image.open(path).convert('RGB')#以RGB的方式，读取文件
        label = self.label[idx] #idx索引的表情标签

        if self.transform is not None:
            image = self.transform(image) #转换图片(旋转、平移、、、)
        
        return image, label #返回图像和标签

class AffDataSet_BA(data.Dataset):
    def __init__(self, aff_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.aff_path = aff_path
        
        df = pd.read_csv(os.path.join(self.aff_path, 'EmoLabel/affectnet_new_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        
        train_lab = [0,1,2,3,4,5,6]
        df1=df[(df['label']==(train_lab[0]+1))].sample(frac=1.0,replace=False)
        df2=df[(df['label']==(train_lab[1]+1))].sample(frac=1.0,replace=False)
        df3=df[(df['label']==(train_lab[2]+1))].sample(frac=1.0,replace=False)
        df4=df[(df['label']==(train_lab[3]+1))].sample(frac=0.1,replace=False)
        df5=df[(df['label']==(train_lab[4]+1))].sample(frac=0.5,replace=False)
        df6=df[(df['label']==(train_lab[5]+1))].sample(frac=0.5,replace=False)
        df7=df[(df['label']==(train_lab[6]+1))].sample(frac=0.2,replace=False)
        df = pd.concat([df1,df2,df3,df4,df5,df6,df7])
        df = df.reset_index(drop=True)

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]    
        else:
            self.data = df[df['name'].str.startswith('test')]
   
        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1  
        _, self.sample_counts = np.unique(self.label, return_counts=True)

        self.file_paths = []
        for f in file_names:
            path = os.path.join(self.aff_path, 'Image/', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx] 

        if self.transform is not None:
            image = self.transform(image) 
        
        return image, label

class Raf_increData(data.Dataset):
    def __init__(self, raf_path, order, iter,frac1,phase, transform = None):
        
        self.phase = phase 
        self.transform = transform 
        self.raf_path = raf_path 
        
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        if iter<2:
            train_lab1 = order[iter][0]
            train_lab2 = order[iter][1]
            df1 = df[(df['label']==(train_lab1+1))].sample(frac=frac1[iter],replace=False)
            df2 = df[(df['label']==(train_lab2+1))].sample(frac=frac1[iter],replace=False)
            df = pd.concat([df1,df2])
        else:
            train_lab1 = order[iter][0]
            train_lab2 = order[iter][1]
            train_lab3 = order[iter][2]
            df1 = df[(df['label']==(train_lab1+1))].sample(frac=frac1[iter],replace=False)
            df2 = df[(df['label']==(train_lab2+1))].sample(frac=frac1[iter],replace=False)
            df3 = df[(df['label']==(train_lab3+1))].sample(frac=frac1[iter],replace=False)
            df = pd.concat([df1,df2,df3])

        df = df.reset_index(drop=True)

        
        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]    
        else:
            self.data = df[df['name'].str.startswith('test')]
   
        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1  
        
        _, self.sample_counts = np.unique(self.label, return_counts=True)

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned/', f)
            self.file_paths.append(path)
       
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx] 
        image = Image.open(path).convert('RGB')
        label = self.label[idx] 

        if self.transform is not None:
            image = self.transform(image) 
        
        return image, label 

class Aff_increData(data.Dataset):
    def __init__(self, aff_path, order, iter,frac1, phase, transform = None):
        
        self.phase = phase 
        self.transform = transform 
        self.aff_path = aff_path 
        
        df = pd.read_csv(os.path.join(self.aff_path, 'EmoLabel/affectnet_new_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        if iter<2:
            train_lab1 = order[iter][0]
            train_lab2 = order[iter][1]
            df1 = df[(df['label']==(train_lab1+1))].sample(frac=frac1[iter],replace=False)
            df2 = df[(df['label']==(train_lab2+1))].sample(frac=frac1[iter],replace=False)
            df = pd.concat([df1,df2])
            
        else:
            train_lab1 = order[iter][0]
            train_lab2 = order[iter][1]
            train_lab3 = order[iter][2]
            df1 = df[(df['label']==(train_lab1+1))].sample(frac=frac1[iter],replace=False)
            df2 = df[(df['label']==(train_lab2+1))].sample(frac=frac1[iter],replace=False)
            df3 = df[(df['label']==(train_lab3+1))].sample(frac=frac1[iter],replace=False)
            df = pd.concat([df1,df2,df3])

        df = df.reset_index(drop=True)
        
        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]    
        else:
            self.data = df[df['name'].str.startswith('test')]
   
        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1  
        
        _, self.sample_counts = np.unique(self.label, return_counts=True)
        
        self.file_paths = []
        for f in file_names:
            path = os.path.join(self.aff_path, 'Image/', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        path = self.file_paths[idx] 
        image = Image.open(path).convert('RGB')
        label = self.label[idx] 

        if self.transform is not None:
            image = self.transform(image) 
        
        return image, label 

#----------------------------------------------------------
""" class Raf_incre_wholevalData(data.Dataset):
    def __init__(self, raf_path, order, iter, transform = None):
        self.phase = 'test' #判断训练或测试
        self.transform = transform #数据增强
        self.raf_path = raf_path #读取地址
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), 
                        sep=' ', header=None,names=['name','label'])

    def __len__(self):
        return len(self.file_paths)#返回文件地址个数，即训练或测试集个数

    def __getitem__(self, idx):
        path = self.file_paths[idx] #idx索引的文件地址 
        image = Image.open(path).convert('RGB')#以RGB的方式，读取文件
        label = self.label[idx] #idx索引的表情标签

        if self.transform is not None:
            image = self.transform(image) #转换图片(旋转、平移、、、)
        
        return image, label #返回图像和标签 """