import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image

class SignatureTrain(Dataset):

    def __init__(self, dataPath, transform=None):
        super(SignatureTrain, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas = self.loadToMem(dataPath)
        self.num_classes = len(self.datas)

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        # datas = {}
        # idx = 1
        # datas[idx] = []
        trainFiles = os.listdir(dataPath)[0:-204]
        self.person_ids = np.unique([x[4:7] for x in trainFiles])
        datas = {new_list: [] for new_list in self.person_ids} 
        for samplePath in trainFiles:
            person_id = samplePath[4:7] # Person ID
            sig_id = samplePath[7:9] #signature ID
            filePath = os.path.join(dataPath, samplePath)
            img = Image.open(filePath).convert('L')
            img_resized = img.resize((128,128), Image.BILINEAR)
            datas[person_id].append(img_resized)
        print("finish loading training dataset to memory")
        return datas

    def __len__(self):
        return  21000000

    def __getitem__(self, index):
        label = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            rand_person_idx = random.choice(self.person_ids)
            image1 = random.choice(self.datas[rand_person_idx])
            image2 = random.choice(self.datas[rand_person_idx])
        # get image from different class
        else:
            label = 0.0
            (rand_person_idx1, rand_person_idx2) = random.sample(self.person_ids.tolist(),2)
            # idx2 = random.randint(0, self.num_classes - 1)
            # while idx1 == idx2:
            #     idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[rand_person_idx1])
            image2 = random.choice(self.datas[rand_person_idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))

class SignatureTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=20):
        np.random.seed(1)
        super(SignatureTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        # self.img1 = None
        # self.c1 = None
        self.datas = self.loadToMem(dataPath)
        self.num_classes = len(self.datas)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        testFiles = os.listdir(dataPath)[-204:]
        self.person_ids = np.unique([x[4:7] for x in testFiles])
        datas = {new_list: [] for new_list in self.person_ids}
        for samplePath in testFiles:
            person_id = samplePath[4:7]
            filePath = os.path.join(dataPath, samplePath)
            img = Image.open(filePath).convert('L')
            img_resized = img.resize((128,128), Image.BILINEAR)
            datas[person_id].append(img_resized)
        print("finish loading test dataset to memory")
        return datas

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        # label = None
        # generate image pair from same class
        if idx == 0:
            rand_person_idx = random.choice(self.person_ids)
            img1 = random.choice(self.datas[rand_person_idx])
            img2 = random.choice(self.datas[rand_person_idx])
        # generate image pair from different class
        else:
            (rand_person_idx1, rand_person_idx2) = random.sample(self.person_ids.tolist(),2)
            img1 = random.choice(self.datas[rand_person_idx1])
            img2 = random.choice(self.datas[rand_person_idx2])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2

# test
if __name__=='__main__':
    SignatureTrain = SignatureTrain('./genuines')
    print(SignatureTrain)
