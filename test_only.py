import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from model import Siamese
import numpy as np
import gflags
import sys
import os
from PIL import Image
import pandas as pd
from sklearn import preprocessing

if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_string("data_path", "genuines", 'path of data folder')
    gflags.DEFINE_string("gpu_ids", "0", "gpu ids used to train")

    Flags(sys.argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    print("use gpu:", Flags.gpu_ids, " for inference.")

    # loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    net = Siamese()
    net.load_state_dict(torch.load('models/model-inter-4001.pt'))
    net.eval()
    net.cuda()
    allFiles = os.listdir(Flags.data_path)[-204:]
    NUM_FILES = len(allFiles)
    results = pd.DataFrame(0, index=np.arange(20910), columns=['PERSON_ID_1','SIG_ID_1','PERSON_ID_2','SIG_ID_2','SCORE'])

    def getImage2Tensor(filePath):
        img = Image.open(filePath).convert('L')
        img = img.resize((128,128), Image.BILINEAR)
        trans = ToTensor()
        img = trans(img).unsqueeze(0).cuda()
        return img

    i = 0
    for idx1, sampleFile1 in enumerate(allFiles):
        filePath1 = os.path.join(Flags.data_path, sampleFile1)
        person_id_1 = sampleFile1[4:7]
        sig_id_1 = sampleFile1[7:9]
        img1 = getImage2Tensor(filePath1)
        for idx2 in range(idx1+1, NUM_FILES):
            sampleFile2 = allFiles[idx2]
            filePath2 = os.path.join(Flags.data_path, sampleFile2)
            person_id_2 = sampleFile2[4:7]
            sig_id_2 = sampleFile2[7:9]
            img2 = getImage2Tensor(filePath2)
            score = net.forward(img1,img2) 
            results.iloc[i] = ([person_id_1, sig_id_1, person_id_2, sig_id_2, score.item()])
            i += 1

    x = results['SCORE'].values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    results_scaled = min_max_scaler.fit_transform(x)
    results['SCORE'] = pd.DataFrame(results_scaled)
    results.to_csv('results.csv', index = False)