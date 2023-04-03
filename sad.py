import cv2 as cv
import matplotlib.pyplot as plt
import os
import random
from joblib import Parallel, delayed
import numpy as np

def getSad(im_pt1, im_pt2):
    img1 = cv.imread(im_pt1, 0)
    img2 = cv.imread(im_pt2, 0)

    assert(img2.shape == img1.shape)
    ws = 5
    sad = 0
    for i in range(0,img1.shape[0],ws):
        for j in range(0,img2.shape[1],ws):
            # 5x5 window size
            r1 = min(i+ws, img1.shape[0])
            c1 = min(j+ws, img1.shape[1])
            for k in range(i,r1):
                for l in range(j,c1):
                    sad += abs(int(img1[k][l]) - int(img2[k][l]))
    return(sad/(896**2)*10)

db_root = 'features/m1/'

def getIntraClassSad():
    dist = {}
    ofile = open('IntraClassSad.txt','w+')
    for i in range(13):
        dir = f'00{i}'
        dir = dir[-3:]
        files = os.listdir(db_root+dir)
        sads = []
        count=0
        for j in range(len(files)-1):
            sads.append(Parallel(n_jobs=4)(delayed(getSad)(f'{db_root}{dir}/{files[j]}', f'{db_root}{dir}/{files[k]}') for k in range(j+1,len(files))))
        for j in range(len(files)-1):
            for k in range(j+1, len(files)):
                sad = sads[j][k-(j+1)]
                ofile.write(f'{dir}/{files[j]},{dir}/{files[k]},{sad}\n')
                if(int(sad) in dist):
                    dist[int(sad)]+=1
                else:
                    dist[int(sad)]=1
    # plt.bar(dist.keys(), dist.values())
    # plt.xlabel('SAD')
    # plt.ylabel('Frequence')
    # plt.show()
    ofile.close()
    return dist
    

def getInterClassSad():
    '''
    For all pairs of classes choose min(20, possible) pairs of images randomly
    '''
    dist = {}
    ofile = open('InterClassSad.txt','w+')
    for i in range(12):
        for j in range(i+1,13):
            print('Class ',i,j)
            dir1 = f'00{i}'[-3:]
            dir2 = f'00{j}'[-3:]
            files1 = os.listdir(db_root+dir1)
            files2 = os.listdir(db_root+dir2)
            n1 = len(files1)
            n2 = len(files2)
            
            sads = []
            count=0
            for k1 in range(n1):
                sads.append(Parallel(n_jobs=8)(delayed(getSad)(f'{db_root}{dir1}/{files1[k1]}', f'{db_root}{dir2}/{files2[k2]}') for k2 in range(n2)))
            
            for k1 in range(n1):
                for k2 in range(n2):
                    sad = sads[k1][k2]
                    ofile.write(f'{dir1}/{files1[k1]},{dir2}/{files2[k2]},{sad}\n')
                    if(int(sad) in dist):
                        dist[int(sad)]+=1
                    else:
                        dist[int(sad)]=1
    ofile.close()
    return dist
    # plt.bar(dist.keys(), dist.values())
    # plt.xlabel('SAD')
    # plt.ylabel('Frequence')
    # plt.title('Inter Class SAD distribution')
    # plt.show()


def PlotSadDistribution():
    intra_dist = {}
    inter_dist = {}
    intra = open('IntraClassSad.txt','r')
    for line in intra:
        sad = int(float(line.split(',')[-1])*2)/2
        if(sad in intra_dist):
            intra_dist[sad]+=1
        else:
            intra_dist[sad]=1
    intra.close()

    inter = open('InterClassSad.txt','r')
    for line in inter:
        sad = int(float(line.split(',')[-1])*2)/2
        if(sad in inter_dist):
            inter_dist[sad]+=1
        else:
            inter_dist[sad]=1
    inter.close()

    plt.bar(intra_dist.keys(), intra_dist.values(), width=0.25, label ='Intra', color='g')
    X = np.array(list(inter_dist.keys()),dtype=np.float16)
    X = X+0.75
    Y = np.array(list(inter_dist.values()))
    Y = Y/10
    plt.bar(X, Y, width=0.25, label='Inter', color='orange')
    plt.legend()
    plt.xlabel('SAD')
    plt.ylabel('Normalized frequency')
    plt.title('SAD distribution')
    plt.show()

def generateDataset(thresh=5):
    intra = open('IntraClassSad.txt','r')
    pos = open('pos_dataset.txt','r')
    for line in intra:
        l = line.split()
        dir = l[0]
        k1 = l[1]
        k2 = l[2]
        sad = float(l[-1])
        if(sad<5):
            pos.write(line)
    intra.close()
    neg = open('neg_data.txt','r')
    inter = open('InterClassSad.txt','r')
    for line in inter:
        sad = float(line.split()[-1])
        if(sad<5):
            neg.write(line)


def rename():
    inter = open('InterClassSad.txt','r')
    inter1 = open('InterClassSad1.txt','w')
    for i in range(12):
        for j in range(i+1,13):
            dir1 = f'00{i}'[-3:]
            dir2 = f'00{j}'[-3:]
            files1 = os.listdir(db_root+dir1)
            files2 = os.listdir(db_root+dir2)
            n1 = len(files1)
            n2 = len(files2)
            for k1 in range(n1):
                for k2 in range(n2):
                    sad = float(inter.readline().split(',')[-1])
                    inter1.write(f'{dir1}/{files1[k1]},{dir2}/{files2[k2]},{sad}\n')
    inter1.close()
    intra = open('IntraClassSad.txt','r')
    intra1 = open('IntraClassSad1.txt','w')
    for i in range(13):
        dir = f'00{i}'
        dir = dir[-3:]
        files = os.listdir(db_root+dir)
        for j in range(len(files)-1):
            for k in range(j+1,len(files)):
                sad = float(intra.readline().split(',')[-1])
                intra1.write(f'{dir}/{files[j]},{dir}/{files[k]},{sad}\n')
    intra1.close()
                    
#rename()     
# intra_dist = getIntraClassSad()
# inter_dist = getInterClassSad()
db_root = 'features/m1/'
print(getSad(f'{db_root}005/23rfb.png',f'{db_root}005/36rfb.png'))