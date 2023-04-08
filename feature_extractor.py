import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import math

train = False
dest = ''
if(train):
    DB_ROOT = "../ear_dataset/left/cropped/"
    dest = 'train'
else:
    DB_ROOT = '../ear_dataset/test/left_cropped/'
    dest='test'
#following is from https://docs.opencv2.org/4.x/da/d22/tutorial_py_canny.html

def getCr(r,g,b):
    return 128+(112.0*r - 93.786*g - 18.214*b)/256

def segmentMask(image):
    '''
    INPUT: color image
    OUTPUT: Mask the region that is not of skin color
    ALGORITHM: Assume the middle of the image is of skin color, take the average value of the middle as reference
    Replace the blocks that have the average much different that the reference
    '''
    ws = 5
    rows, cols, _ = image.shape
    for i in range(0,rows,ws):
        for j in range(0,cols,ws):
            if(i+ws>rows or j+ws>cols):
                continue
            b = np.average(image[i:i+ws, j:j+ws,0:1])
            g = np.average(image[i:i+ws, j:j+ws,1:2])
            r = np.average(image[i:i+ws, j:j+ws,2:3])
            cr = getCr(r,g,b)
            if(not (132<cr<176)):
                image[i:i+ws, j:j+ws,0:3] = 0

def fuzzy_filter(img):
    # block size = 3x3
    zeros = np.zeros((1, img.shape[1]))
    img = np.vstack((zeros, img))
    img = np.vstack((img, zeros))
    zeros = np.zeros((img.shape[0], 1))
    img = np.hstack((zeros, img))
    img = np.hstack((img, zeros))
    n, m = img.shape
    for i in range(1, n-1):
        for j in range(1, m-1):
            # perform ops on a 3x3 image
            # (i, j) is in the centre
            a = img[i-1][j-1]
            b = img[i-1][j]
            c = img[i-1][j+1]
            d = img[i][j+1]
            e = img[i+1][j+1]
            f = img[i+1][j]
            g = img[i+1][j-1]
            h = img[i][j-1]
            feat = [a, b, c, d, e, f, g, h]
            i_avg = np.mean(feat)
            i_min = min(feat)
            i_max = max(feat)
            # change the image pixel value
            if img[i][j] == i_max or img[i][j] == i_min:
                img[i][j] = 0
            elif img[i][j] == i_avg:
                img[i][j] = 1
            elif i_min < img[i][j] and img[i][j] < i_avg:
                img[i][j] = (img[i][j] - i_min)/(i_avg - i_min)
            elif i_avg < img[i][j] and img[i][j] < i_max:
                img[i][j] = (i_max - img[i][j])/(i_max - i_avg)
        return img

def getFeatures(image_path='ref2.png', show=False, edge_only=False):
    #read the image in grayscale
    getFeatures.image_count = 0
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (140,160))
    #segmentMask(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angles = None
    
    if(not edge_only):
        grad_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        angles = np.zeros(grad_x.shape)
        for i in range(grad_x.shape[0]):
            for j in range(grad_x.shape[1]):
                angles[i][j] = int((math.atan2(grad_y[i][j], grad_x[i][j]) + math.pi)/(2*math.pi)*255)
    
        #angles = cv2.resize(angles, (140,160))
        angles = angles.astype(np.uint8)

    gray_image = fuzzy_filter(gray_image)
    gray_image = gray_image.astype(np.uint8)
    #resized = cv2.resize(gray_image, (140,160))
    #extract the canny edges
    edges = cv2.Canny(gray_image,100,150,apertureSize=3)#img, lower threshold, upper threshold, aperture size, L2_gradient=False
    edges = edges[3:-3,3:-3]
    #edges = cv2.resize(edges, (140,160))

    #Show the img
    if(show):
        plt.figure('Image '+str(getFeatures.image_count))
        getFeatures.image_count+=1
        plt.subplot(121),plt.imshow(image)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show(block=False)
        plt.show()
    
    if(edge_only):
        return edges
    else:
        return edges, angles

def generateFeatures():
    dirs = ['000','001','002','003','004','005','006','007','008','009','010','011','012']
    for dir in dirs:
        files = os.listdir(DB_ROOT+dir)
        #print(files)
        for file in files:
            (edges, angles) = getFeatures(DB_ROOT+dir+'/'+file)
            opathe = f"features/{dest}/edges/{dir}/{file}"
            opatha = f"features/{dest}/angles/{dir}/{file}"
            if not os.path.exists(opathe):
                os.makedirs(opathe)
            if(os.path.exists(opatha) == False):
                os.makedirs(opatha)
            cv2.imwrite(opathe, edges)
            cv2.imwrite(opatha, angles)