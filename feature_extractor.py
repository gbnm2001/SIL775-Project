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

def segmentMask(image):
    '''
    INPUT: color image
    OUTPUT: Mask the region that is not of skin color
    ALGORITHM: Assume the middle of the image is of skin color, take the average value of the middle as reference
    Replace the blocks that have the average much different that the reference
    '''
    ws = 5
    rows, cols, _ = image.shape
    row_low = int(rows*0.2)
    row_hi = int(rows*0.75)
    col_low = int(cols*0.2)
    col_hi = int(cols*0.75)
    red = int(np.average(image[row_low:row_hi, col_low:col_hi,0:1]))
    blu = int(np.average(image[row_low:row_hi, col_low:col_hi,1:2]))
    gre = int(np.average(image[row_low:row_hi, col_low:col_hi,2:3]))
    
    
    for i in range(rows//ws):
        for j in range(cols//ws):
            if(row_low<i*ws and col_low<j*ws):
                continue
            r = np.average(image[i*ws:(i+1)*ws, j*ws:(j+1)*ws,0:1])
            g = np.average(image[i*ws:(i+1)*ws, j*ws:(j+1)*ws,1:2])
            b = np.average(image[i*ws:(i+1)*ws, j*ws:(j+1)*ws,2:3])
            if(abs(r-red)>70 or abs(b-blu)>70 or abs(g-gre)>70):
                image[i*ws:(i+1)*ws, j*ws:(j+1)*ws,0:1] = red
                image[i*ws:(i+1)*ws, j*ws:(j+1)*ws,1:2] = gre
                image[i*ws:(i+1)*ws, j*ws:(j+1)*ws,1:3] = blu

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

def edge1(image_path='ref2.png'):
    #read the image in grayscale
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (140,160))
    #segmentMask(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    grad_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    angles = np.zeros(grad_x.shape)
    for i in range(grad_x.shape[0]):
        for j in range(grad_x.shape[1]):
            angles[i][j] = int((math.atan2(grad_y[i][j], grad_x[i][j]) + math.pi)/(2*math.pi)*255)
    
    angles = cv2.resize(angles, (140,160))
    angles = angles.astype(np.uint8)

    gray_image = fuzzy_filter(gray_image)
    gray_image = gray_image.astype(np.uint8)
    #resized = cv2.resize(gray_image, (140,160))
    #extract the canny edges
    edges = cv2.Canny(gray_image,100,150,apertureSize=3)#img, lower threshold, upper threshold, aperture size, L2_gradient=False
    edges = edges[3:-3,3:-3]
    edges = cv2.resize(edges, (140,160))

    #Show the img
    # plt.subplot(121),plt.imshow(image)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    
    # Check whether the specified path exists or not
    filename = image_path.split('/')
    opathe = f"features/{dest}/edges/{filename[-2]}/"
    opatha = f"features/{dest}/angles/{filename[-2]}/"
    if not os.path.exists(opathe):
        # Create a new directory because it does not exist
        os.makedirs(opathe)
    if(os.path.exists(opatha) == False):
        os.makedirs(opatha)
    
    cv2.imwrite(f'{opathe}{filename[-1]}', edges)
    cv2.imwrite(f'{opatha}{filename[-1]}', angles)



dirs = ['000','001','002','003','004','005','006','007','008','009','010','011','012']
for dir in dirs:
    files = os.listdir(DB_ROOT+dir)
    #print(files)
    for file in files:
        edge1(DB_ROOT+dir+'/'+file)