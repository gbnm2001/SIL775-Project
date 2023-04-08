import cv2
import numpy as np
import time
from feature_extractor import *

skin_filter = input('Skin filter (y/n) = ')
recognize = input('Do recognition (y/n) = ')

model = None
if(recognize == 'y'):
    from feature_extractor import *
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    from tensorflow import convert_to_tensor
    from tensorflow import reshape
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
    print('Imported tensorflow')
    def prep_pixels(tensor_arr):
        # convert from integers to floats
        train_norm = tensor_arr.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        # return normalized images
        return train_norm

    def define_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(160, 280, 1)))#16
        model.add(MaxPooling2D((3, 3)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))#150
        model.add(BatchNormalization())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))
        return model

    model = define_model()
    model.load_weights('./tf_models/matcher3.pth')
    print('Loaded model')

nn_input = np.zeros((1,160,280))
if(recognize == 'y'):
    #load template
    temp_path = './template/gautam.png'
    e1 = cv2.imread(temp_path,cv2.IMREAD_GRAYSCALE)
    nn_input[0,:160,0:140] = e1

#face_classifier = cv2.CascadeClassifier('../cascade_files/haarcascade_frontalface_default.xml')
#eye_classifier = cv2.CascadeClassifier('../cascade_files/haarcascade_eye.xml')
left_ear_classifier = cv2.CascadeClassifier('../cascade_files/haarcascade_mcs_leftear.xml')
right_ear_classifier = cv2.CascadeClassifier('../cascade_files/haarcascade_mcs_rightear.xml')
ear_classifier = cv2.CascadeClassifier('../cascade_files/left_ear2.xml')

count = 10

def ear_detector(img, size=0.5):
    # Convert image to grayscale
    global count
    if(skin_filter=='y'):
        segmentMask(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #left = left_ear_classifier.detectMultiScale(gray, 1.05, 2)
    right = right_ear_classifier.detectMultiScale(gray, 1.05,2)
    left = ()
    if left == () and right==():
        return (img,None)
    
    for (x,y,w,h) in right:
        if(recognize=='y'):
            ear_img = img[x-5:x+w+5, y:y+h]
            nn_input[0,0:160,140:280] = getFeatures(ear_img, False, )
            X = convert_to_tensor(nn_input)
            X = prep_pixels(X)
            #print(X.shape)
            res = model.predict(X)
        if(count<10):
            cv2.imwrite(f'./template/gautam{count}.png', img[y:y+h,x-5:x+w+5])
            print(count)
            count+=1
        cv2.rectangle(img,(x-5,y),(x+w+5,y+h),(255,0,0),2)
    for (x,y,w,h) in left:
        cv2.rectangle(img,(x-5,y),(x+w+5,y+h),(0,255,0),2)
    return img,None

cap = cv2.VideoCapture(0)
if(skin_filter=='y'):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
    ret, frame = cap.read()
    full_img, ear_img = ear_detector(frame)
    if(ear_img == None):
        cv2.imshow('Our Face Extractor', full_img)
        time.sleep(0.5)
    else:
        pass
        
cap.release()
cv2.destroyAllWindows()    