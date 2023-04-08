import numpy as np
from matplotlib import pyplot as plt
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

while(True):
    im1 = input('Enter image path 1 : ')
    im2 = input('Enter image path 2 : ')
    (e1, a1) = getFeatures(im1, True)
    (e2, a2) = getFeatures(im2, True)
    X = np.concatenate((e1,e2),axis=1)
    X = convert_to_tensor(X)
    X = reshape(X, (1,160,280))
    X = prep_pixels(X)
    print(X.shape)
    res = model.predict(X)
    print(res)
    if(res[0][0]>= res[0][1]):
      print('Same person, confidence = ', res[0][0]-res[0][1])
    else:
      print('Different person, confidence = ', res[0][1] - res[0][0])
    if(input('Continue (y/n)') == 'n'):
        break
        
