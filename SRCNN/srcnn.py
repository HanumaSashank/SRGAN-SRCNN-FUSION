import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from skimage.transform import resize, rescale
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim

def model():
    net = Sequential()
    net.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',activation='relu', padding='same', use_bias=True, input_shape=(None,None, 1)))
    #hidden layer one with convolution kernal size of 9x9
    net.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',activation='relu', padding='same', use_bias=True))
    #hidden layer two with convolution kernal size of 3x3
    net.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',activation='linear', padding='same', use_bias=True))
    # used optimizer
    adam = Adam(lr=0.0003)
    # model compilation
    net.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    return net

srcnn = tf.keras.models.load_model('srcnn_model_4x_65epoch (1).h5')#loadind the weights for with the  model
# srcnn.summary()

otptfolder = "./output_images/"
inputfolder = "./input_images/"
os.makedirs(otptfolder, exist_ok=True)
imagelist = [s for s in os.listdir(inputfolder) if s.endswith(('.png','.jpg','.jpeg'))]

scalefactor = 2
for img in imagelist:
    img1 = cv2.imread(os.path.join(inputfolder, img), 3)#taking image input
    print("Initial image", img1.shape)
    width, height = img1.shape[0], img1.shape[1]
    img = img1
    img2 = img.astype(np.float32) / 255.0
    ycompCbCr = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)#converting it to ycrcb format
    ycomp = np.expand_dims(cv2.resize(ycompCbCr[:, :, 0], None, fx=scalefactor, fy=scalefactor, interpolation=cv2.INTER_CUBIC),axis=2)
    print("ycomp", ycomp.shape)
    iplr = ycomp.reshape(1, ycomp.shape[0], ycomp.shape[1], 1)
    Y = srcnn.predict([iplr])[0]

    #resizing the image for the output resolution and scale factor given
    a = np.expand_dims(cv2.resize(ycompCbCr[:, :, 1], None, fx=scalefactor, fy=scalefactor, interpolation=cv2.INTER_CUBIC),axis=2)
    b = np.expand_dims(cv2.resize(ycompCbCr[:, :, 2], None, fx=scalefactor, fy=scalefactor, interpolation=cv2.INTER_CUBIC),axis=2)
    HRycrbimg = np.concatenate((Y, a, b), axis=2)
    ophr = ((cv2.cvtColor(HRycrbimg, cv2.COLOR_YCrCb2BGR)) * 255.0).clip(min=0, max=255)
    ophr = (ophr).astype(np.uint8)
    output_path = os.path.join(otptfolder, f"srcnn_{img}")
    cv2.imwrite(output_path, ophr)
    print(f"High Resolution image saved {output_path}")

#for displaying the input and output of model side by side for analysis
input_imagelist = [f for f in os.listdir(inputfolder) if f.endswith(('.png','.jpg','.jpeg'))]
plt.figure(figsize=(20, 20))

for i, input_img in enumerate(input_imagelist):
    img1 = cv2.imread(os.path.join(inputfolder, input_img), 3)
    opimg = f"srcnn_{input_img}"  
    ophr = cv2.imread(os.path.join(otptfolder, opimg), 3)
    plt.subplot(len(input_imagelist), 2, 2 * i + 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("LR image")
    # print(img1.shape)

    plt.subplot(len(input_imagelist), 2, 2 * i + 2)
    plt.imshow(cv2.cvtColor(ophr, cv2.COLOR_BGR2RGB))
    plt.title("SRCNN Output image")
    # print(ophr.shape)
plt.show()
 
  
   
    
     
      
