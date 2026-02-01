from __future__ import print_function
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#import hdf5storage
from numpy import reshape
import scipy.io
from matplotlib import pyplot as plt
from pathlib import Path

DATA_DIR = Path("Preprocessed Data")

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 256
img_cols = 256

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    #model.compile(optimizer=Adam(lr=1e-5), loss='mse')
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    
    
    imgs_train = np.load(DATA_DIR / 'data_train_b_full.npy')
    imgs_mask_train = np.load(DATA_DIR / 'data_mask_train_b_full.npy')

    imgs_test = np.load(DATA_DIR / 'data_test_b_full.npy')
    imgs_mask_test = np.load(DATA_DIR / 'data_mask_test_b_full.npy')
    
    imgs_train = imgs_train.astype('float32')
    imgs_test = imgs_test.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_test = imgs_mask_test.astype('float32')
    
    #imgs_mask_train /= 255.  # scale masks to [0, 1]
    #imgs_mask_test /= 255.  # scale masks to [0, 1]
    

    #print(imgs_mask_train.shape)
    #print(imgs_train.shape, imgs_mask_train.shape, imgs_test.shape, imgs_mask_test.shape)

    imgs_train = imgs_train.reshape(imgs_train.shape[0], img_rows, img_cols, 1)
    imgs_mask_train = imgs_mask_train.reshape(imgs_mask_train.shape[0], img_rows, img_cols, 1)
    imgs_test = imgs_test.reshape(imgs_test.shape[0], img_rows, img_cols, 1)
    
    print(imgs_train.shape, imgs_mask_train.shape, imgs_test.shape, imgs_mask_test.shape)
    print(type(imgs_train), type(imgs_mask_train), type(imgs_test), type(imgs_mask_test))

    #print(imgs_train.dtype)
    #print(imgs_mask_train.dtype)

    '''for index in range(0, 10):
        plt.figure(1)
        plt.imshow(imgs_train[index, :, :, 0], cmap='gray')
        plt.figure(2)
        plt.imshow(imgs_mask_train[index, :, :, 0], cmap='gray')
        plt.pause(1)

    plt.show()'''

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    epoch = 300
    history = model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=epoch, verbose=1, shuffle=True,
              validation_split=0.3,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)

    #np.save('BT_'+str(epoch)+'_unet10_data2_seg.npy', imgs_mask_test)
    # np.save("100_SAR1_unet.csv", imgs_mask_test)
    dict = {}
    dict['imgs_mask_test'] = imgs_mask_test
    #scipy.io.savemat('BT_'+str(epoch)+'_unet10_data1_seg.mat', dict)
    
    np.save('Shoulder_'+str(epoch)+'_unet.npy', imgs_mask_test)


    
    '''# summarize history for accuracy
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model accuracy')
    plt.ylabel('Dice')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()'''
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    resolution_value = 1080
    plt.savefig('Shoulder_'+str(epoch)+'_unet.png',dpi=resolution_value)
    plt.show()    


if __name__ == '__main__':
    train_and_predict()