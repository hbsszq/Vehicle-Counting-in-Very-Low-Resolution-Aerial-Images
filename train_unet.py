from unet_model import *
from gen_patches import *

import os
import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x



N_BANDS = 3  #N_BANDS = 8
N_CLASSES = 4 #3,2,1,4  #N_CLASSES = 5  # buildings, roads, trees, crops and water
CLASS_WEIGHTS = [0.3, 0.2, 0.4, 0.1]  #CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]
N_EPOCHS = 500
UPCONV = True
PATCH_SZ = 32   # should divide by 16
BATCH_SIZE = 32 #BATCH_SIZE = 150
TRAIN_SZ = 4000  # train size
VAL_SZ = 1000    # validation size


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

trainIds = [str(i).zfill(2) for i in range(3, 15)]  # all availiable ids: from "01" to "24"


if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()
    
    X_DICT_TRAIN2 = dict()
    Y_DICT_TRAIN2 = dict()
    X_DICT_VALIDATION2 = dict()
    Y_DICT_VALIDATION2 = dict()
    
    X_DICT_TRAIN3 = dict()
    Y_DICT_TRAIN3 = dict()
    X_DICT_VALIDATION3 = dict()
    Y_DICT_VALIDATION3 = dict()
    
    X_DICT_TRAIN4 = dict()
    Y_DICT_TRAIN4 = dict()
    X_DICT_VALIDATION4 = dict()
    Y_DICT_VALIDATION4 = dict()

    print('Reading images')
    for img_id in trainIds:
        img_m = normalize(tiff.imread('./data/mband/{}.tif'.format(img_id)).transpose([0,1, 2]))#[1, 2, 0]
        mask = tiff.imread('./data/gt_mband/{}.tif'.format(img_id)).transpose([0,1, 2]) / 255
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')

    for img_id in trainIds:
        img_m2 = normalize(tiff.imread('./data/mband2/{}.tif'.format(img_id)).transpose([0,1, 2]))#[1, 2, 0]
        mask2 = tiff.imread('./data/gt_mband2/{}.tif'.format(img_id)).transpose([0,1, 2]) / 255
        train_xsz2 = int(3/4 * img_m2.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN2[img_id] = img_m2[:train_xsz2, :, :]
        Y_DICT_TRAIN2[img_id] = mask2[:train_xsz2, :, :]
        X_DICT_VALIDATION2[img_id] = img_m2[train_xsz2:, :, :]
        Y_DICT_VALIDATION2[img_id] = mask2[train_xsz2:, :, :]
        print(img_id + ' read')
    print('Images were read')

    for img_id in trainIds:
        img_m3 = normalize(tiff.imread('./data/near/{}.tif'.format(img_id)).transpose([0,1, 2]))#[1, 2, 0]  
        mask3 = tiff.imread('./data/gt_mband2/{}.tif'.format(img_id)).transpose([0,1, 2]) / 255  
        train_xsz3 = int(3/4 * img_m3.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN3[img_id] = img_m3[:train_xsz3, :, :]
        Y_DICT_TRAIN3[img_id] = mask3[:train_xsz3, :, :]
        X_DICT_VALIDATION3[img_id] = img_m3[train_xsz3:, :, :]
        Y_DICT_VALIDATION3[img_id] = mask3[train_xsz3:, :, :]

        img_m4 = normalize(tiff.imread('./data/far/{}.tif'.format(img_id)).transpose([0,1, 2]))#[1, 2, 0]
        mask4 = tiff.imread('./data/gt_mband2/{}.tif'.format(img_id)).transpose([0,1, 2]) / 255      
        train_xsz4 = int(3/4 * img_m4.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN4[img_id] = img_m4[:train_xsz4, :, :]
        Y_DICT_TRAIN4[img_id] = mask4[:train_xsz4, :, :]
        X_DICT_VALIDATION4[img_id] = img_m4[train_xsz4:, :, :]
        Y_DICT_VALIDATION4[img_id] = mask4[train_xsz4:, :, :]


    def train_net():
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        
        x_train2, y_train2 = get_patches(X_DICT_TRAIN2, Y_DICT_TRAIN2, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val2, y_val2 = get_patches(X_DICT_VALIDATION2, Y_DICT_VALIDATION2, n_patches=VAL_SZ, sz=PATCH_SZ)

        x_train2n, y_train2n = get_patches(X_DICT_TRAIN3, Y_DICT_TRAIN3, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val2n, y_val2 = get_patches(X_DICT_VALIDATION3, Y_DICT_VALIDATION3, n_patches=VAL_SZ, sz=PATCH_SZ)

        x_train2f, y_train2f = get_patches(X_DICT_TRAIN4, Y_DICT_TRAIN4, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val2f, y_val2 = get_patches(X_DICT_VALIDATION4, Y_DICT_VALIDATION4, n_patches=VAL_SZ, sz=PATCH_SZ)
        
        y_sub=np.zeros((x_train.shape[0],1,1,1024))
        y_sub2=np.zeros((x_val.shape[0],1,1,1024))
        y_mul=np.ones((x_train.shape[0],1,1,1024))
        y_mul2=np.ones((x_val.shape[0],1,1,1024))
        
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet1127.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit([x_train,x_train2,x_train2n,x_train2f], [y_train,y_sub,y_sub], batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=([x_val,x_val2,x_val2n,x_val2f], [y_val,y_sub2,y_sub2]))
        return model

    train_net()
